import re
from itertools import count
from typing import List, Set, Tuple

import torch as t
from transformers import ConvNextV2ForImageClassification

from auto_circuit.types import DestNode, SrcNode


def simple_graph_nodes(
    model: ConvNextV2ForImageClassification,
) -> Tuple[Set[SrcNode], Set[DestNode]]:
    src_nodes = set()
    dest_nodes = set()
    layers, src_idx_ranks = count(), count()

    # Input layer
    src_nodes.add(
        SrcNode(
            name="Input",
            module_name="convnextv2.embeddings",
            layer=next(layers),
            global_rank=next(src_idx_ranks),
            stage=0,
        )
    )

    # ConvNext stages
    for stage_idx, stage in enumerate(model.convnextv2.encoder.stages):
        for sub_layer_idx, layer in enumerate(stage.layers):
            layer_idx = next(layers)

            dest_nodes.add(
                DestNode(
                    name=f"Stage{stage_idx}.Layer{sub_layer_idx}",
                    module_name=f"convnextv2.encoder.stages.{stage_idx}.layers.{sub_layer_idx}",
                    layer=layer_idx,
                    stage=stage_idx,
                    min_layer=layer_idx - 1,
                )
            )

            if (stage_idx < len(model.convnextv2.encoder.stages) - 1) or (sub_layer_idx < len(stage.layers) - 1):
                src_nodes.add(
                    SrcNode(
                        name=f"Stage{stage_idx}.Layer{sub_layer_idx}",
                        module_name=f"convnextv2.encoder.stages.{stage_idx}.layers.{sub_layer_idx}",
                        layer=layer_idx,
                        global_rank=layer_idx,
                        stage=stage_idx,
                    )
                )

    return src_nodes, dest_nodes


def factorized_src_nodes(
    model: ConvNextV2ForImageClassification,
) -> Set[SrcNode]:
    src_nodes = set()
    layers, src_idxs = count(), count()
    # Input layer
    src_nodes.add(
        SrcNode(
            name="Input",
            module_name="convnextv2.embeddings",
            layer=next(layers),
            global_rank=next(src_idxs),
            sublayer_shape=model.convnextv2.embeddings.patch_embeddings.out_channels,
            stage='0',
            head_dim=1,
        )
    )

    for stage_idx, stage in enumerate(model.convnextv2.encoder.stages):
        for sub_layer_idx, layer in enumerate(stage.layers):
            layer_idx = next(layers)
            for ch_idx in range(layer.pwconv2.out_features):
                src_nodes.add(
                    SrcNode(
                        name=f"Stage{stage_idx}.Layer{sub_layer_idx}.{ch_idx}",
                        module_name=f"convnextv2.encoder.stages.{stage_idx}.layers.{sub_layer_idx}.drop_path",
                        layer=layer_idx,
                        global_rank=next(src_idxs),
                        sublayer_shape=layer.pwconv2.out_features,
                        stage=str(stage_idx),
                        head_dim=1,
                        head_idx=ch_idx,
                    )
                )

    return src_nodes


def factorized_dest_nodes(
    model: ConvNextV2ForImageClassification,
) -> Set[DestNode]:
    dest_nodes = set()
    layers = count(start=1)

    for stage_idx, stage in enumerate(model.convnextv2.encoder.stages):
        for ref_layer_idx, layer in enumerate(stage.layers):
            layer_idx = next(layers)
            for ch_idx in range(layer.dwconv.in_channels):
                dest_nodes.add(
                    DestNode(
                        name=f"Stage{stage_idx}.Layer{ref_layer_idx}.{ch_idx}",
                        module_name=f"convnextv2.encoder.stages.{stage_idx}.layers.{ref_layer_idx}.dwconv",
                        layer=layer_idx,
                        stage=str(stage_idx),
                        head_dim=1,
                        head_idx=ch_idx,
                        sublayer_shape=layer.pwconv2.out_features,
                    )
                )

    layer_idx = next(layers)

    for ch_idx in range(model.convnextv2.layernorm.normalized_shape[0]):
        dest_nodes.add(
            DestNode(
                name=f"Layernorm.{ch_idx}",
                module_name="convnextv2.layernorm",
                layer=layer_idx,
                stage=len(model.convnextv2.encoder.stages),
                head_dim=1,
                head_idx=ch_idx,
                sublayer_shape=model.convnextv2.layernorm.normalized_shape[0],
            )
        )

    return dest_nodes


def get_downsample_modules(
    model: ConvNextV2ForImageClassification,
) -> List[t.nn.Module]:
    downsample_modules = []
    for name, module in model.named_modules():
        if "downsampling" in name and isinstance(module, t.nn.Conv2d):
            pattern = r"stages\.(\d+)"
            matches = re.findall(pattern, name)
            assert (
                len(matches) == 1
            ), f"Expected exactly one 'stages.X' pattern in {name}"
            stage = int(matches[0])
            downsample_modules.append((stage, module))

    stage_downsample_modules = [module for _, module in sorted(downsample_modules)]

    # Add a mean over the spatial dimensions for the final layernorm input
    class SpatialMean(t.nn.Module):
        def forward(self, x: t.Tensor) -> t.Tensor:
            return x.mean([-2, -1])

    stage_downsample_modules.append(SpatialMean())

    return stage_downsample_modules
