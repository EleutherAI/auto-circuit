from collections import defaultdict
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import torch as t
from einops import rearrange
from torch import inference_mode

from auto_circuit.data import BatchKey, PromptDataLoader
from auto_circuit.types import AblationType, SrcNode
from auto_circuit.utils.misc import downsample_activations, remove_hooks
from auto_circuit.utils.patchable_model import PatchableModel


def src_out_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    out: t.Tensor,
    src_nodes: List[SrcNode],
    src_outs: Dict[SrcNode, t.Tensor],
    ablation_type: AblationType,
    sublayer_index: bool = False,
):
    assert not ablation_type.mean_over_dataset
    if ablation_type == AblationType.RESAMPLE:
        out = out
    elif ablation_type == AblationType.ZERO:
        out = t.zeros_like(out)
    elif ablation_type == AblationType.BATCH_TOKENWISE_MEAN:
        repeats = [out.size(0)] + [1] * (out.ndim - 1)
        out = out.mean(dim=0, keepdim=True).repeat(repeats)
    elif ablation_type == AblationType.BATCH_ALL_TOK_MEAN:
        repeats = [out.size(0), out.size(1)] + [1] * (out.ndim - 2)
        out = out.mean(dim=(0, 1), keepdim=True).repeat(repeats)
    else:
        raise NotImplementedError(ablation_type)

    head_dim: Optional[int] = None if sublayer_index else src_nodes[0].head_dim
    out = out if head_dim is None else out.split(1, dim=head_dim)
    if sublayer_index:
        src_outs[src_nodes[0]] = out
    else:
        for s in src_nodes:
            src_outs[s] = (
                out if head_dim is None else out[s.head_idx].squeeze(s.head_dim)
            )


def mean_src_out_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    out: t.Tensor,
    src_nodes: List[SrcNode],
    src_outs: Dict[SrcNode, t.Tensor],
    ablation_type: AblationType,
    sublayer_index: bool = False,
):
    assert ablation_type.mean_over_dataset
    repeats = [out.size(0)] + [1] * (out.ndim - 1)
    out = out.mean(dim=0, keepdim=True).repeat(repeats)

    head_dim: Optional[int] = None if sublayer_index else src_nodes[0].head_dim
    out = out if head_dim is None else out.split(1, dim=head_dim)
    if sublayer_index:
        if src_nodes[0] not in src_outs:
            src_outs[src_nodes[0]] = out
        else:
            src_outs[src_nodes[0]] += out
    else:
        for s in src_nodes:
            src_out = out if head_dim is None else out[s.head_idx].squeeze(s.head_dim)
            if s not in src_outs:
                src_outs[s] = src_out
            else:
                src_outs[s] += src_out


def src_ablations(
    model: PatchableModel,
    sample: t.Tensor | PromptDataLoader,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Dict[int, t.Tensor]:
    """
    Get the activations used to ablate each [`Edge`][auto_circuit.types.Edge] in a
    model, given a particular set of model inputs and an ablation type. See
    [`AblationType`][auto_circuit.types.AblationType] for the different types of
    ablations that can be computed.

    Args:
        model: The model to get the ablations for.
        sample: The data sample to get the ablations for. This is not used for all
            `ablation_type`s. Either a single batch of inputs or a DataLoader.
        ablation_type: The type of ablation to perform.

    Returns:
        A dictionary mapping stage indices to tensors of activations used to ablate
        each [`Edge`][auto_circuit.types.Edge] model on the given input. The shape
        of the tensor is `[Srcs, ...]` where `Srcs` is the number of
        [`SrcNode`][auto_circuit.types.SrcNode]s in the model and `...` is the
        shape of the activations of the model. In a transformer this will be
        `[Srcs, batch, seq, d_model]`.
    """
    src_outs: Dict[SrcNode, t.Tensor] = {}
    src_modules: Dict[t.nn.Module, List[SrcNode]] = defaultdict(list)
    src_outs_per_stage: Dict[int, t.Tensor] = {}
    [src_modules[src.module(model)].append(src) for src in model.srcs]
    with remove_hooks() as handles, inference_mode():
        # Install hooks to collect activations at each src module
        for mod, src_nodes in src_modules.items():
            hook_fn = partial(
                mean_src_out_hook if ablation_type.mean_over_dataset else src_out_hook,
                src_nodes=src_nodes,
                src_outs=src_outs,
                ablation_type=ablation_type,
                sublayer_index=bool(src_nodes[0].sublayer_shape),
            )
            handles.add(mod.register_forward_hook(hook_fn))

        if ablation_type.mean_over_dataset:
            device = next(iter(model.parameters())).device
            # Collect activations over the entire dataset and take the mean
            assert isinstance(sample, PromptDataLoader)
            for batch in sample:
                if ablation_type.clean_dataset:
                    model(batch.clean.to(device))
                if ablation_type.corrupt_dataset:
                    model(batch.corrupt.to(device))
            # PromptDataLoader has equal size batches, so we can take the mean of means
            mult = int(ablation_type.clean_dataset) + int(ablation_type.corrupt_dataset)
            assert mult == 2 or mult == 1
            for src, src_out in src_outs.items():
                src_outs[src] = src_out / (len(sample) * mult)
        else:
            # Collect activations for a single batch
            assert isinstance(sample, t.Tensor)
            model(sample)

    # Sort the src_outs dict by node idx
    src_outs = dict(sorted(src_outs.items(), key=lambda x: x[0].src_idx))

    for i in range(len(model.downsample_modules) + 1):
        stage_activations = []

        for j in range(i + 1):
            idxs = [
                src.src_idx for src in src_outs.keys() if src.module(model).stage == j
            ]
            for k, v in src_outs.items():
                if k.src_idx in idxs:
                    stage_activations.append(
                        downsample_activations(
                            model.downsample_modules[j:i], v.clone().detach(), j, i
                        )
                    )

        # TODO: handle the case where there are no head dims
        a_node = next((n for n in src_outs.keys() if n.head_dim is not None), next(iter(src_outs)))
        if a_node.sublayer_shape is not None:
            assert a_node.head_dim is not None
            src_outs_per_stage[i] = rearrange(
                t.cat(stage_activations, dim=a_node.head_dim),
                "b (l ch) ... -> l b ch ...",
                ch=stage_activations[0].shape[a_node.head_dim],
            )
        else:
            src_outs_per_stage[i] = t.stack(stage_activations)
    return src_outs_per_stage


def batch_src_ablations(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    ablation_type: AblationType = AblationType.RESAMPLE,
    clean_corrupt: Optional[Literal["clean", "corrupt"]] = None,
) -> Dict[BatchKey, Dict[int, t.Tensor]]:
    """
    Wrapper of [`src_ablations`][auto_circuit.utils.ablation_activations.src_ablations]
    that returns ablations for each batch in a dataloader.

    Args:
        model: The model to get the ablations for.
        dataloader: The input data to get the ablations for.
        ablation_type: The type of ablation to perform.
        clean_corrupt: Whether to use the clean or corrupt inputs to calculate the
            ablations.

    Returns:
        A dictionary mapping [`BatchKey`][auto_circuit.data.BatchKey]s to the
            activations used to ablate each [`Edge`][auto_circuit.types.Edge] in the
            model on the corresponding batch.
    """
    batch_specific_ablation = [
        AblationType.RESAMPLE,
        AblationType.BATCH_TOKENWISE_MEAN,
        AblationType.BATCH_ALL_TOK_MEAN,
    ]
    assert (clean_corrupt is not None) == (ablation_type in batch_specific_ablation)

    patch_outs: Dict[BatchKey, Dict[int, t.Tensor]] = {}
    if ablation_type.mean_over_dataset:
        mean_patch = src_ablations(model, dataloader, ablation_type)
        patch_outs = {batch.key: mean_patch for batch in dataloader}
    else:
        for batch in dataloader:
            if ablation_type == AblationType.ZERO:
                input_batch = batch.clean
            else:
                input_batch = batch.clean if clean_corrupt == "clean" else batch.corrupt
            patch_outs[batch.key] = src_ablations(
                model, input_batch.to(next(model.wrapped_model.parameters()).device), ablation_type
            )
    return patch_outs
