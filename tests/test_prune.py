#%%
import os
from typing import Set

import torch as t
from torch.utils.data import DataLoader

from auto_circuit.data import (
    PromptPairBatch,
)
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.prune import run_pruned
from auto_circuit.types import ActType, Edge, ExperimentType
from auto_circuit.utils.graph_utils import prepare_model

# from tests.conftest import micro_model, micro_dataloader
# from tests.conftest import micro_model

os.environ["TOKENIZERS_PARALLELISM"] = "False"


def test_pruning(
    micro_model: MicroModel,
    micro_dataloader: DataLoader[PromptPairBatch],
    show_graphs: bool = False,
):
    """Check that pruning works by pruning a "MicroModel"
    where the correct output can be worked out by hand.

    To visualize, set render_graph=True in run_pruned."""
    model = micro_model
    prepare_model(model, factorized=True, device="cpu")
    test_loader = micro_dataloader

    experiment_type = ExperimentType(
        input_type=ActType.CLEAN, patch_type=ActType.CORRUPT
    )

    test_input = next(iter(test_loader))
    with t.inference_mode():
        clean_out = model(test_input.clean)
        corrupt_out = model(test_input.corrupt)

    assert t.allclose(clean_out, t.tensor([[25.0, 49.0]]))
    assert t.allclose(corrupt_out, t.tensor([[-25.0, -49.0]]))

    edges: Set[Edge] = model.edges  # type: ignore
    edge_dict = dict([(edge.name, edge) for edge in edges])

    prune_scores = {
        edge_dict["Block Layer 0 Elem 1->Output"]: 3.0,
        edge_dict["Block Layer 0 Elem 0->Block Layer 1 Elem 1"]: 2.0,
        edge_dict["Input->Block Layer 0 Elem 0"]: 1.0,
    }

    pruned_outs = run_pruned(
        model=model,
        data_loader=test_loader,
        experiment_type=experiment_type,
        test_edge_counts=[1, 2, 3],
        prune_scores=prune_scores,
        include_zero_edges=True,
        output_dim=0,
        render_graph=show_graphs,
    )
    assert t.allclose(pruned_outs[0][0], clean_out, atol=1e-3)
    assert t.allclose(pruned_outs[1][0], t.tensor([[19.0, 41.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[2][0], t.tensor([[13.0, 25.0]]), atol=1e-3)
    assert t.allclose(pruned_outs[3][0], t.tensor([[9.0, 13.0]]), atol=1e-3)


# micro_model = micro_model()
# micro_dataloader = micro_dataloader()
# test_pruning(micro_model, micro_dataloader, show_graphs=True)
