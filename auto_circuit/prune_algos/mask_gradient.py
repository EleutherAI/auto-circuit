from typing import Dict, Literal, Optional, Set

import torch as t
from torch.nn.functional import log_softmax, kl_div

from auto_circuit.data import PromptDataLoader
from auto_circuit.types import AblationType, BatchKey, Edge, PruneScores
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    patch_mode,
    set_all_masks,
    train_mask_mode,
)
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import batch_avg_answer_diff, batch_avg_answer_val
from auto_circuit.utils.misc import get_logits, flatten_patch_mask_grads


def mask_gradient_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    grad_function: Literal["logit", "prob", "logprob", "logit_exp"],
    answer_function: Literal["avg_diff", "avg_val", "mse", "daat"],
    mask_val: Optional[float] = None,
    integrated_grad_samples: Optional[int] = None,
    ablation_type: AblationType = AblationType.RESAMPLE,
    clean_corrupt: Optional[Literal["clean", "corrupt"]] = "corrupt",
    alternate_model: Optional[PatchableModel] = None
) -> PruneScores:
    """
    Prune scores equal to the gradient of the mask values that interpolates the edges
    between the clean activations and the ablated activations.

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for input.
        official_edges: Not used.
        grad_function: Function to apply to the logits before taking the gradient.
        answer_function: Loss function of the model output which the gradient is taken
            with respect to.
        mask_val: Value of the mask to use for the forward pass. Cannot be used if
            `integrated_grad_samples` is not `None`.
        integrated_grad_samples: If not `None`, we compute an approximation of the
            Integrated Gradients
            [(Sundararajan et al., 2017)](https://arxiv.org/abs/1703.01365) of the model
            output with respect to the mask values. This is computed by averaging the
            mask gradients over `integrated_grad_samples` samples of the mask values
            interpolated between 0 and 1. Cannot be used if `mask_val` is not `None`.
        ablation_type: The type of ablation to perform.
        clean_corrupt: Whether to use the clean or corrupt inputs to calculate the
            ablations.
        alternate_model: If not `None`, the model used to compute ablations is the 
            alternate model, otherwise it is the original model.

    Returns:
        An ordering of the edges by importance to the task. Importance is equal to the
            absolute value of the score assigned to the edge.

    Note:
        When `grad_function="logit"` and `mask_val=0` this function is exactly
        equivalent to
        [`edge_attribution_patching_prune_scores`][auto_circuit.prune_algos.edge_attribution_patching.edge_attribution_patching_prune_scores].
    """
    assert (mask_val is not None) ^ (integrated_grad_samples is not None)  # ^ means XOR
    model = model
    out_slice = model.out_slice
    ablation_model = alternate_model or model

    device = next(model.parameters()).device

    src_outs: Dict[BatchKey, Dict[str, t.Tensor]] = batch_src_ablations(
        ablation_model,
        dataloader,
        ablation_type=ablation_type,
        clean_corrupt=clean_corrupt,
    )

    with train_mask_mode(model):
        for sample in (ig_pbar := tqdm(range((integrated_grad_samples or 0) + 1))):
            ig_pbar.set_description_str(f"Sample: {sample}")
            # Interpolate the mask value if integrating gradients. Else set the value.
            if integrated_grad_samples is not None:
                set_all_masks(model, val=sample / integrated_grad_samples)
            else:
                assert mask_val is not None and integrated_grad_samples is None
                set_all_masks(model, val=mask_val)
            for batch in dataloader:
                patch_src_outs = {
                    k: v.clone().detach() for k, v in src_outs[batch.key].items()
                }
                with patch_mode(model, patch_src_outs):
                    logits = get_logits(model(batch.clean.to(device)), out_slice)
                    if grad_function == "logit":
                        token_vals = logits
                    elif grad_function == "prob":
                        token_vals = t.softmax(logits, dim=-1)
                    elif grad_function == "logprob":
                        token_vals = log_softmax(logits, dim=-1)
                    elif grad_function == "logit_exp":
                        numerator = t.exp(logits)
                        denominator = numerator.sum(dim=-1, keepdim=True)
                        token_vals = numerator / denominator.detach()
                    else:
                        raise ValueError(f"Unknown grad_function: {grad_function}")

                    if answer_function == "avg_diff":
                        loss = -batch_avg_answer_diff(token_vals, batch)
                    elif answer_function == "avg_val":
                        loss = -batch_avg_answer_val(token_vals, batch)
                    elif answer_function == "mse":
                        loss = t.nn.functional.mse_loss(token_vals, batch.answers)
                    elif answer_function == "daat":
                        target_logits = get_logits(ablation_model(batch.clean.to(device)), out_slice).detach()
                        loss = kl_div(log_softmax(logits, dim=-1), t.softmax(target_logits, dim=-1), reduction='batchmean')
 
                    else:
                        raise ValueError(f"Unknown answer_function: {answer_function}")

                    loss.backward()

    prune_scores: PruneScores = {}
    patch_mask_grads = flatten_patch_mask_grads(model.patch_masks, model)
    for mod_name, patch_mask_grad in patch_mask_grads.items():
        grad = patch_mask_grad
        assert grad is not None
        prune_scores[mod_name] = grad.detach().clone()
    return prune_scores
