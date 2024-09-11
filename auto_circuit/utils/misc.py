from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from auto_circuit.utils.patchable_model import PatchableModel

from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2LayerNorm
from transformers.modeling_outputs import ModelOutput
import torch as t
from torch.nn import ParameterDict
from einops import einsum, rearrange
from torch.utils.hooks import RemovableHandle


def repo_path_to_abs_path(path: str) -> Path:
    """
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    """
    Save a dictionary to a cache file.

    Args:
        data_dict: The dictionary to save.
        folder_name: The name of the folder to save the cache in.
        base_filename: The base name of the file to save the cache in. The current date
            and time will be appended to the base filename.
    """
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    t.save(data_dict, file_path)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    """
    Load a dictionary from a cache file.

    Args:
        folder_name: The name of the folder to load the cache from.
        filename: The name of the file to load the cache from.

    Returns:
        The loaded dictionary.
    """
    folder = repo_path_to_abs_path(folder_name)
    return t.load(folder / filename)


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    """
    Context manager that makes it easier to use temporary PyTorch hooks without
    accidentally leaving them attached.

    Add hooks to the set yielded by this context manager, and they will be removed when
    the context manager exits.

    Yields:
        An empty set that can be used to store the handles of the hooks.
    """
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def module_by_name(model: Any, module_name: str) -> t.nn.Module:
    """
    Gets a module from a model by name.

    Args:
        model: The model to get the module from.
        module_name: The name of the module to get.

    Returns:
        The module.
    """
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    return reduce(getattr, init_mod + module_name.split("."))  # type: ignore


def set_module_by_name(model: Any, module_name: str, new_module: t.nn.Module):
    """
    Sets a module in a model by name.

    Args:
        model: The model to set the module in.
        module_name: The name of the module to set.
        new_module: The module to replace the existing module with.

    Warning:
        This function modifies the model in place.
    """
    parent = model
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    if "." in module_name:
        parent = reduce(getattr, init_mod + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)


def percent_gpu_mem_used(total_gpu_mib: int = 49000) -> str:
    """
    Get the percentage of GPU memory used.

    Args:
        total_gpu_mib: The total amount of GPU memory in MiB.

    Returns:
        The percentage of GPU memory used.
    """
    return (
        "Memory used {:.1f}".format(
            ((t.cuda.memory_allocated() / (2**20)) / total_gpu_mib) * 100
        )
        + "%"
    )


def run_prompt(
    model: t.nn.Module,
    prompt: str,
    answer: Optional[str] = None,
    top_k: int = 10,
    prepend_bos: bool = False,
):
    """
    Helper function to run a string prompt through a TransformerLens `HookedTransformer`
    model and print the top `top_k` output logits.

    Args:
        model: The model to run the prompt through.
        prompt: The prompt to run through the model.
        answer: Token to show the rank of in the output logits.
        top_k: The number of top output logits to show.
        prepend_bos: Whether to prepend the `BOS` token to the prompt.
    """
    print(" ")
    print("Testing prompt", model.to_str_tokens(prompt))
    toks = model.to_tokens(prompt, prepend_bos=prepend_bos)
    logits = model(toks)
    get_most_similar_embeddings(model, logits[0, -1], answer, top_k=top_k)


def get_most_similar_embeddings(
    model: t.nn.Module,
    out: t.Tensor,
    answer: Optional[str] = None,
    top_k: int = 10,
    apply_ln_final: bool = False,
    apply_unembed: bool = False,
    apply_embed: bool = False,
):
    """
    Helper function to print the top `top_k` most similar embeddings to a given vector.
    Can be used for either embeddings or unembeddings.

    Args:
        model: The model to get the embeddings from.
        out: The vector to get the most similar embeddings to.
        answer: Token to show the rank of in the output logits.
        top_k: The number of top output logits to show.
        apply_ln_final: Whether to apply the final layer normalization to the vector
            before getting the most similar embeddings.
        apply_unembed: If `True`, compare to the unembeddings.
        apply_embed: If `True`, compare to the embeddings.
    """
    assert not (apply_embed and apply_unembed), "Can't apply both embed and unembed"
    show_answer_rank = answer is not None
    answer = " cheese" if answer is None else answer
    out = out.unsqueeze(0).unsqueeze(0) if out.ndim == 1 else out
    out = model.ln_final(out) if apply_ln_final else out
    if apply_embed:
        unembeded = einsum(
            out, model.embed.W_E, "batch pos d_model, vocab d_model -> batch pos vocab"
        )
    elif apply_unembed:
        unembeded = model.unembed(out)
    else:
        unembeded = out
    answer_token = model.to_tokens(answer, prepend_bos=False).squeeze()
    answer_str_token = model.to_str_tokens(answer, prepend_bos=False)
    assert len(answer_str_token) == 1
    logits = unembeded.squeeze()  # type: ignore
    probs = logits.softmax(dim=-1)

    sorted_token_probs, sorted_token_values = probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list
    if answer is not None:
        correct_rank = t.arange(len(sorted_token_values))[
            (sorted_token_values == answer_token).cpu()
        ].item()
    else:
        correct_rank = -1
    if show_answer_rank:
        print(
            f'\n"{answer_str_token[0]}" token rank:',
            f"{correct_rank: <8}",
            f"\nLogit: {logits[answer_token].item():5.2f}",
            f"Prob: {probs[answer_token].item():6.2%}",
        )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f}",
            f"Prob: {sorted_token_probs[i].item():6.2%}",
            f'Token: "{model.to_string(sorted_token_values[i])}"',
        )


def downsample_activations(
    downsample_modules: List[t.nn.Module],
    activations: t.Tensor,
    clean_activations: t.Tensor,
) -> t.Tensor:

    act_shape = activations.shape
    activations = activations.reshape(-1, *act_shape[3:])

    for module in downsample_modules:
        if isinstance(module, ConvNextV2LayerNorm):
            activations = modified_layernorm(module, activations, clean_activations)
        else:
            activations = module(activations)

    activations = activations.reshape(*act_shape[:3], *activations.shape[1:])
    return activations

def modified_layernorm(module: ConvNextV2LayerNorm, activations: t.Tensor, clean_activations: t.Tensor) -> t.Tensor:
    if module.data_format != "channels_first":
        raise NotImplementedError("Only channels_first format is supported in this implementation")

    # Compute mean and variance for clean activations
    clean_mean = clean_activations.mean(1, keepdim=True)
    clean_var = (clean_activations - clean_mean).pow(2).mean(1, keepdim=True)

    # Recompute mean and variance for each position
    n = activations.size(1)  # number of channels
    new_mean = clean_mean - clean_activations / n + activations / n
    new_var = clean_var - (clean_activations - clean_mean).pow(2) / n + (activations - new_mean).pow(2) / n

    # Normalize activations using the recomputed mean and variance
    x = (activations - new_mean) / t.sqrt(new_var + module.eps)
    
    # Apply weight and bias
    x = module.weight[:, None, None] * x + module.bias[:, None, None]

    return x


def get_logits(model_output: t.Tensor | ModelOutput, out_slice: Tuple[slice | int, ...]) -> t.Tensor:
    if isinstance(model_output, ModelOutput):
        assert hasattr(model_output, "logits")
        return model_output.logits[out_slice]
    return model_output[out_slice]


def flatten_patch_masks(patch_masks: ParameterDict, model: "PatchableModel") -> dict[str, t.Tensor]:
    """
    Convert a possibly 2D patch mask to a flat 1D tensor.
    """
    flat_masks_lists: dict[str, list[t.Tensor]] = defaultdict(list)
    for mod_name, patch_mask in patch_masks.items():
        for stage in patch_mask.keys():
            stage_srcs = [node for node in model.srcs if node.stage == stage]
            if len(stage_srcs) == 0:
                continue
            a_node = stage_srcs[0]
            if a_node.sublayer_shape is not None:
                if a_node.head_idx is not None:
                    flat_masks_lists[mod_name].append(rearrange(patch_mask[stage], 'dest_heads l src_heads -> dest_heads (l src_heads)'))
                else:
                    flat_masks_lists[mod_name].append(patch_mask[stage].mean(2))
            else:
                flat_masks_lists[mod_name].append(patch_mask[stage])

    flat_masks: dict[str, t.Tensor] = {mod_name: t.cat(tensor_list, dim=1) for mod_name, tensor_list in flat_masks_lists.items()}
    return flat_masks

def flatten_patch_mask_grads(patch_masks: ParameterDict, model: "PatchableModel") -> dict[str, t.Tensor]:
    """
    Convert possibly 2D patch mask gradients to flat 1D tensors.
    """
    flat_grad_lists: dict[str, list[t.Tensor]] = defaultdict(list)
    for mod_name, patch_mask in patch_masks.items():
        for stage in patch_mask.keys():
            stage_srcs = [node for node in model.srcs if node.stage == stage]
            if len(stage_srcs) == 0:
                continue
            a_node = stage_srcs[0]
            grad = patch_mask[stage].grad
            if grad is None:
                continue
            if a_node.sublayer_shape is not None:
                if a_node.head_idx is not None:
                    flat_grad_lists[mod_name].append(rearrange(grad, 'dest_heads l src_heads -> dest_heads (l src_heads)'))
                else:
                    flat_grad_lists[mod_name].append(grad.mean(2))
            else:
                flat_grad_lists[mod_name].append(grad)

    flat_grads: dict[str, t.Tensor] = {mod_name: t.cat(tensor_list, dim=1) for mod_name, tensor_list in flat_grad_lists.items()}
    return flat_grads