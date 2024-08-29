import pdb
from typing import Any, Dict, List, Optional

import torch as t
from einops import einsum

from auto_circuit.types import MaskFn, PatchWrapper
from auto_circuit.utils.misc import downsample_activations
from auto_circuit.utils.tensor_ops import sample_hard_concrete


class PatchWrapperImpl(PatchWrapper):
    """
    PyTorch module that wraps another module, a [`Node`][auto_circuit.types.Node] in
    the computation graph of the model. Implements the abstract
    [`PatchWrapper`][auto_circuit.types.PatchWrapper] class, which exists to work around
    circular import issues.

    If the wrapped module is a [`SrcNode`][auto_circuit.types.SrcNode], the tensor
    `self.curr_src_outs` (a single instance of which is shared by all PatchWrappers in
    the model) is updated with the output of the wrapped module.

    If the wrapped module is a [`DestNode`][auto_circuit.types.DestNode], the input to
    the wrapped module is adjusted in order to interpolate the activations of the
    incoming edges between the default activations (`self.curr_src_outs`) and the
    ablated activations (`self.patch_src_outs`).

    Note:
        Most `PatchWrapper`s are both [`SrcNode`][auto_circuit.types.SrcNode]s and
        [`DestNode`][auto_circuit.types.DestNode]s.

    Args:
        module_name: Name of the wrapped module.
        module: The module to wrap.
        head_dim: The dimension along which to split the heads. In TransformerLens
            `HookedTransformer`s this is `2` because the activations have shape
            `[batch, seq_len, n_heads, head_dim]`.
        seq_dim: The sequence dimension of the model. This is the dimension on which new
            inputs are concatenated. In transformers, this is `1` because the
            activations are of shape `[batch_size, seq_len, hidden_dim]`.
        is_src: Whether the wrapped module is a [`SrcNode`][auto_circuit.types.SrcNode].
        src_idxs: The slice of the list of indices of
            [`SrcNode`][auto_circuit.types.SrcNode]s which output from this
            module. This is used to slice the shared `curr_src_outs` tensor when
            updating the activations of the current forward pass.
        is_dest (bool): Whether the wrapped module is a
            [`DestNode`][auto_circuit.types.DestNode].
        patch_mask: The mask that interpolates between the default activations
            (`curr_src_outs`) and the ablation activations (`patch_src_outs`).
        in_srcs: The slice of the list of indices of
            [`SrcNode`][auto_circuit.types.SrcNode]s which input to this module. This is
            used to slice the shared `curr_src_outs` tensor and the shared
            `patch_src_outs` tensor, when interpolating the activations of the incoming
            edges.
        in_stages: 
        stage: The stage of the module in the model. This is used to handle vision
            models where activations are downsampled at each stage.
    """

    def __init__(
        self,
        module_name: str,
        module: t.nn.Module,
        head_dim: Optional[int] = None,
        seq_dim: Optional[int] = None,
        is_src: bool = False,
        src_idxs: Optional[slice] = None,
        is_dest: bool = False,
        patch_mask: Dict[str, t.Tensor | None] = {},
        in_srcs: Dict[str, slice | None] = {},
        stage: str = "0",
        downsample_modules: List[t.nn.Module] = [],
        sublayer_index: bool = False,
    ):
        super().__init__()
        self.module_name: str = module_name
        self.module: t.nn.Module = module
        self.head_dim: Optional[int] = head_dim
        self.seq_dim: Optional[int] = seq_dim
        self.curr_src_outs: Optional[Dict[str, t.Tensor]] = None
        self.in_srcs: Dict[str, slice | None] = in_srcs
        self.stage: str = stage
        self.downsample_modules: List[t.nn.Module] = downsample_modules
        self.sublayer_index: bool = sublayer_index

        self.is_src = is_src
        if self.is_src:
            assert src_idxs is not None
            self.src_idxs: slice = src_idxs

        self.is_dest = is_dest
        if self.is_dest:
            assert patch_mask is not None
            self.patch_mask: t.nn.ParameterDict = t.nn.ParameterDict({stage: t.nn.Parameter(mask) for stage, mask in patch_mask.items()})
            self.patch_src_outs: Optional[Dict[str, t.Tensor]] = None
            self.mask_fn: MaskFn = None
            self.dropout_layer: t.nn.Module = t.nn.Dropout(p=0.0)
        self.patch_mode = False
        self.batch_size = None

        assert head_dim is None or seq_dim is None or head_dim > seq_dim
        dims = range(1, max(head_dim if head_dim else 2, seq_dim if seq_dim else 2))
        self.dims = " ".join(["seq" if i == seq_dim else f"d{i}" for i in dims])

    def set_mask_batch_size(self, batch_size: int | None):
        """
        Set the batch size of the patch mask. Should only be used by context manager
        [`set_mask_batch_size`][auto_circuit.utils.graph_utils.set_mask_batch_size]

        The current primary use case is to collect gradients on the patch mask for
        each input in the batch.

        Warning:
            This is an experimental feature that breaks some parts of the library and
            should be used with caution.

        Args:
            batch_size: The batch size of the patch mask.
        """
        pdb.set_trace()
        if batch_size is None and self.batch_size is None:
            return
        if batch_size is None:  # removing batch dim
            self.patch_mask = t.nn.ParameterDict({stage: t.nn.Parameter(
                self.patch_mask[stage].clone()
                ) for stage in self.patch_mask.keys()})
        elif self.batch_size is None:  # adding batch_dim
            self.patch_mask = t.nn.ParameterDict({stage: t.nn.Parameter(
                self.patch_mask[stage].repeat(batch_size, *((1,) * self.patch_mask[stage].ndim))
                ) for stage in self.patch_mask.keys()})
        elif self.batch_size != batch_size:  # modifying batch dim
            self.patch_mask = t.nn.ParameterDict({stage: t.nn.Parameter(
                self.patch_mask[stage][0]
                .clone()
                .repeat(batch_size, *((1,) * self.patch_mask[stage].ndim))
            ) for stage in self.patch_mask.keys()})
        self.batch_size = batch_size

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arg_0: t.Tensor = args[0].clone()

        if self.patch_mode and self.is_dest:
            assert (
                self.patch_src_outs is not None
                and self.curr_src_outs is not None
                and self.stage is not None
            )
            for stage in self.patch_mask.keys():
                if stage not in self.patch_src_outs:
                    continue
                
                d_upstream = (
                    self.patch_src_outs[stage][self.in_srcs[stage]]
                    - self.curr_src_outs[stage][self.in_srcs[stage]]
                )

                masked_d = self.apply_mask(d_upstream, arg_0, stage)
                arg_0 += masked_d

        new_args = (arg_0,) + args[1:]
        out = self.module(*new_args, **kwargs)

        if self.patch_mode and self.is_src:
            assert self.curr_src_outs is not None
            if self.head_dim is None or self.sublayer_index:
                src_out = out
            else:
                squeeze_dim = self.head_dim if self.head_dim < 0 else self.head_dim + 1
                src_out = t.stack(out.split(1, dim=self.head_dim)).squeeze(squeeze_dim)
            self.curr_src_outs[self.stage][self.src_idxs] = src_out.clone().detach()

        return out

    def apply_mask(self, d_upstream: t.Tensor, arg_0: t.Tensor, stage: str) -> t.Tensor:
        assert self.curr_src_outs is not None
        batch_str = ""
        head_str = "" if self.head_dim is None else "dest"  # Patch heads separately
        seq_str = "" if self.seq_dim is None else "seq"  # Patch tokens separately
        subl_str = (
            ""
            if (not self.sublayer_index or self.head_dim is None)
            else f"d{self.head_dim}"
        )
        if self.mask_fn == "hard_concrete":
            mask = sample_hard_concrete(
                self.patch_mask[stage], arg_0.size(0), self.batch_size is not None
            )
            batch_str = "batch"  # Sample distribution for each batch element
        elif self.mask_fn == "sigmoid":
            mask = t.sigmoid(self.patch_mask[stage])
        else:
            assert self.mask_fn is None
            batch_str = "batch" if self.batch_size is not None else ""
            mask = self.patch_mask[stage]

        mask = self.dropout_layer(mask)

        ein_pre = f"{batch_str} {seq_str} {head_str} src {subl_str},\
                src batch {self.dims} ..."
        
        if len(self.downsample_modules) > 0:
            if self.sublayer_index:
                ein_post = f"batch {head_str} src {self.dims} ..."
            else:
                ein_post = f"batch src {self.dims} {head_str} ..."
            masked_d_upstream = einsum(mask, d_upstream, f"{ein_pre} -> {ein_post}")
            masked_d = downsample_activations(self.downsample_modules[int(stage):], masked_d_upstream, self.curr_src_outs[stage][self.in_srcs[stage]])

            if self.sublayer_index:
                ein_post_final = f"batch {head_str} ..."
            else:
                ein_post_final = f"batch {self.dims} {head_str} ..."

            masked_d = einsum(masked_d, f"batch {seq_str} {head_str} src {subl_str} ... -> {ein_post_final}")

        else:
            if self.sublayer_index:
                ein_post = f"batch {head_str} ..."
            else:
                ein_post = f"batch {self.dims} {head_str} ..."
            masked_d = einsum(mask, d_upstream, f"{ein_pre} -> {ein_post}")

        return masked_d

    def __repr__(self):
        module_str = self.module.name if hasattr(self.module, "name") else self.module
        repr = [f"PatchWrapper({module_str})"]
        repr.append(("Src✓" if self.is_src else "") + ("Dest✓" if self.is_dest else ""))
        repr.append(f"Patch Mask: [{self.patch_mask.shape}]") if self.is_dest else None
        repr.append(str(self.patch_mask.data)) if self.is_dest else None
        return "\n".join(repr)
