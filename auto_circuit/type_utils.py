from auto_circuit.types import SrcNode


def slice_src_nodes(min_node: SrcNode, max_node: SrcNode) -> slice:
    # We assume that we're only interested in slicing layers
    if min_node.sublayer_shape is not None:
        return slice(min_node.layer, max_node.layer + 1)
    else:
        return slice(min_node.global_rank, max_node.global_rank + 1)
