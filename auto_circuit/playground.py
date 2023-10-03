#%%
import os
import pickle
import random
from datetime import datetime
from typing import Callable, Dict, Set, Tuple

import numpy as np
import torch as t
import torch.backends.mps
import transformer_lens as tl

import auto_circuit
import auto_circuit.data
import auto_circuit.prune
import auto_circuit.utils.graph_utils
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.metrics.ROC import measure_roc
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.ACDC import acdc_edge_counts
from auto_circuit.prune_functions.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_functions.ioi_official import (
    ioi_true_edges,
    ioi_true_edges_prune_scores,
)
from auto_circuit.prune_functions.random_edges import random_prune_scores

# from auto_circuit.prune_functions.parameter_integrated_gradients import (
#     BaselineWeights,
# )
from auto_circuit.types import ActType, Edge, EdgeCounts, ExperimentType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model
from auto_circuit.utils.misc import percent_gpu_mem_used
from auto_circuit.visualize import kl_vs_edges_plot, roc_plot

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"
#%%

device = "cuda" if t.cuda.is_available() else "cpu"
print("device", device)
toy_model = False
if toy_model:
    cfg = tl.HookedTransformerConfig(
        d_vocab=50257,
        n_layers=3,
        d_model=4,
        n_ctx=64,
        n_heads=2,
        d_head=2,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device=device,
    )
    model = tl.HookedTransformer(cfg)
    model.init_weights()
else:
    model = tl.HookedTransformer.from_pretrained("gpt2-small", device=device)

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
assert model.tokenizer is not None
# model.tokenizer.padding_side = "left"
assert model.tokenizer.padding_side == "right"
# model = t.compile(model)
model.eval()

# repo_root = "/Users/josephmiller/Documents/auto-circuit"
repo_root = "/home/dev/auto-circuit"
data_file = "datasets/indirect_object_identification.json"
data_path = f"{repo_root}/{data_file}"
print(percent_gpu_mem_used())

#%%
# ---------- Config ----------
experiment_type = ExperimentType(input_type=ActType.CLEAN, patch_type=ActType.CORRUPT)
factorized = True
# pig_baseline, pig_samples = BaselineWeights.ZERO, 50
edge_counts = EdgeCounts.LOGARITHMIC
one_tao = 6e-2
acdc_tao_range, acdc_tao_step = (one_tao, one_tao), one_tao

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    data_path,
    device=device,
    prepend_bos=True,
    batch_size=32,
    train_test_split=[0.5, 0.5],
    length_limit=64,
)
prepare_model(model, factorized=factorized, device=device)
prune_scores_dict: Dict[str, Dict[Edge, float]] = {}

# ----------------------------
#%%
prune_funcs: Dict[str, Callable] = {
    # f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
    #     parameter_integrated_grads_prune_scores,
    #     baseline_weights=pig_baseline,
    #     samples=pig_samples,
    # ),
    "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
    # "ACDC": partial(
    #     acdc_prune_scores,
    #     tao_exps=list(range(-1, 1)),
    #     output_dim=1,
    # ),
    # "Subnetwork Probing": partial(
    #     subnetwork_probing_prune_scores,
    #     learning_rate=0.1,
    #     epochs=3000,
    #     regularize_lambda=10,
    #     mask_fn="hard_concrete",
    #     show_train_graph=True,
    # ),
    "IOI Official": ioi_true_edges_prune_scores,
}
for name, prune_func in (prune_score_pbar := tqdm(prune_funcs.items())):
    prune_score_pbar.set_description_str(f"Computing prune scores: {name}")
    new_prune_scores = prune_func(model, train_loader)
    if name in prune_scores_dict:
        prune_scores_dict[name].update(new_prune_scores)
    else:
        prune_scores_dict[name] = new_prune_scores

#%%
# SAVE / LOAD PRUNE SCORES DICT
save = False
load = True
if save:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    file_postfix = "(ACDC-factorized-gpt2-MEGA-RUN)"
    file_name = f"prune_scores_dict-{dt_string}-{file_postfix}.pkl"
    with open(f"{repo_root}/.prune_scores_cache/{file_name}", "wb") as f:
        pickle.dump(prune_scores_dict, f)
if load:
    ioi_acdc_pth = "IOI-ACDC-prune-scores"
    ioi_sp_pth = "IOI-Subnetwork-Edge-Probing-prune-scores"
    for pth in [ioi_acdc_pth, ioi_sp_pth]:
        with open(f"{repo_root}/.prune_scores_cache/{pth}.pkl", "rb") as f:
            loaded_prune_scores = pickle.load(f)
            if prune_scores_dict is None:
                prune_scores_dict = loaded_prune_scores
            else:
                for k, v in loaded_prune_scores.items():
                    if k in prune_scores_dict:
                        prune_scores_dict[k].update(v)
                    else:
                        prune_scores_dict[k] = v

#%%
test_edge_counts = edge_counts_util(model, edge_counts)
# pruned_outs_dict: Dict[str, Dict[int, List[t.Tensor]]] = {}
kl_divs: Dict[str, Dict[int, float]] = {}
for prune_func_str, prune_scores in (
    prune_func_pbar := tqdm(prune_scores_dict.items())
):
    prune_func_pbar.set_description_str(f"Pruning with {prune_func_str} scores")
    test_edge = (
        acdc_edge_counts(model, experiment_type, prune_scores)
        if prune_func_str.startswith("ACDC") or prune_func_str.startswith("IOI")
        else test_edge_counts
    )
    pruned_outs = run_pruned(
        model,
        test_loader,
        experiment_type,
        test_edge,
        prune_scores,
        include_zero_edges=~prune_func_str.startswith("IOI"),
    )
    kl_clean, kl_corrupt = measure_kl_div(model, test_loader, pruned_outs)
    kl_divs[prune_func_str + " clean"] = kl_clean
    kl_divs[prune_func_str + " corr"] = kl_corrupt
    del pruned_outs
    t.cuda.empty_cache()

kl_vs_edges_plot(kl_divs, experiment_type, edge_counts, factorized).show()
#%%
test_edge_counts = edge_counts_util(model, edge_counts)
rocs: Dict[str, Set[Tuple[float, float]]] = {}
for prune_func_str, prune_scores in (
    prune_func_pbar := tqdm(prune_scores_dict.items())
):
    test_edge = (
        acdc_edge_counts(model, experiment_type, prune_scores)
        if prune_func_str.startswith("ACDC") or prune_func_str.startswith("IOI")
        else test_edge_counts
    )
    rocs[prune_func_str] = measure_roc(
        model, prune_scores, test_edge, ioi_true_edges(model)
    )

roc_plot(rocs).show()

#%%
