from typing import Literal, List, Callable, Dict, Any

import torch as t
import pdb

from auto_circuit.utils.ablation_activations import src_ablations, batch_src_ablations
from auto_circuit.types import AblationType
from auto_circuit.data import PromptPair, PromptDataset, PromptDataLoader
from auto_circuit.utils.patchable_model import PatchableModel

def get_transplant_sampler(
        model: PatchableModel, 
        ablation_dataset_choice: Literal["clean", "corrupt"] = "clean", 
        ablation_type: AblationType = AblationType.RESAMPLE
    ) -> Callable[[List[PromptPair]], Dict[int, t.Tensor]]:
    def transplant_sampler(batch: List[PromptPair]) -> Dict[int, t.Tensor]:
        input_batch = t.stack([getattr(p, ablation_dataset_choice) for p in batch])
        ablations = src_ablations(
            model,
            input_batch.to(next(model.wrapped_model.parameters()).device),
            ablation_type
        )
        # Assuming we want the last stage's ablations
        return {k: v.clone().detach() for k, v in ablations.items()}
    
    return transplant_sampler

def get_transplant_sampler_precomputed(
        model: PatchableModel, 
        dataset: PromptDataset,
        ablation_dataset_choice: Literal["clean", "corrupt"] = "clean", 
        ablation_type: AblationType = AblationType.RESAMPLE,
        dataloader_config: Dict[str, Any] = {}
    ) -> Callable[[List[PromptPair]], Dict[int, t.Tensor]]:
    dataloader = PromptDataLoader(dataset, **dataloader_config)
    ablations = batch_src_ablations(
        model,
        dataloader,
        ablation_type,
        clean_corrupt=ablation_dataset_choice
    )
    def transplant_sampler(batch: List[PromptPair]) -> Dict[int, t.Tensor]:
        clean = t.stack([p.clean for p in batch])
        corrupt = t.stack([p.corrupt for p in batch])
        key = hash((str(clean.tolist()), str(corrupt.tolist())))
        return ablations[key]
    
    return transplant_sampler

def get_mean_transplant_sampler(
        model: PatchableModel, 
        dataset: PromptDataset, 
        ablation_type: AblationType = AblationType.TOKENWISE_MEAN_CLEAN,
        dataloader_config: Dict[str, Any] = {}
    ) -> Callable[[List[PromptPair]], Dict[int, t.Tensor]]:
    dataloader = PromptDataLoader(dataset, **dataloader_config)
    ablations = src_ablations(
        model,
        dataloader,
        ablation_type
    )
    ablations = {k: v.clone().detach() for k, v in ablations.items()}
    def mean_transplant_sampler(batch: List[PromptPair]) -> Dict[int, t.Tensor]:
        return ablations
    
    return mean_transplant_sampler