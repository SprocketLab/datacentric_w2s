import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append("/u/c/h/chshin/changho/datacentric_w2s")
from pathlib import Path
import os
import torch
import gc
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from ruptures import Binseg
from datasets import DatasetDict, load_from_disk
from simple_parsing import parse
from transformers import (
    TrainingArguments,
)

from w2s.ds_registry import load_and_process_dataset
from w2s.model import ModelConfig
from w2s.sft import train, linear_probe_train, load_model_and_predict, load_model_and_save_activations
from w2s.sft_config import SFTConfig
from w2s.probe import ProbeConfig, LogisticProbeConfig
from w2s.utils import get_config_foldername
from simple_parsing import Serializable, field, subgroups
from w2s.ds_registry import VALID_DATASETS
from w2s.probe import PROBES

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Overlap probing experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

args = parse_args()
seed = args.seed


probe_name = "logreg"
probe_cfg = LogisticProbeConfig()
# dataset_name = "sciq"
df_result = []

for dataset_name in VALID_DATASETS:
    cfg = SFTConfig(
        dataset=dataset_name,
        # n_train=500,
        # n_val=100,
        # n_test=300,
        n_train=10_000,
        n_val=1_000,
        n_test=5_000,
        n_predict=0,
        minibatch_size=1,
        batch_size=32,
        results_folder="../../results",
        seed=seed,
        disable_lora=True,
        strong_only=True,
        probe=LogisticProbeConfig(),
            run_name=f"{dataset_name}_{seed}",
        )

    root = Path(cfg.results_folder) / cfg.run_name
    shared_root = Path(cfg.results_folder) / cfg.shared_folder
    cfg_name = f"{cfg.run_name}_{cfg.weak_model_name.split('/')[-1]}_{cfg.strong_model_name.split('/')[-1]}"

    # Save splits first
    save_path = shared_root / cfg_name / "splits"
    if os.path.exists(save_path):
        print(f"Loading splits from {save_path}")
        splits = load_from_disk(str(save_path)) 
    else:
        print(f"Loading and processing dataset {cfg.dataset}")
        splits = load_and_process_dataset(
            cfg.dataset, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_predict
        )

        train_halves = splits["train"].train_test_split(test_size=0.5, seed=cfg.seed)
        splits["weak_train"] = train_halves["train"]
        splits["strong_train"] = train_halves["test"]

        cols = ["hard_label", "txt"]
        splits = splits.select_columns(cols).rename_column("hard_label", "labels")
        for split in splits:
            splits[split] = splits[split].add_column("gt_labels", splits[split]["labels"])

        print(
            f"Example:\n\n{splits['strong_train'][0]['txt']}\n\nLabel: {splits['strong_train'][0]['labels']}"
        )


        print(f"Saving splits to {save_path}")
        save_path.mkdir(parents=True, exist_ok=True)
        splits.save_to_disk(str(save_path))


    
    weak_train_args: dict = dict(
        num_train_epochs=cfg.n_epochs,
        adam_beta2=0.95,
        gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
        eval_strategy="steps",
        label_names=["labels"],
        load_best_model_at_end=cfg.load_best_model_at_end,
        logging_steps=25,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=cfg.n_warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_schedule,
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
    )

    strong_train_args = dict(
        num_train_epochs=cfg.n_epochs,
        adam_beta2=0.95,
        gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
        eval_strategy="steps",
        label_names=["labels"],
        load_best_model_at_end=cfg.load_best_model_at_end,
        logging_steps=25,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=cfg.n_warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_schedule,
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
    )

    w2s_train_args = dict(
        num_train_epochs=cfg.n_epochs,
        adam_beta2=0.95,
        gradient_accumulation_steps=cfg.batch_size // cfg.minibatch_size,
        eval_strategy="steps",
        label_names=["labels"],
        load_best_model_at_end=cfg.load_best_model_at_end,
        logging_steps=25,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        per_device_train_batch_size=cfg.minibatch_size,
        per_device_eval_batch_size=cfg.minibatch_size,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        tf32=True,  # Use Tensor Cores even for fp32 matmuls
        warmup_steps=cfg.n_warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_schedule,
        eval_steps=cfg.eval_every,
        save_steps=cfg.save_every,
    )



    def get_model_and_run_name(model_name, current_name):
        model_last = model_name.split("/")[-1]
        model_cfg = ModelConfig(name=model_name, enable_lora=not cfg.disable_lora)
        run_name = f"{current_name}-{cfg.run_name}-{cfg.dataset}-{model_last}"
        return model_cfg, run_name
    
    # train weak floor, get predictions
    print("\n\033[32m===== Linear probing experiments =====\033[0m")
    weak_model_cfg, weak_run_name = get_model_and_run_name(cfg.weak_model_name, "weak")
    weak_train_args["run_name"] = weak_run_name
    weak_train_args["output_dir"] = str(shared_root / cfg_name / "weak")
    weak_train_args["learning_rate"] = cfg.weak_lr

    w2s_model_cfg, w2s_run_name = get_model_and_run_name(cfg.strong_model_name, "strong")
    w2s_train_args["run_name"] = w2s_run_name
    w2s_train_args["output_dir"] = str(shared_root / cfg_name / "w2s")
    w2s_train_args["learning_rate"] = cfg.strong_lr

    # Do sampling for faster check
    w2s_train = splits["strong_train"]
    w2s_val = splits["val"]

    weak_ds_dict = DatasetDict(
        {
            "train": splits["weak_train"],
            "val": splits["val"],
            "test": splits["test"],
        }
    )

    strong_ds_dict = DatasetDict(
        {
            "train": w2s_train,
            "val": w2s_val,
            "test": splits["test"],
        }
    )

    
    weak_acts_dir = shared_root / cfg_name / "weak_activations"
    strong_acts_dir = shared_root / cfg_name / "strong_activations"

    x_weak_train = torch.load(weak_acts_dir / f"weak_train.pt", map_location="cuda")
    x_strong_train = torch.load(strong_acts_dir / f"strong_train.pt", map_location="cuda")
    x_w2s_train_for_pseudolabeling = torch.load(weak_acts_dir / f"strong_train.pt", map_location="cuda")
    x_weak_test = torch.load(weak_acts_dir / f"test.pt", map_location="cuda")
    x_strong_test = torch.load(strong_acts_dir / f"test.pt", map_location="cuda")
    y_weak_train = torch.tensor(splits["weak_train"]["labels"], device="cuda")
    y_strong_train = torch.tensor(splits["strong_train"]["labels"], device="cuda")
    y_test = torch.tensor(splits["test"]["labels"], device="cuda")


    print(f"Weak acts shape: {x_weak_train.shape}")
    print(f"Strong acts shape: {x_strong_train.shape}")

    weak_probe = PROBES[probe_name](probe_cfg)
    weak_probe.fit(x_weak_train, y_weak_train)

    strong_probe = PROBES[probe_name](probe_cfg)
    strong_probe.fit(x_strong_train, y_strong_train)

    y_w2s_train_for_pseudolabeling = weak_probe.predict(x_w2s_train_for_pseudolabeling)
    w2s_probe = PROBES[probe_name](probe_cfg)
    w2s_probe.fit(x_strong_train, torch.tensor(y_w2s_train_for_pseudolabeling, device="cuda"))

    # Compute accuracy for weak probe on test set
    weak_preds = weak_probe.predict(x_weak_test)
    strong_preds = strong_probe.predict(x_strong_test)
    w2s_preds = w2s_probe.predict(x_strong_test)

    weak_test_labels = torch.tensor(splits["test"]["labels"], device="cuda")
    weak_test_accuracy = (weak_preds.round() == weak_test_labels).float().mean().item()
    
    # Compute accuracy for strong probe on test set
    strong_test_labels = torch.tensor(splits["test"]["labels"], device="cuda")
    strong_test_accuracy = (strong_preds.round() == strong_test_labels).float().mean().item()

    # Compute accuracy for w2s probe on test set
    w2s_test_labels = torch.tensor(splits["test"]["labels"], device="cuda")
    w2s_test_accuracy = (w2s_preds.round() == w2s_test_labels).float().mean().item()

    pgr = (w2s_test_accuracy - weak_test_accuracy) / (strong_test_accuracy - weak_test_accuracy)
    
    print(f"Weak probe test accuracy: {weak_test_accuracy:.4f}")
    print(f"Strong probe test accuracy: {strong_test_accuracy:.4f}")
    print(f"W2S probe test accuracy: {w2s_test_accuracy:.4f}")
    print(f"PGR: {pgr:.4f}")


    y_w2s_train_for_pseudolabeling = y_w2s_train_for_pseudolabeling.cpu().detach().numpy()
    x_strong_train = x_strong_train.cpu().detach().numpy()
    
    confidence_w2s_train = 2*np.abs(y_w2s_train_for_pseudolabeling-0.5)
    # Sort confidence scores
    sorted_confidence = np.sort(confidence_w2s_train)
    
    # Perform change point detection
    model = Binseg(model="l2").fit(sorted_confidence.reshape(-1, 1))
    change_points = model.predict(n_bkps=1)[0]
    
    # Use the detected change point as the threshold
    confidence_threshold = sorted_confidence[change_points]
    low_confidence_indices = np.where(confidence_w2s_train <= confidence_threshold)[0]
    high_confidence_indices = np.where(confidence_w2s_train > confidence_threshold)[0]

    x_w2s_train = x_strong_train # shared feature set
    y_w2s_train = y_w2s_train_for_pseudolabeling
    x_w2s_train_hard = x_strong_train[low_confidence_indices]
    y_w2s_train_hard = y_w2s_train_for_pseudolabeling[low_confidence_indices]
    x_w2s_train_easy_or_overlap = x_strong_train[high_confidence_indices]
    y_w2s_train_easy_or_overlap = y_w2s_train_for_pseudolabeling[high_confidence_indices]

    x_w2s_train_hard_normalized = x_w2s_train_hard / np.linalg.norm(x_w2s_train_hard, axis=1, keepdims=True)
    x_w2s_train_easy_or_overlap_normalized = x_w2s_train_easy_or_overlap / np.linalg.norm(x_w2s_train_easy_or_overlap, axis=1, keepdims=True)
    align_scores = np.abs(x_w2s_train_easy_or_overlap_normalized @ x_w2s_train_hard_normalized.T).max(axis=1)
    # align_scores = np.abs(x_w2s_train_easy_or_overlap @ x_w2s_train_hard.T).max(axis=1)
    # Apply change point detection to decide threshold for align scores
    sorted_align_scores = np.sort(align_scores)
    
    # Perform change point detection
    model = Binseg(model="l2").fit(sorted_align_scores.reshape(-1, 1))
    change_points = model.predict(n_bkps=1)[0]
    
    # Use the detected change point as the threshold
    align_score_threshold = sorted_align_scores[change_points]
    overlap_indices = np.where(align_scores >= align_score_threshold)[0]
    nonoverlap_indices = np.where(align_scores < align_score_threshold)[0]

    x_w2s_train_overlap = x_w2s_train_easy_or_overlap[overlap_indices]
    y_w2s_train_overlap = y_w2s_train_easy_or_overlap[overlap_indices]
    x_w2s_train_nonoverlap = np.concatenate([x_w2s_train_easy_or_overlap[nonoverlap_indices], x_w2s_train_hard])
    y_w2s_train_nonoverlap = np.concatenate([y_w2s_train_easy_or_overlap[nonoverlap_indices], y_w2s_train_hard])

    # Run mixing experiments
    acc_list = []
    wl_dataset_size = min(len(x_w2s_train_overlap), len(x_w2s_train_nonoverlap))
    w2s_train_indices = np.arange(len(x_w2s_train))
    w2s_train_sampled_indices = np.random.choice(w2s_train_indices, size=wl_dataset_size, replace=False)
    
    x_w2s_train_sampled = x_w2s_train[w2s_train_sampled_indices]
    y_w2s_train_sampled = y_w2s_train[w2s_train_sampled_indices]
    strong_sampled_probe = PROBES[probe_name](probe_cfg)
    x_w2s, y_w2s = torch.tensor(x_w2s_train_sampled, device="cuda"), torch.tensor(y_w2s_train_sampled, device="cuda")
    strong_sampled_probe.fit(x_w2s, y_w2s)
    strong_sampled_preds = strong_sampled_probe.predict(x_strong_test)
    strong_sampled_acc = (strong_sampled_preds.round() == y_test).float().mean().item()
    # model, gt_acc = train_dnn(x_w2s_train_sampled, y_w2s_train_sampled, x_test, y_test, verbose=verbose)

    proportion_list = np.arange(0, 1.01, 0.1)
    overlap_indices_full = np.arange(len(x_w2s_train_overlap))
    nonoverlap_indices_full = np.arange(len(x_w2s_train_nonoverlap))

    for overlap_portion in tqdm(proportion_list):
        if overlap_portion==0:
            sample_indices = np.random.choice(nonoverlap_indices_full, size=wl_dataset_size, replace=False).tolist()
            x_w2s = x_w2s_train_nonoverlap[sample_indices]
            y_w2s = y_w2s_train_nonoverlap[sample_indices]
        elif overlap_portion==1:
            sample_indices = np.random.choice(overlap_indices_full, size=wl_dataset_size, replace=False).tolist()
            x_w2s = x_w2s_train_overlap[sample_indices]
            y_w2s = y_w2s_train_overlap[sample_indices]
        else:   
            overlap_portion = np.round(overlap_portion, 1)
            nonoverlap_size = int(wl_dataset_size * (1-overlap_portion))
            overlap_size = int(wl_dataset_size * (overlap_portion))

            nonoverlap_indices = np.random.choice(nonoverlap_indices_full, size=nonoverlap_size, replace=False).tolist()
            overlap_indices = np.random.choice(overlap_indices_full, size=overlap_size, replace=False).tolist()
            
            
            x_w2s = np.concatenate([x_w2s_train_nonoverlap[nonoverlap_indices], x_w2s_train_overlap[overlap_indices]])
            y_w2s = np.concatenate([y_w2s_train_nonoverlap[nonoverlap_indices], y_w2s_train_overlap[overlap_indices]])
            
        # Replace LGBMClassifier with train_dnn 
        w2s_overlap_probe = PROBES[probe_name](probe_cfg)
        x_w2s, y_w2s = torch.tensor(x_w2s, device="cuda"), torch.tensor(y_w2s, device="cuda")
        w2s_overlap_probe.fit(x_w2s, y_w2s)
        w2s_overlap_preds = w2s_overlap_probe.predict(x_strong_test)
        w2s_overlap_acc = (w2s_overlap_preds.round() == y_test).float().mean().item()
        acc_list.append(w2s_overlap_acc)
    
    plt.axhline(y=weak_test_accuracy, color='b', linestyle='--', label='weak')
    plt.axhline(y=w2s_test_accuracy, color='g', linestyle='--', label='ws')
    plt.axhline(y=strong_test_accuracy, color='r', linestyle='--', label='strong (gt)')
    plt.plot(proportion_list, acc_list, 'o-', label='w2s')
    plt.xlabel('Proportion of overlap density')
    plt.ylabel('Acc')
    plt.title(dataset_name+f' seed: {seed}')
    plt.legend()
    plt.grid()
    plt.show()

    result = {
        'dataset_name': dataset_name,
        'seed': seed,
        'weak_test_accuracy': weak_test_accuracy,
        'strong_test_accuracy': strong_test_accuracy,
        'w2s_test_accuracy': w2s_test_accuracy,
        'pgr': pgr,
        'strong_sampled_acc': strong_sampled_acc,
        'acc_list': acc_list,
    }

    
    if not os.path.exists(f'../../results/linear_probing_eval/'):
        os.makedirs(f'../../results/linear_probing_eval/')

    with open(f'../../results/linear_probing_eval/overlap_probing_results_{dataset_name}_{seed}.json', 'w') as f:
        json.dump(result, f)
