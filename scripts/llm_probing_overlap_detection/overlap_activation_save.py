import os
import sys
sys.path.append("/u/c/h/chshin/changho/datacentric_w2s")
from pathlib import Path
import os
import torch
import gc
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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run activation save script with specified seed.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

args = parse_args()
seed = args.seed

print(f"Using seed: {seed}")

# dataset_name = "sciq"
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
    print("\n\033[32m===== Training w2s model =====\033[0m")
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


    weak_act_ds_dict = DatasetDict(
        {
            "weak_train": splits["weak_train"],
            "strong_train": splits["strong_train"],
            "val": splits["val"],
            "test": splits["test"],
        }
    )

    strong_act_ds_dict = DatasetDict(
        {
            "strong_train": splits["strong_train"],
            "val": splits["val"],
            "test": splits["test"],
        }
    )
    # save strong activations
    acts_dir = shared_root / cfg_name / "weak_activations"
    acts_dir.mkdir(parents=True, exist_ok=True)
    load_model_and_save_activations(
        ds_dict=weak_act_ds_dict, 
        model_cfg=weak_model_cfg,
        train_args=TrainingArguments(**weak_train_args),
        acts_dir=acts_dir,
    )

    acts_dir = shared_root / cfg_name / "strong_activations"
    acts_dir.mkdir(parents=True, exist_ok=True)
    load_model_and_save_activations(
        ds_dict=strong_act_ds_dict,
        model_cfg=w2s_model_cfg,
        train_args=TrainingArguments(**w2s_train_args),
        acts_dir=acts_dir,
    )