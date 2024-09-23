import os
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
from transformers import TrainingArguments
from tqdm import tqdm
from ruptures import Binseg
from datasets import DatasetDict, load_from_disk
from simple_parsing import parse
from transformers import (
    TrainingArguments,
)
from datasets import Dataset
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

probe_name = "logreg"
probe_cfg = LogisticProbeConfig()
df_result = []

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run experiments with specific seed and dataset')
parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
parser.add_argument('--dataset_name', type=str, required=True, choices=VALID_DATASETS, help='Name of the dataset to use')

args = parser.parse_args()

# Set the seed and dataset_name from parsed arguments
seed = args.seed
dataset_name = args.dataset_name

print(f"Running experiment with seed: {seed} and dataset: {dataset_name}")


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
print("\n\033[32m===== Training weak model =====\033[0m")
weak_ds_dict = DatasetDict(
    {
        "train": splits["weak_train"],
        "val": splits["val"],
        "test": splits["test"],
    }
)
weak_predict_dict = {"train": splits["strong_train"], "val": splits["val"]}
weak_model_cfg, weak_run_name = get_model_and_run_name(cfg.weak_model_name, "weak")
weak_train_args["run_name"] = weak_run_name
weak_train_args["output_dir"] = str(shared_root / cfg_name / "weak")
weak_train_args["learning_rate"] = cfg.weak_lr

weak_train_output = train(
    weak_ds_dict,
    weak_model_cfg,
    TrainingArguments(**weak_train_args),
    cfg.to_dict(),
    transfer=False,
    predict_dict=weak_predict_dict,
)

if weak_train_output is not None:
    _, eval_results = weak_train_output
    weak_test_accuracy = eval_results['eval_accuracy']
    del weak_train_output
    torch.cuda.empty_cache()
    gc.collect()
else:
    with open(weak_train_args["output_dir"]+"/results.json", "r") as f:
        weak_test_accuracy = json.load(f)["eval_accuracy"]


print("\n\033[32m===== Training strong model =====\033[0m")
model_cfg, run_name = get_model_and_run_name(cfg.strong_model_name, "strong")
strong_train_args["run_name"] = run_name
strong_train_args["output_dir"] = str(shared_root / cfg_name / "strong")
strong_train_args["learning_rate"] = cfg.strong_lr
strong_ds_dict = DatasetDict(
    {
        "train": splits["strong_train"],
        "val": splits["val"],
        "test": splits["test"],
    }
)
strong_train_output = train(
    strong_ds_dict,
    model_cfg,
    TrainingArguments(**strong_train_args),
    cfg.to_dict(),
    transfer=False,
)

if strong_train_output is not None:
    _, eval_results = strong_train_output
    strong_test_accuracy = eval_results['eval_accuracy']
    del strong_train_output
    torch.cuda.empty_cache()
    gc.collect()
else:
    with open(strong_train_args["output_dir"]+"/results.json", "r") as f:
        strong_test_accuracy = json.load(f)["eval_accuracy"]


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

# train w2s model training
print("\n\033[32m===== Training w2s model =====\033[0m")
model_cfg, run_name = get_model_and_run_name(cfg.strong_model_name, "w2s")
weak_predict_dict = {"train": splits["strong_train"], "val": splits["val"]}

weak_preds_root = shared_root / cfg_name / "weak" / "predictions"
if (not os.path.exists(os.path.join(weak_preds_root, "train")) or not os.path.exists(os.path.join(weak_preds_root, "val"))):
    _, w2s_pseudolabeling = load_model_and_predict(
        cfg=cfg.to_dict(),
        model_cfg=weak_model_cfg,
        train_args=TrainingArguments(**weak_train_args),
        ds_dict=splits,
        predict_dict=weak_predict_dict
    )

weak_train_preds_ds = load_from_disk(str(weak_preds_root / "train"))
weak_val_preds_ds = load_from_disk(str(weak_preds_root / "val"))

model_cfg, run_name = get_model_and_run_name(cfg.strong_model_name, "w2s")
w2s_ds_dict = DatasetDict(
    {
        "train": (
            splits["strong_train"]
            .remove_columns("labels")
            .add_column("labels", weak_train_preds_ds["soft_pred"])  # type: ignore
        ),
        "val": (
            splits["val"]
            .remove_columns("labels")
            .add_column("labels", weak_val_preds_ds["soft_pred"])
        ),  # type: ignore
        "test": splits["test"],
    }
)
# assert (weak_train_preds_ds["id"] == w2s_ds_dict["train"]["id"])
# assert (weak_val_preds_ds["id"] == w2s_ds_dict["val"]["id"])

w2s_predict_dict = {"train": splits["strong_train"], "val": splits["val"]}
w2s_train_output = train(
    w2s_ds_dict,
    model_cfg,
    TrainingArguments(**w2s_train_args),
    cfg.to_dict(),
    transfer=True,
)

if w2s_train_output is not None:
    _, eval_results = w2s_train_output
    w2s_test_accuracy = eval_results['eval_accuracy']
    del w2s_train_output
    torch.cuda.empty_cache()
    gc.collect()
else:
    with open(w2s_train_args["output_dir"]+"/results.json", "r") as f:
        w2s_test_accuracy = json.load(f)["eval_accuracy"]


y_w2s_train_for_pseudolabeling = np.array(weak_train_preds_ds["soft_pred"])
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
sorted_align_scores = np.sort(align_scores)

# Perform change point detection
model = Binseg(model="l2").fit(sorted_align_scores.reshape(-1, 1))
change_points = model.predict(n_bkps=1)[0]

# Use the detected change point as the threshold
align_score_threshold = sorted_align_scores[change_points]
overlap_indices = np.where(align_scores >= align_score_threshold)[0]
nonoverlap_indices = np.where(align_scores < align_score_threshold)[0]

# Replace embeddings with raw text for model training
x_raw_w2s_train = np.array(splits["strong_train"]["txt"])
x_w2s_train_easy_or_overlap = x_raw_w2s_train[high_confidence_indices]
x_w2s_train_hard = x_raw_w2s_train[low_confidence_indices]


del x_w2s_train
x_w2s_train_overlap = x_w2s_train_easy_or_overlap[overlap_indices]
y_w2s_train_overlap = y_w2s_train_easy_or_overlap[overlap_indices]
x_w2s_train_nonoverlap = np.concatenate([x_w2s_train_easy_or_overlap[nonoverlap_indices], x_w2s_train_hard])
y_w2s_train_nonoverlap = np.concatenate([y_w2s_train_easy_or_overlap[nonoverlap_indices], y_w2s_train_hard])

# Run mixing experiments
acc_list = []
wl_dataset_size = min(len(x_w2s_train_overlap), len(x_w2s_train_nonoverlap))
w2s_train_indices = np.arange(len(x_raw_w2s_train))
w2s_train_sampled_indices = np.random.choice(w2s_train_indices, size=wl_dataset_size, replace=False)

x_w2s = x_raw_w2s_train[w2s_train_sampled_indices]
y_w2s = y_w2s_train[w2s_train_sampled_indices]

strong_size_controlled_ds = DatasetDict(
    {
        "train": Dataset.from_dict({
            "txt": x_w2s,
            "labels": y_w2s
        }),
        "val": splits["val"],
        "test": splits["test"],
    }
)


strong_train_args["output_dir"] = str(shared_root / cfg_name / f"strong_size_controlled")
print(f"Training size controlled strong model")
strong_size_controlled_train_output = train(
    strong_size_controlled_ds,
            model_cfg,
            TrainingArguments(**w2s_train_args),
            cfg.to_dict(),
            transfer=True,
            predict_dict=w2s_predict_dict
)

if strong_size_controlled_train_output is not None:
    _, eval_results = strong_size_controlled_train_output
    strong_sampled_acc = eval_results['eval_accuracy']
    del strong_size_controlled_train_output
    torch.cuda.empty_cache()
    gc.collect()
else:
    with open(w2s_train_args["output_dir"]+"/results.json", "r") as f:
        strong_sampled_acc = json.load(f)["eval_accuracy"]

proportion_list = np.arange(0, 1.01, 0.1)
overlap_indices_full = np.arange(len(x_w2s_train_overlap))
nonoverlap_indices_full = np.arange(len(x_w2s_train_nonoverlap))


for overlap_portion in tqdm(proportion_list):
    print(f"\n\033[32m===== Training w2s model with overlap_portion {overlap_portion} =====\033[0m")
    
    if overlap_portion==0:
        sample_indices = np.random.choice(nonoverlap_indices_full, size=wl_dataset_size, replace=False).tolist()
        x_w2s = x_raw_w2s_train[sample_indices]
        y_w2s = y_w2s_train[sample_indices]
    elif overlap_portion==1:
        sample_indices = np.random.choice(overlap_indices_full, size=wl_dataset_size, replace=False).tolist()
        x_w2s = x_raw_w2s_train[sample_indices]
        y_w2s = y_w2s_train[sample_indices]
    else:   
        overlap_portion = np.round(overlap_portion, 1)
        nonoverlap_size = int(wl_dataset_size * (1-overlap_portion))
        overlap_size = int(wl_dataset_size * (overlap_portion))

        nonoverlap_indices = np.random.choice(nonoverlap_indices_full, size=nonoverlap_size, replace=False).tolist()
        overlap_indices = np.random.choice(overlap_indices_full, size=overlap_size, replace=False).tolist()
        x_w2s_train_nonoverlap = x_raw_w2s_train[nonoverlap_indices]
        x_w2s_train_overlap = x_raw_w2s_train[overlap_indices]

        y_w2s_train_nonoverlap = y_w2s_train[nonoverlap_indices]
        y_w2s_train_overlap = y_w2s_train[overlap_indices]
        x_w2s = np.concatenate([x_w2s_train_nonoverlap, x_w2s_train_overlap])
        y_w2s = np.concatenate([y_w2s_train_nonoverlap, y_w2s_train_overlap])
        
    w2s_overlap_controlled_ds = DatasetDict(
        {
            "train": Dataset.from_dict({
                "txt": x_w2s,
                "labels": y_w2s
            }),
            "val": splits["val"],
            "test": splits["test"],
        }
    )

    w2s_train_args["output_dir"] = str(shared_root / cfg_name / f"w2s_overlap_ratio_{overlap_portion}_from_pretrained_embeddings")
    print(f"Training w2s model with overlap ratio {overlap_portion}")
    
    w2s_train_output = train(
        w2s_overlap_controlled_ds,
        model_cfg,
        TrainingArguments(**w2s_train_args),
        cfg.to_dict(),
        transfer=True,
        predict_dict=w2s_predict_dict
    )
    if w2s_train_output is not None:
        _, eval_results = w2s_train_output
        w2s_overlap_acc = eval_results['eval_accuracy']
        acc_list.append(w2s_overlap_acc)
        del w2s_train_output
        torch.cuda.empty_cache()
        gc.collect()
    else:
        with open(w2s_train_args["output_dir"]+"/results.json", "r") as f:
            w2s_overlap_acc = json.load(f)["eval_accuracy"]
            acc_list.append(w2s_overlap_acc)

pgr = (w2s_test_accuracy - weak_test_accuracy) / (strong_test_accuracy - weak_test_accuracy)

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


if not os.path.exists(f'../../results/finetuning_experiments_with_pretrained_embeddings/'):
    os.makedirs(f'../../results/finetuning_experiments_with_pretrained_embeddings/')

with open(f'../../results/finetuning_experiments_with_pretrained_embeddings/overlap_finetuning_results_{dataset_name}_{seed}.json', 'w') as f:
    json.dump(result, f)