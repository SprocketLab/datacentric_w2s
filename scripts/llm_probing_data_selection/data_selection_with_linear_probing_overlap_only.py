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
from w2s.data_source_selection_linear_probing import (
    detect_and_partition,
    detect_hard_nonhard,
    detect_overlap_easy,
    partition_indices_with_ratio,
    setup_arms,
    StrategyTracker
)
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Data selection with linear probing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--dataset", type=str, choices=VALID_DATASETS, default="amazon_polarity", help="Dataset name")
    return parser.parse_args()

args = parse_args()
seed = args.seed
dataset_name = args.dataset


if os.path.exists(f'../../results/data_selection_linear_probing_overlap_only_eval/data_selection_with_linear_probing_overlap_only_results_{dataset_name}_{seed}.json'):
    print("The result exists already!")
    print(f"Loading results from ../../results/data_selection_linear_probing_overlap_only_eval/data_selection_with_linear_probing_overlap_only_results_{dataset_name}_{seed}.json")
    with open(f'../../results/data_selection_linear_probing_overlap_only_eval/data_selection_with_linear_probing_overlap_only_results_{dataset_name}_{seed}.json', 'r') as f:
        result = json.load(f)
    print(result)
    exit()
else:
        

    probe_name = "logreg"
    probe_cfg = LogisticProbeConfig()
    # dataset_name = "sciq"
    df_result = []
    data_stats = []

    overlap_sampling_ratio_list = [0.1, 0.9]
    K = len(overlap_sampling_ratio_list)
    hard_sampling_ratio_list = [1/K for _ in range(K)]
    easy_sampling_ratio_list = [1/K for _ in range(K)]
    T = 50

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

    weak_model = PROBES[probe_name](probe_cfg)
    weak_model.fit(x_weak_train, y_weak_train)

    with torch.no_grad():
        y_w2s_train_for_pseudolabeling = weak_model.predict(x_w2s_train_for_pseudolabeling)

    # Compute accuracy for weak probe on test set
    with torch.no_grad():
        weak_preds = weak_model.predict(x_weak_test)
        weak_test_labels = torch.tensor(splits["test"]["labels"], device="cuda")
    weak_test_accuracy = (weak_preds.round() == weak_test_labels).float().mean().item()

    y_w2s_train_for_pseudolabeling = y_w2s_train_for_pseudolabeling.cpu().detach().numpy()
    x_strong_train = x_strong_train.cpu().detach().numpy()
    y_strong_train = y_strong_train.cpu().detach().numpy()

    detected_hard_indices, detected_nonhard_indices = detect_hard_nonhard(x_w2s_train_for_pseudolabeling, y_w2s_train_for_pseudolabeling)


    x_w2s_train_hard = x_strong_train[detected_hard_indices]
    y_w2s_train_hard = y_strong_train[detected_hard_indices]
    x_w2s_train_nonhard = x_strong_train[detected_nonhard_indices]
    y_w2s_train_nonhard = y_strong_train[detected_nonhard_indices]
    detected_easy_indices, detected_overlap_indices = detect_overlap_easy(x_w2s_train_nonhard, x_w2s_train_hard, detected_nonhard_indices)
    x_w2s_train_overlap = x_strong_train[detected_overlap_indices]
    y_w2s_train_overlap = y_strong_train[detected_overlap_indices]
    x_w2s_train_nonoverlap = np.concatenate([x_strong_train[detected_easy_indices], x_strong_train[detected_hard_indices]])
    y_w2s_train_nonoverlap = np.concatenate([y_strong_train[detected_easy_indices], y_strong_train[detected_hard_indices]])

    # Random partition x_strong based on overlap sampling ratio
    np.random.seed(seed)
    np.random.shuffle(detected_easy_indices)
    np.random.shuffle(detected_hard_indices)
    np.random.shuffle(detected_overlap_indices)

    detected_overlap_partition = partition_indices_with_ratio(detected_overlap_indices, overlap_sampling_ratio_list, seed)
    detected_easy_partition = partition_indices_with_ratio(detected_easy_indices, easy_sampling_ratio_list, seed)
    detected_hard_partition = partition_indices_with_ratio(detected_hard_indices, hard_sampling_ratio_list, seed)

    # Loop to decie number_of_samples_per_round
    sample_size_list = []
    overlap_density_list = []
    for easy_indices, hard_indices, overlap_indices in zip(detected_easy_partition, detected_hard_partition, detected_overlap_partition):
        sample_indices = np.concatenate([easy_indices, hard_indices, overlap_indices])
        sample_size_list.append(sample_indices.shape[0])
        overlap_density_list.append(overlap_indices.shape[0] / sample_indices.shape[0])
    min_sample_size = min(sample_size_list)
    number_of_samples_per_round = int(min_sample_size / T)


    ############## Bandit #############
    # Setup sources
    print("Setting up sources...")
    arms = setup_arms(K, detected_easy_partition, detected_hard_partition, detected_overlap_partition, x_strong_train, x_w2s_train_for_pseudolabeling, y_strong_train, overlap_density_list, number_of_samples_per_round, T, seed)
    opt_o_log = []
    opt_acc_overlap_only_log = []
    ucb_o_log = []
    ucb_acc_overlap_only_log = []
    ucb_ucb_log =[]
    random_o_log = []
    random_acc_overlap_only_log = []

    # Run Bandit: Optimal
    print("Running Optimal...")



    opt_source = np.argmax(overlap_sampling_ratio_list)
    opt_tracker = StrategyTracker(x_strong_test, y_test)

    for i in tqdm(range(T)):
        source = opt_source
        arm = arms[source]
        X_train_sample, X_weak_train_sample, y_train_sample = arm.get_samples()
        with torch.no_grad():
            y_train_sample_pseudo = weak_model.predict(X_weak_train_sample)
        
        partition_dict = detect_and_partition(X_train_sample, y_train_sample, y_train_sample_pseudo, opt_tracker)
        arm.update(partition_dict)
        
        opt_tracker.update_samples(partition_dict)
        opt_o_log.append(opt_tracker.get_est_overlap_ratio())
        opt_acc_overlap_only_log.append(opt_tracker.train_and_eval_overlap_only())

    ### UCB
    arms = setup_arms(K, detected_easy_partition, detected_hard_partition, detected_overlap_partition, x_strong_train, x_w2s_train_for_pseudolabeling, y_strong_train, overlap_density_list, number_of_samples_per_round, T, seed)
    ucb_tracker = StrategyTracker(x_strong_test, y_test)

    for i in tqdm(range(K)):
        source = i
        arm = arms[source]
        X_train_sample, X_weak_train_sample, y_train_sample = arm.get_samples()
        with torch.no_grad():
            y_train_sample_pseudo = weak_model.predict(X_weak_train_sample)
        
        partition_dict = detect_and_partition(X_train_sample, y_train_sample, y_train_sample_pseudo, ucb_tracker)
        arm.update(partition_dict)
        ucb_tracker.update_samples(partition_dict)
        ucb_o_log.append(ucb_tracker.get_est_overlap_ratio())
        ucb_acc_overlap_only_log.append(ucb_tracker.train_and_eval_overlap_only())
    for i in tqdm(range(T-K)):
        ucbs = []
        for k in range(K):
            ucbs.append(arms[k].compute_ucb())
        ucb_ucb_log.append(ucbs)
        source = np.argmax(ucbs)
        arm = arms[source]
        X_train_sample, X_weak_train_sample, y_train_sample = arm.get_samples()
        with torch.no_grad():
            y_train_sample_pseudo = weak_model.predict(X_weak_train_sample)
        
        partition_dict = detect_and_partition(X_train_sample, y_train_sample, y_train_sample_pseudo, ucb_tracker)
        
        arm.update(partition_dict)
        ucb_tracker.update_samples(partition_dict)
        ucb_o_log.append(ucb_tracker.get_est_overlap_ratio())
        ucb_acc_overlap_only_log.append(ucb_tracker.train_and_eval_overlap_only())

    ### Random
    print("Running Random...")
    arms = setup_arms(K, detected_easy_partition, detected_hard_partition, detected_overlap_partition, x_strong_train, x_w2s_train_for_pseudolabeling, y_strong_train, overlap_density_list, number_of_samples_per_round, T, seed)
    random_tracker = StrategyTracker(x_strong_test, y_test)

    for i in tqdm(range(T)):
        source = np.random.randint(K)
        arm = arms[source]
        X_train_sample, X_weak_train_sample, y_train_sample = arm.get_samples()
        with torch.no_grad():
            y_train_sample_pseudo = weak_model.predict(X_weak_train_sample)
        partition_dict = detect_and_partition(X_train_sample, y_train_sample, y_train_sample_pseudo, random_tracker)
        arm.update(partition_dict)
        random_tracker.update_samples(partition_dict)
        random_o_log.append(random_tracker.get_est_overlap_ratio())
        random_acc_overlap_only_log.append(random_tracker.train_and_eval_overlap_only())

    result = {
        'seed': seed,
        'dataset_name': dataset_name,
        'opt_o_log': opt_o_log,
        'ucb_o_log': ucb_o_log,
        'random_o_log': random_o_log,
        'opt_acc_overlap_only_log': opt_acc_overlap_only_log,
        'ucb_acc_overlap_only_log': ucb_acc_overlap_only_log,
        'random_acc_overlap_only_log': random_acc_overlap_only_log,
        'ucb_ucb_log': ucb_ucb_log,
        'overlap_density_list': overlap_density_list,
        'min_sample_size': min_sample_size,
        'number_of_samples_per_round': number_of_samples_per_round,
    }
    df_result.append(result)
    print(result)

    if not os.path.exists(f'../../results/data_selection_linear_probing_overlap_only_eval/'):
        os.makedirs(f'../../results/data_selection_linear_probing_overlap_only_eval/')

    with open(f'../../results/data_selection_linear_probing_overlap_only_eval/data_selection_with_linear_probing_overlap_only_results_{dataset_name}_{seed}.json', 'w') as f:
        json.dump(result, f)