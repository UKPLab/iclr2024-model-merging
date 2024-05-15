import os
import sys

import numpy as np

sys.setrecursionlimit(2500)

import sisyphus.toolkit as tk
from ukp.huggingface.search import HuggingfaceSearchJob
from ukp.huggingface.training import HuggingfaceTrainingJob
from ukp.huggingface.task_arithmetic import TaskArithmeticJob, UncertaintyBasedGradientMatchingJob

Path = tk.Path

code_root = gs.CODE_ROOT

def calculate_fisher_information(model_name_or_path, dataset, dataset_config_name, model_description, per_device_eval_batch_size=1,
                   dataset_test_split="train", time_rqmt=2, mem_rqmt=24, gpu_mem=16):
    config = {
        'model_name_or_path': model_name_or_path,
        'method': 'sequence_classification_squared_gradients',
        'per_device_eval_batch_size': 1, # unbiased estimator of the diagonal Fisher
        'track_fim': True,
        "max_input_length": 384,
    }
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'merging/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': dataset_test_split,
    }

    search_job = HuggingfaceSearchJob(
        code_root=code_root,
        model_path=model_name_or_path,
        config=config,
        search_data_config=search_data_config,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem,
        dummy="rerun"
    )
    tk.register_output(f'results/{dataset}/{dataset_config_name}_{model_description}_fisher', search_job.out_checkpoints_dir)

    return search_job.out_checkpoints_dir


async def example():

    pretraining_tasks = {
        "imdb": {
            "method": "sequence_classification",
            "train_split": "train",
            "test_split": "test[:10%]",
            "num_epochs": 1,
            "dataset_size": 25000,
        },
    }

    finetuning_tasks = {
        "yelp": {
            "method": "sequence_classification",
            "test_split": "test[:2%]",
            "num_epochs": 1,
            "train_split": "train[:20%]",
            "dataset_size": 112000
        },
        "rotten_tomatoes": {
            "method": "sequence_classification",
            "test_split": "test[:10%]",
            "num_epochs": 1,
            "train_split": "train",
            "dataset_size": 8530
        },
        "sst2": {
            "method": "sequence_classification",
            "test_split": "validation[:10%]",
            "num_epochs": 1,
            "train_split": "train",
            "dataset_size": 67349
        },
        "amazon": {
            "method": "sequence_classification",
            "test_split": "test[:2%]",
            "num_epochs": 1,
            "train_split": "train[:5%]",
            "dataset_size": 180000
        },
    }
    dataset_config_name = "classification"

    finetuned_models, finetuned_fishers = [], []

    for dataset, task_config in pretraining_tasks.items():

        beta1 = 0.9
        beta2 = 0.999

        config = {
            "model_name_or_path": "roberta-base",
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "max_input_length": 384,
            "max_output_length": 128,
            "optimizer_name": "AdamW",
            "warmup_steps": 100,
            "min_lr": 0.0,
            "beta1": beta1,
            "beta2": beta2,
            "dropout": 0.0,
            "max_grad_norm": 1.0,
        }

        val_split = task_config["test_split"]

        train_data_config = {
            'dataset_name': os.path.join(code_root, f'merging/datasets/{dataset}.py'),
            'dataset_config_name': dataset_config_name,
            'dataset_train_split': 'train',
            'dataset_val_split': val_split,
        }

        config["method"] = task_config["method"]

        learning_rate = 1e-5
        config["learning_rate"] = learning_rate

        pretrain_train_job = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=task_config["num_epochs"],
            mem_rqmt=24,
            time_rqmt=4,
            gpumem=16,
        )
        pretrain_train_job.add_alias(f"pretrained_model")
        tk.register_output(f"pretrained_model", pretrain_train_job.out_best_model)

        pretrained_fisher = calculate_fisher_information(
            pretrain_train_job.out_best_model,
            dataset,
            "classification",
            "roberta_baseline",
            dataset_test_split="train"
        )

    config["learning_rate"] = 5e-6

    for dataset, task_config in finetuning_tasks.items():

        val_split = task_config["test_split"]

        train_data_config = {
            'dataset_name': os.path.join(code_root, f'merging/datasets/{dataset}.py'),
            'dataset_config_name': dataset_config_name,
            'dataset_train_split': task_config["train_split"],
            'dataset_val_split': val_split,
        }

        config["method"] = task_config["method"]
        config["model_name_or_path"] = pretrain_train_job.out_best_model

        finetune_train_job = HuggingfaceTrainingJob(
            code_root=code_root,
            config=config,
            train_data_config=train_data_config,
            num_epochs=task_config["num_epochs"],
            mem_rqmt=24,
            time_rqmt=4,
            gpumem=16,
        )
        finetune_train_job.add_alias(f"finetuned_model_{dataset}")
        tk.register_output(f"finetuned_model_{dataset}", finetune_train_job.out_best_model)

        finetuned_models.append(finetune_train_job.out_best_model)

        fisher = calculate_fisher_information(
            finetune_train_job.out_best_model,
            dataset,
            "classification",
            "roberta_baseline",
            dataset_test_split="train" if dataset not in ["amazon", "yelp"] else "train[:2%]"
        )

        finetuned_fishers.append(fisher)


    finetuning_train_splits = ";".join([task_config["train_split"] for task_config in finetuning_tasks.values()])
    finetuning_test_splits = ";".join([task_config["test_split"] for task_config in finetuning_tasks.values()])

    train_data_config = {
        'dataset_name': ";".join([os.path.join(code_root, f'merging/datasets/{task}.py') for task in finetuning_tasks.keys()]),
        'dataset_config_name': dataset_config_name,
        'dataset_train_split': finetuning_train_splits,
        'dataset_val_split': finetuning_test_splits,
    }

    multitask_train_job = HuggingfaceTrainingJob(
        code_root=code_root,
        config=config,
        train_data_config=train_data_config,
        num_epochs=1,
        mem_rqmt=24,
        time_rqmt=24,
        gpumem=16,
        dummy="rerun"
    )
    multitask_train_job.add_alias(f"multitask_model")
    tk.register_output(f"multitask_model", multitask_train_job.out_best_model)


    merged_model_task_arithmetic = TaskArithmeticJob(
        code_root,
        pretrained_model,
        finetuned_models,
        1.0,
    )

    delta_0 = 1e-12

    merged_model_ours = UncertaintyBasedGradientMatchingJob(
        code_root,
        pretrained_model,
        finetuned_models,
        pretrained_fisher,
        finetuned_fishers,
        delta_0,
        scaling_factor=1.0,
        scaling_factor_pretrained=pretraining_tasks["imdb"]["dataset_size"],
        scaling_factors_finetuned=[task["dataset_size"] for task in finetuning_tasks.values()],
    )

    for task, task_config in pretraining_tasks.items():

        search_data_config = {
            'dataset_name': os.path.join(code_root, f'merging/datasets/{task}.py'),
            'dataset_config_name': dataset_config_name,
            'dataset_test_split': task_config["test_split"].split("[")[0],
        }

        config["method"] = task_config["method"]

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=merged_model_task_arithmetic.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11
        )
        tk.register_output(f"example/{task}_task_arithmetic.metrics.json", search_job.out_metric_file)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=pretrain_train_job.out_best_model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11
        )
        tk.register_output(f"example/{task}_baseline.metrics.json", search_job.out_metric_file)


        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=multitask_train_job.out_best_model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11
        )
        tk.register_output(f"example/{task}_multitask.metrics.json", search_job.out_metric_file)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=merged_model_ours.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11,
            dummy="rerun_"
        )
        tk.register_output(f"example/{task}_gradient_matching.metrics.json", search_job.out_metric_file)


    for idx, (task, task_config) in enumerate(finetuning_tasks.items()):

        search_data_config = {
            'dataset_name': os.path.join(code_root, f'merging/datasets/{task}.py'),
            'dataset_config_name': dataset_config_name,
            'dataset_test_split': task_config["test_split"].split("[")[0],
        }

        config["method"] = task_config["method"]

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=merged_model_task_arithmetic.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11
        )
        tk.register_output(f"example/{task}_task_arithmetic.metrics.json", search_job.out_metric_file)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=finetuned_models[idx],
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11
        )
        tk.register_output(f"example/{task}_baseline.metrics.json", search_job.out_metric_file)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=multitask_train_job.out_best_model,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11
        )
        tk.register_output(f"example/{task}_multitask.metrics.json", search_job.out_metric_file)

        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path=merged_model_ours.out_model_path,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=8,
            time_rqmt=3,
            gpumem=11,
            dummy="rerun_"
        )
        tk.register_output(f"example/{task}_gradient_matching.metrics.json", search_job.out_metric_file)


async def async_main():
    await example()

async def py():
    await async_main()
