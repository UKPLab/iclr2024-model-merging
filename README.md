# Model Merging by Uncertainty-Based Gradient Matching

This repository contains code and resources related to the paper "Model Merging by Uncertainty-Based Gradient Matching", published as a poster at ICLR 2024.

A description of how to use this implementation is found below.

If you use this repository and our work, please cite

```
@inproceedings{
    daheim2024model,
    title={Model Merging by Uncertainty-Based Gradient Matching},
    author={Nico Daheim and Thomas M{\"o}llenhoff and Edoardo Ponti and Iryna Gurevych and Mohammad Emtiyaz Khan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=D7KJmfEDQP}
}
```

> **Abstract:** Models trained on different datasets can be merged by a weighted-averaging of their parameters, but why does it work and when can it fail? Here, we connect the inaccuracy of weighted-averaging to mismatches in the gradients and propose a new uncertainty-based scheme to improve the performance by reducing the mismatch. The connection also reveals implicit assumptions in other schemes such as averaging, task arithmetic, and Fisher-weighted averaging. Our new method gives consistent improvements for large language models and vision transformers, both in terms of performance and robustness to hyperparameters.

Contact person: Nico Daheim, nico.daheim@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Please don't hesitate to contact us in case of questions, or to report issues.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## Contents
  1. [Environment Set-up](#environment-set-up)
  2. [Running the code](#running-the-code)
  3. [Code Structure](#code-structure)

## Environment Set-up
Requirements are found in `requirements.txt`.

To ensure that the experiments are consistent and comparable, we use the [sisyphus](https://github.com/rwth-i6/sisyphus) workflow manager.

Sisyphus requires 3 folders (depending on the cluster set-up, you might want to symlink them to other folders, for example to use partitions optimised for large files):
  1. `alias`: It's possible to identify aliases for each job to identify it quickly (as a default, a hash is appended to the jobclass name as an identifier), and sisyphus adds a symlink to the job under the alias.
  2. `output`: `tk.register_output("name", job_class.file)` registers an output under the filename `name` in the output folder that symlinks to `job_class.file`
  3. `work`: All jobs will be placed here under their hash.

## Running the code

### Merging

If you are only interested in running merging with already-trained models and their hessian estimates, an example is found in `scripts/uncertainty_based_gradient_matching.sh` which invokes `code/uncertainty_based_gradient_matching.py` with the following parameters:

- `pretrained_model_name_or_path`: path to `\theta_llm`, i.e. the pretrained model that all models were finetuned from.
- `pretrained_hessian_path`: path to the hessian (estimate, for example squared gradients approximation of the Fisher) of the pretrained model `H_0`
- `ft_model_name_or_paths`: comma-separated list to all fine-tuned task models `\theta_t`.
- `ft_hessian_paths`: comma-separated list to all hessians of the fine-tuned task models `H_t`.
- `scaling_factor_pt`: scaling factor for the pretrained hessian. This is mostly for convenience, for example, to multiply the average squared gradients approximation of the Fisher by the dataset size.
- `scaling_factors_ft`: comma-separated list of scaling factors for each hessian estimate, again for convenience.
- `out_model_path`: path to the merged model.
- `delta_0`: From theory this should be the weight decay factor of the pretrained model divided by the training data size. In practice, setting this to a small value like `1e-12` works well.
- `scaling_factor`: Scaling factor for each task vector (`\theta_t - \theta_llm`).
- `task_type`: Defaults to `SEQ_CLS` to load `AutoModelForSequenceClassification` from huggingface.

### Running the code with the Sisyphus workflow scheduler 

Using the code is as easy as invoking a sisyphus *config* by, for example using: ```sis --config config/example.py m``` which starts the manager that guides you through starting jobs and schedules all jobs depending on them automatically once they are finished!
The example config reproduces our Table 1 for Task Arithmetic (with scaling factor 1.0) and our uncertainty-based gradient matching with only 1 epoch of training per task on a subset of each data.
The only thing needed for this is to fill all required fields, such as the python path, in `settings.py`.

The code can also be ran without it, relying solely on shell scripts, as is described later.

If you want to write custom configs, you can use the existing `Job` objects that define an experiment to be scheduled. For example, training a model might involve multiple subjobs, such as downloading data, saving it on disk, and then training the actual model.
These are defined in the `recipe` folder. For example, for training, you may use `HuggingfaceTrainingJob` found under `recipe/ukp/huggingface/training.py`.

The `TrainingJob` relies on configs that define all necessary information: method, model, datasets, hyperparameters, as shown in the dictionaries in `config/example.py`.

This way, sisyphus will automatically take care of creating all files (your configs are stored in the job folder, for example), starting the job, etc.
Also, hyperparameters for job scheduling, like the time, cpu memory, gpu memory, are all taken care of.

### Running the code with shell scripts

Running the code with only shell scripts is described in examples in `scripts/`.

For example, to train a model, one has to invoke `python3 code/train.py train_config.json`, where the `train_config.json` defines all parameters like model, dataset, dataset split, batch size, etc.

To use this model for inference one can invoke `python3 code/predict.py search_config.json` by pointing to the trained model and dataset. This will also take care of calculating metrics.
This way, the squared gradients approximation of the Fisher can also be run by specifying the method `sequence_classification_squared_gradients`, which will save the squared gradients in a dedicated folder.

## Code Structure
The code is mainly based on the concept of ''methods'' that are found in the `/code/merging/methods/` folder which wrap all of the functionality needed to reproduce a certain method:
  1. Defining and loading Trainer and Data Collator classes
  2. Loading all datasets
  3. Defining and applying the preprocessing methods, defined in `/code/merging/methods/preprocessing`

To understand how the method classes are structured it's best to check `code/merging/methods/base.py` which defines a base class from which all methods inherit.

The main entry point for the code is `/code/merging/main.py` that handles loading method classes, models, and running the Trainers.
