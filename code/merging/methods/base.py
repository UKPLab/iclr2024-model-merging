import abc
from enum import Enum

from datasets import Dataset, load_dataset, concatenate_datasets
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import AutoConfig, DataCollatorWithPadding, EvalPrediction, Trainer

class Method(abc.ABC):
    
    def __init__(self, model_args, data_args, optimizer_args, config, tokenizer):
        self.model_args = model_args
        self.data_args = data_args
        self.optimizer_args = optimizer_args
        self.config = config
        self.tokenizer = tokenizer

        tokenizer.add_special_tokens({
            "additional_special_tokens": sorted(self.get_special_tokens())
            })
        self.metrics = []

    def get_special_tokens(self):
        return [
        ]

    @abc.abstractmethod
    def get_model_class(self, config):
        raise NotImplementedError()

    def get_model(self, run_mode, config):
        model_class = self.get_model_class(config)
        if self.model_args.model_name_or_path is not None:
            if self.optimizer_args.load_peft_model:
                model = model_class.from_pretrained(
                    self.model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                    config=config,
                    cache_dir=self.model_args.cache_dir,
                    revision=self.model_args.model_revision,
                    use_auth_token=self.model_args.use_auth_token,
                    is_trainable=True,#self.data_args.is_training,
                )
            else:
                model = model_class.from_pretrained(
                    self.model_args.model_name_or_path,
                    from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
                    config=config,
                    cache_dir=self.model_args.cache_dir,
                    revision=self.model_args.model_revision,
                    use_auth_token=self.model_args.use_auth_token,
                )
        else:
            print("Initializing model from scratch")
            model_config = AutoConfig.from_pretrained(self.model_args.config_name)
            model = model_class.from_config(model_config)
        model.resize_token_embeddings(len(self.tokenizer))
        print(f"# Parameters: {model.num_parameters()}")
        if self.optimizer_args.use_peft:
            peft_config = LoraConfig(
                task_type=self.peft_task_type, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
        return model

    @abc.abstractmethod
    def preprocess_features(self, features):
        raise NotImplementedError()

    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)

    def get_trainer_class(self):
        return Trainer

    def postprocess_predictions(self, p, dataset):
        return p

    @abc.abstractmethod
    def compute_metrics(self, p: EvalPrediction):
        raise NotImplementedError()

    def _get_dataset(self, split, config_name=None):
        all_datasets = []
        if config_name is None:
            for dataset_name, local_split in zip(
                self.data_args.dataset_name.split(";"), 
                split.split(";")
            ):
                dataset = load_dataset(
                    dataset_name,
                    self.data_args.dataset_config_name,
                    split=local_split,
                    cache_dir=self.model_args.cache_dir,
                    data_files=self.data_args.dataset_data_files
                )
                all_datasets.append(dataset)

        dataset = concatenate_datasets(all_datasets)

        old_eval_column_names = dataset.column_names

        dataset = dataset.map(
            self.preprocess_features,
            batched=True,
            batch_size=5000,
            load_from_cache_file=False,
            remove_columns=old_eval_column_names,
            )
        return dataset

    def get_train_dataset(self):
        return self._get_dataset(self.data_args.dataset_train_split)

    def get_test_dataset(self):
        return self._get_dataset(self.data_args.dataset_test_split)

    def get_validation_dataset(self):
        return self._get_dataset(self.data_args.dataset_val_split)
