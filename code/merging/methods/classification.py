import numpy as np
from datasets import load_metric
from peft import AutoPeftModelForSequenceClassification
from scipy.special import log_softmax, softmax
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import PretrainedConfig, Trainer, AutoTokenizer, \
 DataCollatorWithPadding, AutoModelForSequenceClassification

from merging.methods.base import Method
from merging.methods.preprocessing.classification import SequenceClassificationPreprocessor
from merging.trainer.trainer_hessians import CustomTrainerForSquaredGradients

class SequenceClassificationMethod(Method):

    name = "sequence_classification"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = [
        ]
        self.peft_task_type = "SEQ_CLS"

    def compute_test_metrics(self, p):
        return self.compute_metrics(p)

    def compute_metrics(self, p):
        predictions = np.argmax(p.predictions, axis=-1)
        confidence = np.exp(np.max(p.predictions, axis=-1))
        accuracy = sum(predictions == p.label_ids) / len(p.label_ids)

        results = {
            "accuracy": round(accuracy, 5)
        }

        return results

    def preprocess_features(self, features):
        processor = SequenceClassificationPreprocessor(self.config, self.data_args, self.model_args, self.tokenizer)
        input_ids, labels = processor.preprocess(features)

        return_dict = {
            "input_ids": input_ids,
        }

        return_dict["labels"] = labels

        return return_dict

    def get_trainer_class(self):
        return Trainer
        
    def get_data_collator(self):
        return DataCollatorWithPadding(self.tokenizer)

    def get_model_class(self, config: PretrainedConfig):
        if self.optimizer_args.load_peft_model:
            return AutoPeftModelForSequenceClassification
        else:
            return AutoModelForSequenceClassification

class SquaredGradientsForSequenceClassificationMethod(SequenceClassificationMethod):

    name = "sequence_classification_squared_gradients"

    def get_trainer_class(self):
        return CustomTrainerForSquaredGradients