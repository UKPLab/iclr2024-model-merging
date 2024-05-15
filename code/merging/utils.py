import json
import logging
import sys

import transformers
import numpy as np
import torch
from transformers.hf_argparser import DataClass
from transformers.trainer_utils import is_main_process

from transformers import PreTrainedModel
from merging.arguments import ModelArguments, DataPredictionArguments, DataTrainingArguments

from enum import Enum


class RunMode(Enum):
    TRAIN = 1
    PREDICT = 2


logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def default_dict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = default_dict2dict(v)
    return dict(d)


def randrange_excluding(a, b, excluded):
    """
    Samples a random number x, with a <= x < b with x != excluded and a <= excluded < b
    """
    assert a < b
    assert a <= excluded < b
    random_int = torch.randint(a, b - 1, ()).item()
    if random_int >= excluded:
        random_int += 1
    return random_int


def iterate_values_in_nested_dict(nested_dict):
    for value in nested_dict.values():
        if isinstance(value, dict):
            yield from iterate_values_in_nested_dict(value)
        else:
            yield value


def parse_arguments(raw_args, run_mode: RunMode) -> tuple[DataClass, ...]:
    if run_mode == RunMode.TRAIN:
        argument_classes = (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    else:
        argument_classes = (ModelArguments, DataPredictionArguments, Seq2SeqTrainingArguments)

    parser = HfArgumentParser(argument_classes)
    json_index = -1 if raw_args[-1].endswith(".json") and (len(
        raw_args) == 1 or not raw_args[-2].startswith('-') or '=' in raw_args[-2]) else 0
    if len(raw_args) > 0 and raw_args[json_index].endswith(".json"):
        with open(raw_args[json_index]) as fp:
            json_args_dict = json.load(fp)
        del raw_args[json_index]

        if run_mode == RunMode.TRAIN:
            train_parser = HfArgumentParser(Seq2SeqTrainingArguments)
            training_args_dict = vars(train_parser.parse_args(raw_args + ['--output_dir',
                                                                          json_args_dict['output_dir']]))
            training_args_dict.update(json_args_dict)
            json_args_dict = training_args_dict

        return parser.parse_dict(json_args_dict, allow_extra_keys=True)
    else:
        return parser.parse_args_into_dataclasses()


def update_model_args(model: PreTrainedModel, model_args: ModelArguments) -> PreTrainedModel:
    """
    Passes the model arguments to the model
    :param model:
    :param model_args:
    :return:
    """
    model.config.keys_to_ignore_at_inference = [
        "decoder_attentions"
    ]
    return model


def _setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
