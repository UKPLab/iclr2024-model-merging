import argparse
import copy
import os
import shutil

import torch
from transformers import AutoModelForSequenceClassification

MODEL_TYPES = {
    "SEQ_CLS": AutoModelForSequenceClassification,
}


def get_model_class(args):
    return MODEL_TYPES[args.task_type]

def run(args):
    model_class = get_model_class(args)
    
    pretrained_model = model_class.from_pretrained(args.pretrained_model_name_or_path).state_dict()

    ft_models = []

    for model_path in args.ft_model_name_or_paths:
        ft_model = model_class.from_pretrained(model_path).state_dict()
        ft_models.append(ft_model)

    merged_model = {}

    new_model = model_class.from_pretrained(args.pretrained_model_name_or_path)

    with torch.no_grad():

        for n, p in new_model.named_parameters():
            merged_model[n] = pretrained_model[n]
            if not ("position_ids" in n or "attn.masked_bias" in n or p.dtype == torch.bool):

                summed = sum([ft_model[n] - pretrained_model[n] for ft_model in ft_models])

                merged_model[n] += float(args.scaling_factor) * summed

                p.data.copy_(merged_model[n])

    if args.pretrained_model_name_or_path is not None:
        if os.path.exists(args.pretrained_model_name_or_path):
            for file in os.listdir(args.pretrained_model_name_or_path):
                if not file.endswith(".safetensors"):
                    input_path = os.path.join(args.pretrained_model_name_or_path, file)
                    output_path = os.path.join(args.out_model_path, file)
                    if not os.path.isdir(input_path):
                        shutil.copy(input_path, output_path)
    else:
        raise NotImplementedError

    new_model.save_pretrained(args.out_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ft_model_name_or_paths",
        default=[],
        type=lambda x: x.split(","),
    )
    parser.add_argument(
        "--out_model_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--scaling_factor",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--task_type",
        default="SEQ_CLS",
        type=str,
    )
    parsed_args = parser.parse_args()

    run(parsed_args)