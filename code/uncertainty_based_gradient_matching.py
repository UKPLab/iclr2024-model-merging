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

    if not "hessian" in args.pretrained_hessian_path:
        args.pretrained_hessian_path = os.path.join(args.pretrained_hessian_path, "hessian")
    pretrained_hessian = model_class.from_pretrained(args.pretrained_hessian_path).state_dict()

    ft_models, ft_hessians = [], []

    for model_path, hessian_path in zip(args.ft_model_name_or_paths, args.ft_hessian_paths):
        if not "hessian" in hessian_path:
            hessian_path = os.path.join(hessian_path, "hessian")

        ft_model = model_class.from_pretrained(model_path).state_dict()
        ft_models.append(ft_model)

        ft_hessian = model_class.from_pretrained(hessian_path).state_dict()
        ft_hessians.append(ft_hessian)

    merged_model = {}

    new_model = model_class.from_pretrained(args.pretrained_model_name_or_path)

    with torch.no_grad():

        for n, p in new_model.named_parameters():
            merged_model[n] = pretrained_model[n]
            if not ("position_ids" in n or "attn.masked_bias" in n or p.dtype == torch.bool):

                # common part of preconditioner: \bar{H}^{-1}
                preconditioner = 1. / (sum([float(args.scaling_factor) * float(scaling_factor_ft) * ft_hessian[n]
                                            for ft_hessian, scaling_factor_ft in zip(ft_hessians, args.scaling_factors_ft)]) \
                                + float(args.scaling_factor_pt) * pretrained_hessian[n] \
                                + args.delta_0)

                # summing the task-specific parts: \sum_t * \alpha_t * \bar{H}^{-1} * (H_{0+t}) * (\theta_ft - \llm)
                summed = sum([(preconditioner * (float(scaling_factor_ft) * ft_hessian[n] + args.delta_0 + float(args.scaling_factor_pt) * (pretrained_hessian[n])))
                                * (ft_model[n] - pretrained_model[n]) for ft_model, ft_hessian, scaling_factor_ft in zip(ft_models, ft_hessians, args.scaling_factors_ft)])

                merged_model[n] += float(args.scaling_factor) * summed

                p.data.copy_(merged_model[n])

    if args.pretrained_model_name_or_path is not None:
        if os.path.exists(args.pretrained_model_name_or_path):
            for file in os.listdir(args.pretrained_model_name_or_path):
                if not file.endswith(".safetensors") or file == "pytorch_model.bin":
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
        "--pretrained_hessian_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ft_model_name_or_paths",
        default=[],
        type=lambda x: x.split(","),
    )
    parser.add_argument(
        "--ft_hessian_paths",
        default=[],
        type=lambda x: x.split(","),
    )
    parser.add_argument(
        "--scaling_factors_ft",
        type=lambda x: x.split(","),
    )
    parser.add_argument(
        "--out_model_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--delta_0",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--scaling_factor_pt",
        default=1.0,
        type=float,
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