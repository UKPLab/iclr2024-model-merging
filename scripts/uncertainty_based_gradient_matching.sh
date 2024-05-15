#!/usr/bin/env bash
/path/to/python3 \
 /path/to/iclr2024-model-merging/code/uncertainty_based_gradient_matching.py \
 --pretrained_model_name_or_path /path/to/roberta-base-imdb \
 --pretrained_hessian_path /path \
 --out_model_path gradient_matched_model \
 --delta_0 1e-12 \
 --scaling_factor 1.0 \
 --scaling_factor_pt 25000 \
 --ft_model_name_or_paths /path/to/roberta-base-yelp/,/path/to/roberta-base-rt/,/path/to/roberta-base-sst2/,/path/to/roberta-base-amazon/ \
 --ft_hessian_paths /path/to/roberta-base-yelp-fisher/,/path/to/roberta-base-rt-fisher/,/path/to/roberta-base-sst2-fisher/,/path/to/roberta-base-amazon-fisher/ \
 --task_type SEQ_CLS \
 --scaling_factors_ft 112000,8530,67349,180000