import copy
import os
import shutil
import subprocess as sp
import copy
import json
import torch

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)

class TaskArithmeticJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        finetuned_models,
        scaling_factor=1.0,
        task_type="SEQ_CLS",
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=0,
        **kwargs
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.finetuned_models = finetuned_models
        self.scaling_factor = scaling_factor
        self.task_type = task_type

        self.out_model_path = self.output_path("model", directory=True)
        self.python_exe = gs.PYTHON_EXE

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def __get_path_or_str(self, path):
        if isinstance(path, str):
            return path
        else:
            return path.get_path()

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.python_exe),
            os.path.join(tk.uncached_path(self.code_root), "task_arithmetic.py"),
            '--pretrained_model_name_or_path', self.__get_path_or_str(self.pretrained_model_name_or_path),
            '--ft_model_name_or_paths', ",".join([self.__get_path_or_str(model) for model in self.finetuned_models]),
            '--out_model_path', self.__get_path_or_str(self.out_model_path),
            '--scaling_factor', str(self.scaling_factor),
            '--task_type', self.task_type
        ]
        return run_cmd


    def run(self):
        sp.check_call(self._get_run_cmd())

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class UncertaintyBasedGradientMatchingJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        finetuned_model_name_or_paths,
        pretrained_hessian_path,
        finetuned_hessian_paths,
        delta_0,
        task_type="SEQ_CLS",
        scaling_factor=1.0,
        scaling_factor_pretrained=1.0,
        scaling_factors_finetuned=[],
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=0,
        **kwargs
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.finetuned_model_name_or_paths = finetuned_model_name_or_paths
        self.pretrained_hessian_path = pretrained_hessian_path
        self.finetuned_hessian_paths = finetuned_hessian_paths
        self.delta_0 = delta_0
        self.scaling_factor = scaling_factor
        self.scaling_factor_pretrained = scaling_factor_pretrained
        self.scaling_factors_finetuned = scaling_factors_finetuned
        self.task_type = task_type

        self.out_model_path = self.output_path("model", directory=True)

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

        self.python_exe = gs.PYTHON_EXE

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def __get_path_or_str(self, path):
        if isinstance(path, str):
            return path
        else:
            return path.get_path()

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.python_exe),
            os.path.join(tk.uncached_path(self.code_root), "uncertainty_based_gradient_matching.py"),
            '--pretrained_model_name_or_path', self.__get_path_or_str(self.pretrained_model_name_or_path),
            '--pretrained_hessian_path', self.__get_path_or_str(self.pretrained_hessian_path),
            '--out_model_path', self.__get_path_or_str(self.out_model_path),
            '--delta_0', str(self.delta_0),
            '--scaling_factor', str(self.scaling_factor),
            '--scaling_factor_pt', str(self.scaling_factor_pretrained),
            '--ft_model_name_or_paths', ",".join([self.__get_path_or_str(model) for model in self.finetuned_model_name_or_paths]),
            '--ft_hessian_paths', ",".join([self.__get_path_or_str(hessian) for hessian in self.finetuned_hessian_paths]),
            '--task_type', self.task_type,
            '--scaling_factors_ft', ",".join([str(scaling_factor) for scaling_factor in self.scaling_factors_finetuned])
        ]

        return run_cmd


    def run(self):
        sp.check_call(self._get_run_cmd())

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)
