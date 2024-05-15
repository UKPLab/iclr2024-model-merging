import copy
import os
import shutil
import subprocess as sp
import copy
import json

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)


class HuggingfaceSearchJob(Job):
  """
  Train a Huggingface transformer model
  """
  __sis_hash_exclude__ = {
    'keep_only_best': False,
    'distributed': False,
    'sbatch_args': None,
  }

  def __init__(
      self,
      code_root,
      model_path,
      config,
      search_data_config,
      *,  # args below are keyword only
      time_rqmt=4,
      mem_rqmt=4,
      cpu_rqmt=2,
      gpu_rqmt=1,
      python_exe=None,
      sbatch_args=None,
      gpumem=0,
      **kwargs
  ):
    """
    :param code_root: Root directory for the training scripts. Expected to contain a training script.
    :param config:
    :param num_epochs:
    :param time_rqmt:
    :param mem_rqmt:
    :param cpu_rqmt:
    :param gpu_rqmt:
    """

    self.code_root = code_root
    self.model_path = model_path
    self.config = config
    self.search_data_config = search_data_config
    self.python_exe = (python_exe if python_exe is not None else gs.PYTHON_EXE)

    if gpu_rqmt > 1:
      sbatch_args = "-P multigpu"
    elif sbatch_args is None:
      sbatch_args = []

    self.rqmt = {
      "gpu": gpu_rqmt,
      "cpu": cpu_rqmt,
      "mem": mem_rqmt,
      "time": time_rqmt,
      "gpumem": gpumem,
      "sbatch_args": sbatch_args,
    }

    self.out_config_file = self.output_path("search_config.json")
    self.out_metric_file = self.output_path("metrics.json")
    self.out_search_file = self.output_path("search_output.json")
    self.out_checkpoints_dir = self.output_path("checkpoints", directory=True)

    self._update_config()

  def _update_config(self):
    fixed_config = {
      'metric_output_file': self.out_metric_file,
      'prediction_output_file': self.out_search_file,
      'output_dir': self.out_checkpoints_dir.get_path(),
    }
    assert fixed_config.keys().isdisjoint(self.config.keys())
    self.config = copy.deepcopy(self.config)
    self.config.update(fixed_config)
    # Overwrite model path
    self.config['model_name_or_path'] = self.model_path
    self.config['config_name'] = None
    self.config['tokenizer_name'] = None
    assert self.config.keys().isdisjoint(self.search_data_config.keys())

  def _get_run_cmd(self):
      run_cmd = [
          tk.uncached_path(self.python_exe),
          os.path.join(tk.uncached_path(self.code_root), "predict.py"),
          self.out_config_file.get_path(),
      ]
      return run_cmd

  def create_files(self):
    instanciated_config = instanciate_delayed({
      **copy.deepcopy(self.config),
      **copy.deepcopy(self.search_data_config),
    })
    with util.uopen(self.out_config_file, 'wt') as fp:
      json.dump(instanciated_config, fp)

    util.create_executable("run.sh", self._get_run_cmd())

  def run(self):
    sp.check_call(self._get_run_cmd())

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
