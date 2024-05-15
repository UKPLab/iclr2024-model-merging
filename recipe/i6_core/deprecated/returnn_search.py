import copy
import os
import subprocess as sp

from sisyphus import Job, Task, gs, tk

from i6_core import util
from i6_core.returnn.config import ReturnnConfig


class ReturnnSearchJob(Job):
    """
    Given a model checkpoint, run search task with RETURNN
    """

    def __init__(
        self,
        search_data,
        model_checkpoint,
        returnn_config,
        *,
        output_mode="py",
        log_verbosity=3,
        device="gpu",
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """
        :param dict[str] search_data: dataset used for search
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param str output_mode: "txt" or "py"
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :param float|int time_rqmt: job time requirement in hours
        :param float|int mem_rqmt: job memory requirement in GB
        :param float|int cpu_rqmt: job cpu requirement in GB
        :param tk.Path|str|None returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param tk.Path|str|None returnn_root: path to the RETURNN src folder
        """
        assert isinstance(returnn_config, ReturnnConfig)
        kwargs = locals()
        del kwargs["self"]

        self.model_checkpoint = model_checkpoint

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )

        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_search_file = self.output_path("search_out")

        self.returnn_config = ReturnnSearchJob.create_returnn_config(**kwargs)
        self.returnn_config.post_config["search_output_file"] = self.out_search_file

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        config = self.returnn_config
        config.write(self.out_returnn_config_file.get_path())

        cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]

        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(
            tk.uncached_path(self.model_checkpoint.index_path)
        ), "Provided model does not exists: %s" % str(self.model_checkpoint)

    def run(self):
        call = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]
        sp.check_call(call)

    @classmethod
    def create_returnn_config(
        cls,
        search_data,
        model_checkpoint,
        returnn_config,
        output_mode,
        log_verbosity,
        device,
        **kwargs,
    ):
        """
        Creates search RETURNN config
        :param dict[str] search_data: dataset used for search
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param str output_mode: "txt" or "py"
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :rtype: ReturnnConfig
        """
        assert device in ["gpu", "cpu"]
        original_config = returnn_config.config
        assert "network" in original_config
        assert output_mode in ["py", "txt"]

        config = {
            "load": model_checkpoint.ckpt_path,
            "search_output_file_format": output_mode,
            "need_data": False,
            "search_do_eval": 0,
        }

        config.update(copy.deepcopy(original_config))  # update with the original config

        # override always
        config["task"] = "search"
        config["max_seq_length"] = 0

        if "search_data" in original_config:
            config["search_data"] = {
                **original_config["search_data"].copy(),
                **search_data,
            }
        else:
            config["search_data"] = search_data

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
        }

        post_config.update(copy.deepcopy(returnn_config.post_config))

        res = copy.deepcopy(returnn_config)
        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnSearchJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)
