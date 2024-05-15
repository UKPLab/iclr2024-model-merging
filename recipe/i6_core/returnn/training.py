__all__ = [
    "ReturnnModel",
    "Checkpoint",
    "ReturnnTrainingJob",
    "ReturnnTrainingFromFileJob",
]

import copy
import os
import shutil
import subprocess as sp

from sisyphus import *

import i6_core.util as util
from .config import ReturnnConfig

Path = setup_path(__package__)


class ReturnnModel:
    """
    Defines a RETURNN model as config, checkpoint meta file and epoch

    This is deprecated, use :class:`Checkpoint` instead.
    """

    def __init__(self, returnn_config_file, model, epoch):
        """

        :param Path returnn_config_file: Path to a returnn config file
        :param Path model: Path to a RETURNN checkpoint (only the .meta for Tensorflow)
        :param int epoch:
        """
        self.returnn_config_file = returnn_config_file
        self.model = model
        self.epoch = epoch


class Checkpoint:
    """
    Checkpoint object which holds the (Tensorflow) index file path as tk.Path,
    and will return the checkpoint path as common prefix of the .index/.meta/.data[...]

    A checkpoint object should directly assigned to a RasrConfig entry (do not call `.ckpt_path`)
    so that the hash will resolve correctly
    """

    def __init__(self, index_path):
        """
        :param Path index_path:
        """
        self.index_path = index_path

    def _sis_hash(self):
        return self.index_path._sis_hash()

    @property
    def ckpt_path(self):
        return self.index_path.get_path()[: -len(".index")]

    def __str__(self):
        return self.ckpt_path

    def __repr__(self):
        return "'%s'" % self.ckpt_path


class ReturnnTrainingJob(Job):
    """
    Train a RETURNN model using the rnn.py entry point.

    Only returnn_config, returnn_python_exe and returnn_root influence the hash.

    The outputs provided are:

     - out_returnn_config_file: the finalized Returnn config which is used for the rnn.py call
     - out_learning_rates: the file containing the learning rates and training scores (e.g. use to select the best checkpoint or generate plots)
     - out_model_dir: the model directory, which can be used in succeeding jobs to select certain models or do combinations
        note that the model dir is DIRECTLY AVAILABLE when the job starts running, so jobs that do not have other conditions
        need to implement an "update" method to check if the required checkpoints are already existing
     - out_checkpoints: a dictionary containing all created checkpoints. Note that when using the automatic checkpoint cleaning
        function of Returnn not all checkpoints are actually available.
    """

    def __init__(
        self,
        returnn_config,
        *,  # args below are keyword only
        log_verbosity=3,
        device="gpu",
        num_epochs=1,
        save_interval=1,
        keep_epochs=None,
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        horovod_num_processes=None,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param ReturnnConfig returnn_config:
        :param int log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param str device: "cpu" or "gpu"
        :param int num_epochs: number of epochs to run, will also set `num_epochs` in the config file.
            Note that this value is NOT HASHED, so that this number can be increased to continue the training.
        :param int save_interval: save a checkpoint each n-th epoch
        :param list[int]|set[int]|None keep_epochs: specify which checkpoints are kept, use None for the RETURNN default
            This will also limit the available output checkpoints to those defined. If you want to specify the keep
            behavior without this limitation, provide `cleanup_old_models/keep` in the post-config and use `None` here.
        :param int|float time_rqmt:
        :param int|float mem_rqmt:
        :param int cpu_rqmt:
        :param int horovod_num_processes:
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """
        assert isinstance(returnn_config, ReturnnConfig)
        self.check_blacklisted_parameters(returnn_config)
        kwargs = locals()
        del kwargs["self"]

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )
        self.use_horovod = True if (horovod_num_processes is not None) else False
        self.horovod_num_processes = horovod_num_processes
        self.returnn_config = ReturnnTrainingJob.create_returnn_config(**kwargs)

        stored_epochs = list(range(save_interval, num_epochs, save_interval)) + [
            num_epochs
        ]
        if keep_epochs is None:
            self.keep_epochs = set(stored_epochs)
        else:
            self.keep_epochs = set(keep_epochs)

        suffix = ".meta" if self.returnn_config.get("use_tensorflow", False) else ""

        self.out_returnn_config_file = self.output_path("returnn.config")
        self.out_learning_rates = self.output_path("learning_rates")
        self.out_model_dir = self.output_path("models", directory=True)
        self.out_models = {
            k: ReturnnModel(
                self.out_returnn_config_file,
                self.output_path("models/epoch.%.3d%s" % (k, suffix)),
                k,
            )
            for k in stored_epochs
            if k in self.keep_epochs
        }
        if self.returnn_config.get("use_tensorflow", False):
            self.out_checkpoints = {
                k: Checkpoint(index_path)
                for k in stored_epochs
                if k in self.keep_epochs
                for index_path in [self.output_path("models/epoch.%.3d.index" % k)]
            }
        self.out_plot_se = self.output_path("score_and_error.png")
        self.out_plot_lr = self.output_path("learning_rate.png")

        self.returnn_config.post_config["model"] = os.path.join(
            self.out_model_dir.get_path(), "epoch"
        )

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

        if self.use_horovod:
            self.rqmt["cpu"] *= self.horovod_num_processes
            self.rqmt["gpu"] *= self.horovod_num_processes
            self.rqmt["mem"] *= self.horovod_num_processes

    def _get_run_cmd(self):
        run_cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]

        if self.use_horovod:
            run_cmd = [
                "mpirun",
                "-np",
                str(self.horovod_num_processes),
                "-bind-to",
                "none",
                "-map-by",
                "slot",
                "-mca",
                "pml",
                "ob1",
                "-mca",
                "btl",
                "^openib",
                "--report-bindings",
            ] + run_cmd

        return run_cmd

    def path_available(self, path):
        # if job is finished the path is available
        res = super().path_available(path)
        if res:
            return res

        # learning rate files are only available at the end
        if path == self.out_learning_rates:
            return super().path_available(path)

        # maybe the file already exists
        res = os.path.exists(path.get_path())
        if res:
            return res

        # maybe the model is just a pretrain model
        file = os.path.basename(path.get_path())
        directory = os.path.dirname(path.get_path())
        if file.startswith("epoch."):
            segments = file.split(".")
            pretrain_file = ".".join([segments[0], "pretrain", segments[1]])
            pretrain_path = os.path.join(directory, pretrain_file)
            return os.path.exists(pretrain_path)

        return False

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("plot", resume="plot", mini_task=True)

    def create_files(self):
        self.returnn_config.write(self.out_returnn_config_file.get_path())

        util.create_executable("rnn.sh", self._get_run_cmd())

    @staticmethod
    def _relink(src, dst):
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)

    def run(self):
        sp.check_call(self._get_run_cmd())

        lrf = self.returnn_config.get("learning_rate_file", "learning_rates")
        self._relink(lrf, self.out_learning_rates.get_path())

    def plot(self):
        def EpochData(learningRate, error):
            return {"learning_rate": learningRate, "error": error}

        with open(self.out_learning_rates.get_path(), "rt") as f:
            text = f.read()

        data = eval(text)

        epochs = list(sorted(data.keys()))
        train_score_keys = [
            k for k in data[epochs[0]]["error"] if k.startswith("train_score")
        ]
        dev_score_keys = [
            k for k in data[epochs[0]]["error"] if k.startswith("dev_score")
        ]
        dev_error_keys = [
            k for k in data[epochs[0]]["error"] if k.startswith("dev_error")
        ]

        train_scores = [
            [
                (epoch, data[epoch]["error"][tsk])
                for epoch in epochs
                if tsk in data[epoch]["error"]
            ]
            for tsk in train_score_keys
        ]
        dev_scores = [
            [
                (epoch, data[epoch]["error"][dsk])
                for epoch in epochs
                if dsk in data[epoch]["error"]
            ]
            for dsk in dev_score_keys
        ]
        dev_errors = [
            [
                (epoch, data[epoch]["error"][dek])
                for epoch in epochs
                if dek in data[epoch]["error"]
            ]
            for dek in dev_error_keys
        ]
        learing_rates = [data[epoch]["learning_rate"] for epoch in epochs]

        colors = ["#2A4D6E", "#AA3C39", "#93A537"]  # blue red yellowgreen

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        for ts in train_scores:
            ax1.plot([d[0] for d in ts], [d[1] for d in ts], "o-", color=colors[0])
        for ds in dev_scores:
            ax1.plot([d[0] for d in ds], [d[1] for d in ds], "o-", color=colors[1])
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("scores", color=colors[0])
        for tl in ax1.get_yticklabels():
            tl.set_color(colors[0])

        if len(dev_errors) > 0 and any(len(de) > 0 for de in dev_errors):
            ax2 = ax1.twinx()
            ax2.set_ylabel("dev error", color=colors[2])
            for de in dev_errors:
                ax2.plot([d[0] for d in de], [d[1] for d in de], "o-", color=colors[2])
            for tl in ax2.get_yticklabels():
                tl.set_color(colors[2])

        fig.savefig(fname=self.out_plot_se.get_path())

        fig, ax1 = plt.subplots()
        ax1.semilogy(epochs, learing_rates, "ro-")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("learning_rate")

        fig.savefig(fname=self.out_plot_lr.get_path())

    @classmethod
    def create_returnn_config(
        cls,
        returnn_config,
        log_verbosity,
        device,
        num_epochs,
        save_interval,
        keep_epochs,
        horovod_num_processes,
        **kwargs,
    ):
        assert device in ["gpu", "cpu"]

        res = copy.deepcopy(returnn_config)

        config = {
            "task": "train",
            "target": "classes",
            "learning_rate_file": "learning_rates",
        }

        post_config = {
            "device": device,
            "log": ["./returnn.log"],
            "log_verbosity": log_verbosity,
            "num_epochs": num_epochs,
            "save_interval": save_interval,
        }

        if horovod_num_processes is not None:
            config["use_horovod"] = True

        config.update(copy.deepcopy(returnn_config.config))
        if returnn_config.post_config is not None:
            post_config.update(copy.deepcopy(returnn_config.post_config))

        if keep_epochs is not None:
            if not "cleanup_old_models" in post_config or isinstance(
                post_config["cleanup_old_models"], bool
            ):
                assert (
                    post_config.get("cleanup_old_models", True) == True
                ), "'cleanup_old_models' can not be False if 'keep_epochs' is specified"
                post_config["cleanup_old_models"] = {"keep": keep_epochs}
            elif isinstance(post_config["cleanup_old_models"], dict):
                assert (
                    "keep" not in post_config["cleanup_old_models"]
                ), "you can only provide either 'keep_epochs' or 'cleanup_old_models/keep', but not both"
                post_config["cleanup_old_models"]["keep"] = keep_epochs
            else:
                assert False, "invalid type of cleanup_old_models: %s" % type(
                    post_config["cleanup_old_models"]
                )

        res.config = config
        res.post_config = post_config
        res.check_consistency()

        return res

    def check_blacklisted_parameters(self, returnn_config):
        """
        Check for parameters that should not be set in the config directly

        :param ReturnnConfig returnn_config:
        :return:
        """
        blacklisted_keys = [
            "log_verbosity",
            "device",
            "num_epochs",
            "save_interval",
            "keep_epochs",
        ]
        for key in blacklisted_keys:
            assert returnn_config.get(key) is None, (
                "please define %s only as parameter to ReturnnTrainingJob directly"
                % key
            )

    @classmethod
    def hash(cls, kwargs):
        d = {
            "returnn_config": ReturnnTrainingJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        if kwargs["horovod_num_processes"] is not None:
            d["horovod_num_processes"] = kwargs["horovod_num_processes"]

        return super().hash(d)


class ReturnnTrainingFromFileJob(Job):
    """
    The Job allows to directly execute returnn config files. The config files have to have the line
    `ext_model = config.value("ext_model", None)` and `model = ext_model` to correctly set the model path

    If the learning rate file should be available, add
    `ext_learning_rate_file = config.value("ext_learning_rate_file", None)` and
    `learning_rate_file = ext_learning_rate_file`

    Other externally controllable parameters may also defined in the same way, and can be set by providing the parameter
    value in the parameter_dict. The "ext_" prefix is used for naming convention only, but should be used for all
    external parameters to clearly mark them instead of simply overwriting any normal parameter.

    Also make sure that task="train" is set.
    """

    def __init__(
        self,
        returnn_config_file,
        parameter_dict,
        time_rqmt=4,
        mem_rqmt=4,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param tk.Path|str returnn_config_file: a returnn training config file
        :param dict parameter_dict: provide external parameters to the rnn.py call
        :param int|str time_rqmt:
        :param int|str mem_rqmt:
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """

        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.returnn_config_file_in = returnn_config_file
        self.parameter_dict = parameter_dict
        if self.parameter_dict is None:
            self.parameter_dict = {}

        self.returnn_config_file = self.output_path("returnn.config")

        self.rqmt = {"gpu": 1, "cpu": 2, "mem": mem_rqmt, "time": time_rqmt}

        self.learning_rates = self.output_path("learning_rates")
        self.model_dir = self.output_path("models", directory=True)

        self.parameter_dict["ext_model"] = tk.uncached_path(self.model_dir) + "/epoch"
        self.parameter_dict["ext_learning_rate_file"] = tk.uncached_path(
            self.learning_rates
        )

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    def path_available(self, path):
        # if job is finised the path is available
        res = super().path_available(path)
        if res:
            return res

        # learning rate files are only available at the end
        if path == self.learning_rates:
            return super().path_available(path)

        # maybe the file already exists
        res = os.path.exists(path.get_path())
        if res:
            return res

        # maybe the model is just a pretrain model
        file = os.path.basename(path.get_path())
        directory = os.path.dirname(path.get_path())
        if file.startswith("epoch."):
            segments = file.split(".")
            pretrain_file = ".".join([segments[0], "pretrain", segments[1]])
            pretrain_path = os.path.join(directory, pretrain_file)
            return os.path.exists(pretrain_path)

        return False

    def get_parameter_list(self):
        parameter_list = []
        for k, v in sorted(self.parameter_dict.items()):
            if isinstance(v, tk.Variable):
                v = v.get()
            elif isinstance(v, tk.Path):
                v = tk.uncached_path(v)
            elif isinstance(v, (list, dict, tuple)):
                v = '"%s"' % str(v).replace(" ", "")

            if isinstance(v, (float, int)) and v < 0:
                v = "+" + str(v)
            else:
                v = str(v)

            parameter_list.append("++%s" % k)
            parameter_list.append(v)

        return parameter_list

    def create_files(self):
        # returnn
        shutil.copy(
            tk.uncached_path(self.returnn_config_file_in),
            tk.uncached_path(self.returnn_config_file),
        )

        parameter_list = self.get_parameter_list()
        cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.returnn_config_file.get_path(),
        ] + parameter_list

        util.create_executable("rnn.sh", cmd)

    def run(self):
        sp.check_call(["./rnn.sh"])

    @classmethod
    def hash(cls, kwargs):

        d = {
            "returnn_config_file": kwargs["returnn_config_file"],
            "parameter_dict": kwargs["parameter_dict"],
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return super().hash(d)
