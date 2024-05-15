__all__ = [
    "ExtractPriorFromHDF5Job",
    "ReturnnComputePriorJob",
    "ReturnnRasrComputePriorJob",
]

import copy
import math
import os
import subprocess as sp

import h5py
import numpy as np
from sisyphus import *

import i6_core.rasr as rasr
import i6_core.util as util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.rasr_training import ReturnnRasrTrainingJob
from i6_core.returnn.training import ReturnnTrainingJob

Path = setup_path(__package__)


class ExtractPriorFromHDF5Job(Job):
    """
    Extracts the prior information from a RETURNN generated HDF file,
    and saves it in the RASR compatible .xml format
    """

    def __init__(self, prior_hdf_file, layer="output", plot_prior=False):
        """

        :param Path prior_hdf_file:
        :param str layer:
        :param bool plot_prior:
        """
        self.returnn_model = prior_hdf_file
        self.layer = layer
        self.plot_prior = plot_prior

        self.out_prior = self.output_path("prior.xml", cached=True)
        if self.plot_prior:
            self.out_prior_plot = self.output_path("prior.png")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        model = h5py.File(tk.uncached_path(self.returnn_model), "r")
        priors_set = model["%s/priors" % self.layer]

        priors_list = np.asarray(priors_set[:])

        with open(self.out_prior.get_path(), "wt") as out:
            out.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % priors_list.shape[0]
            )
            out.write(" ".join("%.20e" % math.log(s) for s in priors_list) + "\n")
            out.write("</vector-f32>")

        if self.plot_prior:
            import matplotlib.pyplot as plt

            xdata = range(len(priors_list))
            plt.semilogy(xdata, priors_list)

            plt.xlabel("emission idx")
            plt.ylabel("prior")
            plt.grid(True)
            plt.savefig(self.out_prior_plot.get_path())


class ReturnnComputePriorJob(Job):
    """
    Given a model checkpoint, run compute_prior task with RETURNN
    """

    def __init__(
        self,
        model_checkpoint,
        returnn_config,
        prior_data=None,
        *,
        log_verbosity=3,
        device="gpu",
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param dict[str]|None prior_data: dataset used to compute prior (None = use one train epoch)
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

        self.returnn_config = ReturnnComputePriorJob.create_returnn_config(**kwargs)

        self.out_returnn_config_file = self.output_path("returnn.config")

        self.out_prior_txt_file = self.output_path("prior.txt")
        self.out_prior_xml_file = self.output_path("prior.xml")
        self.out_prior_png_file = self.output_path("prior.png")

        self.returnn_config.post_config["output_file"] = self.out_prior_txt_file

        self.rqmt = {
            "gpu": 1 if device == "gpu" else 0,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)
        yield Task("plot", resume="plot", mini_task=True)

    def create_files(self):
        config = self.returnn_config
        config.write(self.out_returnn_config_file.get_path())

        cmd = self._get_run_cmd()
        util.create_executable("rnn.sh", cmd)

        # check here if model actually exists
        assert os.path.exists(
            tk.uncached_path(self.model_checkpoint.index_path)
        ), "Provided model does not exists: %s" % str(self.model_checkpoint)

    def run(self):
        cmd = self._get_run_cmd()
        sp.check_call(cmd)

        with open(self.out_prior_txt_file.get_path(), "rt") as f:
            merged_scores = np.loadtxt(f, delimiter=" ")

        with open(self.out_prior_xml_file.get_path(), "wt") as f:
            f.write(
                '<?xml version="1.0" encoding="UTF-8"?>\n<vector-f32 size="%d">\n'
                % len(merged_scores)
            )
            f.write(" ".join("%.20e" % s for s in merged_scores) + "\n")
            f.write("</vector-f32>")

    def plot(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(self.out_prior_txt_file.get_path(), "rt") as f:
            merged_scores = np.loadtxt(f, delimiter=" ")

        xdata = range(len(merged_scores))
        plt.semilogy(xdata, np.exp(merged_scores))
        plt.xlabel("emission idx")
        plt.ylabel("prior")
        plt.grid(True)
        plt.savefig(self.out_prior_png_file.get_path())

    def _get_run_cmd(self):
        return [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "rnn.py"),
            self.out_returnn_config_file.get_path(),
        ]

    @classmethod
    def create_returnn_config(
        cls,
        model_checkpoint,
        returnn_config,
        prior_data,
        log_verbosity,
        device,
        **kwargs,
    ):
        """
        Creates compute_prior RETURNN config
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param dict[str]|None prior_data: dataset used to compute prior (None = use one train epoch)
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :rtype: ReturnnConfig
        """
        assert device in ["gpu", "cpu"]
        original_config = returnn_config.config
        assert "network" in original_config

        config = copy.deepcopy(original_config)
        config["load"] = model_checkpoint.ckpt_path
        config["task"] = "compute_priors"

        if prior_data is not None:
            config["train"] = prior_data

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
            "returnn_config": ReturnnComputePriorJob.create_returnn_config(**kwargs),
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }
        return super().hash(d)


class ReturnnRasrComputePriorJob(ReturnnComputePriorJob, ReturnnRasrTrainingJob):
    """
    Given a model checkpoint, run compute_prior task with RETURNN using RASR Dataset
    """

    def __init__(
        self,
        train_crp,
        dev_crp,
        feature_flow,
        model_checkpoint,
        returnn_config,
        prior_data=None,
        alignment=None,
        *,  # args below are keyword only
        log_verbosity=3,
        device="gpu",
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=1,
        returnn_python_exe=None,
        returnn_root=None,
        # these are new parameters
        num_classes=None,
        disregarded_classes=None,
        class_label_file=None,
        buffer_size=200 * 1024,
        partition_epochs=None,
        extra_rasr_config=None,
        extra_rasr_post_config=None,
        use_python_control=True,
    ):
        """
        :param rasr.CommonRasrParameters train_crp:
        :param rasr.CommonRasrParameters dev_crp:
        :param rasr.FlowNetwork feature_flow: RASR flow file for feature extraction or feature cache
        :param Checkpoint model_checkpoint:  TF model checkpoint. see `ReturnnTrainingJob`.
        :param ReturnnConfig returnn_config: object representing RETURNN config
        :param dict[str]|None prior_data: dataset used to compute prior (None = use one train epoch)
        :param Path|None alignment: path to an alignment cache or cache bundle
        :param int log_verbosity: RETURNN log verbosity
        :param str device: RETURNN device, cpu or gpu
        :param float|int time_rqmt: job time requirement in hours
        :param float|int mem_rqmt: job memory requirement in GB
        :param float|int cpu_rqmt: job cpu requirement in GB
        :param tk.Path|str|None returnn_python_exe: path to the RETURNN executable (python binary or launch script)
        :param tk.Path|str|None returnn_root: path to the RETURNN src folder
        :param int num_classes:
        :param disregarded_classes:
        :param class_label_file:
        :param buffer_size:
        :param dict[str, int]|None partition_epochs: a dict containing the partition values for "train" and "dev"
        :param extra_rasr_config:
        :param extra_rasr_post_config:
        :param use_python_control:
        """
        datasets = self.create_dataset_config(
            train_crp, returnn_config, partition_epochs
        )
        returnn_config.config["train"] = datasets["train"]
        returnn_config.config["dev"] = datasets["dev"]
        super().__init__(
            model_checkpoint=model_checkpoint,
            returnn_config=returnn_config,
            time_rqmt=time_rqmt,
            mem_rqmt=mem_rqmt,
            cpu_rqmt=cpu_rqmt,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )

        kwargs = locals()
        del kwargs["self"]

        self.num_classes = num_classes
        self.alignment = alignment  # allowed to be None
        self.rasr_exe = rasr.RasrCommand.select_exe(
            train_crp.nn_trainer_exe, "nn-trainer"
        )

        del kwargs["train_crp"]
        del kwargs["dev_crp"]
        kwargs["crp"] = train_crp
        self.feature_flow = ReturnnRasrTrainingJob.create_flow(**kwargs)
        (
            self.rasr_train_config,
            self.rasr_train_post_config,
        ) = ReturnnRasrTrainingJob.create_config(**kwargs)
        kwargs["crp"] = dev_crp
        (
            self.rasr_dev_config,
            self.rasr_dev_post_config,
        ) = ReturnnRasrTrainingJob.create_config(**kwargs)

    def create_files(self):
        if self.num_classes is not None:
            if "num_outputs" not in self.returnn_config.config:
                self.returnn_config.config["num_outputs"] = {}
            self.returnn_config.config["num_outputs"]["classes"] = [
                util.get_val(self.num_classes),
                1,
            ]

        super().create_files()

        rasr.RasrCommand.write_config(
            self.rasr_train_config,
            self.rasr_train_post_config,
            "rasr.train.config",
        )
        rasr.RasrCommand.write_config(
            self.rasr_dev_config, self.rasr_dev_post_config, "rasr.dev.config"
        )

        self.feature_flow.write_to_file("feature.flow")

    def path_available(self, path):
        return self._sis_finished()

    @classmethod
    def hash(cls, kwargs):
        flow = ReturnnRasrTrainingJob.create_flow(**kwargs)
        kwargs = copy.copy(kwargs)
        train_crp = kwargs["train_crp"]
        dev_crp = kwargs["dev_crp"]
        del kwargs["train_crp"]
        del kwargs["dev_crp"]
        kwargs["crp"] = train_crp
        train_config, train_post_config = cls.create_config(**kwargs)
        kwargs["crp"] = dev_crp
        dev_config, dev_post_config = cls.create_config(**kwargs)

        datasets = ReturnnRasrTrainingJob.create_dataset_config(
            train_crp, kwargs["returnn_config"], kwargs["partition_epochs"]
        )
        kwargs["returnn_config"].config["train"] = datasets["train"]
        kwargs["returnn_config"].config["dev"] = datasets["dev"]
        returnn_config = ReturnnComputePriorJob.create_returnn_config(**kwargs)
        d = {
            "train_config": train_config,
            "dev_config": dev_config,
            "alignment_flow": flow,
            "returnn_config": returnn_config,
            "rasr_exe": train_crp.nn_trainer_exe,
            "returnn_python_exe": kwargs["returnn_python_exe"],
            "returnn_root": kwargs["returnn_root"],
        }

        return Job.hash(d)
