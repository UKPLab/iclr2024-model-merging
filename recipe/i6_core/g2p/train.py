__all__ = ["TrainG2PModelJob"]

import os
import subprocess as sp

from sisyphus import *

from i6_core.util import uopen

Path = setup_path(__package__)


class TrainG2PModelJob(Job):
    """
    Train a G2P model using Sequitur

    see https://github.com/sequitur-g2p/sequitur-g2p
    """

    def __init__(
        self,
        g2p_lexicon,
        num_ramp_ups=4,
        min_iter=1,
        max_iter=60,
        devel="5%",
        size_constrains="0,1,0,1",
        extra_args=None,
        g2p_path=None,
        g2p_python=None,
    ):
        """

        :param Path g2p_lexicon: g2p_lexicon for training, use BlissLexiconToG2PLexiconJob to generate a g2p_lexicon
            from a bliss lexicon
        :param int num_ramp_ups: number of global ramp-ups (n-gram-iness)
        :param int min_iter: minimum iterations per ramp-up
        :param int max_iter: maximum iteration sper ramp-up
        :param str devel: passed as -d argument, percent of train lexicon held out as validation set
        :param str size_constrains: passed as -s argument,
            multigrams must have l1 ... l2 left-symbols and r1 ... r2 right-symbols
        :param list[str] extra_args: extra cmd arguments that are passed to the g2p process
        :param DelayedBase|str|None g2p_path: path to the g2p installation. If None, searches for a global G2P_PATH,
            and uses the default binary path if not existing.
        :param DelayedBase|str|None g2p_python: path to the g2p python binary. If None, searches for a global G2P_PYTHON,
            and uses the default python binary if not existing.
        """
        if extra_args is None:
            extra_args = []
        if g2p_path is None:
            g2p_path = (
                tk.gs.G2P_PATH
                if hasattr(tk.gs, "G2P_PATH")
                else os.path.join(os.path.dirname(gs.SIS_COMMAND[0]), "g2p.py")
            )
        if g2p_python is None:
            g2p_python = (
                tk.gs.G2P_PYTHON if hasattr(tk.gs, "G2P_PYTHON") else gs.SIS_COMMAND[0]
            )

        self.g2p_lexicon = g2p_lexicon
        self.num_ramp_ups = num_ramp_ups
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.devel = devel
        self.size_constrains = size_constrains
        self.extra_args = extra_args
        self.g2p_path = g2p_path
        self.g2p_python = g2p_python

        self.out_g2p_models = [
            self.output_path("model-%d" % idx) for idx in range(self.num_ramp_ups + 1)
        ]
        self.out_error_rates = [
            self.output_var("err-%d" % idx) for idx in range(self.num_ramp_ups + 1)
        ]
        self.out_best_model = self.output_path("model-best")
        self.out_best_error_rate = self.output_var("err-best")

        self.rqmt = {
            "time": max(0.5, (self.max_iter / 20.0) * (self.num_ramp_ups + 1)),
            "cpu": 1,
            "mem": 2,
        }

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        for idx in range(self.num_ramp_ups + 1):
            if os.path.exists(self.out_g2p_models[idx].get_path()):
                continue

            args = [
                str(self.g2p_python),
                str(self.g2p_path),
                "-e",
                "utf-8",
                "-i",
                str(self.min_iter),
                "-I",
                str(self.max_iter),
                "-d",
                self.devel,
                "-s",
                self.size_constrains,
                "-n",
                "tmp-model",
                "-S",
                "-t",
                self.g2p_lexicon.get_path(),
            ]
            if idx > 0:
                args += ["-r", "-m", self.out_g2p_models[idx - 1].get_path()]
            args += self.extra_args

            if os.path.exists("tmp-model"):
                os.unlink("tmp-model")

            with uopen("stdout.%d" % idx, "w") as out:
                sp.check_call(args, stdout=out)

            with uopen("stdout.%d" % idx, "rt") as log:
                for line in log:
                    if "total symbol errors" in line:
                        error_rate = float(line.split("(")[1].split("%")[0])
                        self.out_error_rates[idx].set(error_rate)

            os.rename("tmp-model", self.out_g2p_models[idx].get_path())

        best = min(
            ((idx, err_var.get()) for idx, err_var in enumerate(self.out_error_rates)),
            key=lambda t: t[1],
        )
        os.symlink("model-%d" % best[0], self.out_best_model.get_path())
        self.out_best_error_rate.set(best[1])
