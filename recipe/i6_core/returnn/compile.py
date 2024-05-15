__all__ = ["CompileTFGraphJob", "CompileNativeOpJob"]

from sisyphus import *

Path = setup_path(__package__)

import copy
import logging
import os
import shutil
import subprocess as sp

import i6_core.util as util

from .config import ReturnnConfig


class CompileTFGraphJob(Job):
    """
    This Job is a wrapper around the RETURNN tool comptile_tf_graph.py

    """

    __sis_hash_exclude__ = {"device": None}

    def __init__(
        self,
        returnn_config,
        train=0,
        eval=0,
        search=0,
        verbosity=4,
        device=None,
        summaries_tensor_name=None,
        output_format="meta",
        returnn_python_exe=None,
        returnn_root=None,
    ):
        """

        :param ReturnnConfig|Path|str returnn_config: Path to a RETURNN config file
        :param int train:
        :param int eval:
        :param int search:
        :param int log_verbosity: RETURNN log verbosity from 1 (least verbose) to 5 (most verbose)
        :param str|None device: optimize graph for cpu or gpu. If `None`, defaults to cpu for current RETURNN.
            For any RETURNN version before `cd4bc382`, the behavior will depend on the `device` entry in the
            `returnn_conig`, or on the availability of a GPU on the execution host if not defined at all.
        :param summaries_tensor_name:
        :param str output_format: graph output format, one of ["pb", "pbtxt", "meta", "metatxt"]
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        """
        self.returnn_config = returnn_config
        self.train = train
        self.eval = eval
        self.search = search
        self.verbosity = verbosity
        self.device = device
        self.summaries_tensor_name = summaries_tensor_name
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.out_graph = self.output_path("graph.%s" % output_format)
        self.out_model_params = self.output_var("model_params.pickle", pickle=True)
        self.out_state_vars = self.output_var("state_vars.pickle", pickle=True)

        self.rqmt = None

    def tasks(self):
        if self.rqmt:
            yield Task("run", resume="run", rqmt=self.rqmt)
        else:
            yield Task("run", resume="run", mini_task=True)

    def run(self):
        if isinstance(self.returnn_config, tk.Path):
            returnn_config_path = self.returnn_config.get_path()

        elif isinstance(self.returnn_config, ReturnnConfig):
            returnn_config_path = "returnn.config"
            self.returnn_config.write(returnn_config_path)

        else:
            returnn_config_path = self.returnn_config

        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(
                tk.uncached_path(self.returnn_root), "tools/compile_tf_graph.py"
            ),
            returnn_config_path,
            "--train=%d" % self.train,
            "--eval=%d" % self.eval,
            "--search=%d" % self.search,
            "--verbosity=%d" % self.verbosity,
            "--output_file=%s" % self.out_graph.get_path(),
            "--output_file_model_params_list=model_params",
            "--output_file_state_vars_list=state_vars",
        ]
        if self.device is not None:
            args.append("--device=%s" % self.device)
        if self.summaries_tensor_name is not None:
            args.append("--summaries_tensor_name=%s" % self.summaries_tensor_name)

        util.create_executable("run.sh", args)

        sp.check_call(args)

        with open("model_params", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.out_model_params.set(lines)
        with open("state_vars", "rt") as input:
            lines = [l.strip() for l in input if len(l.strip()) > 0]
            self.out_state_vars.set(lines)

    @classmethod
    def hash(cls, kwargs):
        c = copy.copy(kwargs)
        del c["verbosity"]
        return super().hash(c)


class CompileNativeOpJob(Job):
    """
    Compile a RETURNN native op into a shared object file.
    """

    __sis_hash_exclude__ = {"search_numpy_blas": True, "blas_lib": None}

    def __init__(
        self,
        native_op,
        returnn_python_exe=None,
        returnn_root=None,
        search_numpy_blas=True,
        blas_lib=None,
    ):
        """
        :param str native_op: Name of the native op to compile (e.g. NativeLstm2)
        :param Path|str returnn_python_exe: file path to the executable for running returnn (python binary or .sh)
        :param Path|str returnn_root: file path to the RETURNN repository root folder
        :param bool search_numpy_blas: search for blas lib in numpy's .libs folder
        :param Path|str blas_lib: explicit path to the blas library to use
        """
        self.native_op = native_op
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )
        self.search_numpy_blas = search_numpy_blas
        self.blas_lib = blas_lib

        self.out_op = self.output_path("%s.so" % native_op)
        self.out_grad_op = self.output_path("GradOf%s.so" % native_op)

        self.rqmt = None

    def tasks(self):
        if self.rqmt is None:
            yield Task("run", resume="run", mini_task=True)
        else:
            yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        cmd = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(
                tk.uncached_path(self.returnn_root), "tools/compile_native_op.py"
            ),
            "--native_op",
            self.native_op,
            "--output_file",
            "compile.out",
        ]
        if not self.search_numpy_blas:
            cmd += ["--no_search_for_numpy_blas"]
        if self.blas_lib is not None:
            cmd += ["--blas_lib", tk.uncached_path(self.blas_lib)]
        logging.info(cmd)

        util.create_executable(
            "compile.sh", cmd
        )  # convenience file for manual execution
        sp.run(cmd, check=True)

        with open("compile.out", "rt") as f:
            files = [l.strip() for l in f]

        if len(files) > 0:
            shutil.move(files[0], self.out_op.get_path())
        if len(files) > 1:
            shutil.move(files[1], self.out_grad_op.get_path())
