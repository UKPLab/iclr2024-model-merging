__all__ = ["ApplyG2PModelJob"]

import os
import subprocess as sp
from tempfile import mkstemp

from sisyphus import *

from i6_core.util import uopen

Path = setup_path(__package__)


class ApplyG2PModelJob(Job):
    """
    Apply a trained G2P on a word list file
    """

    __sis_hash_exclude__ = {"filter_empty_words": False}

    def __init__(
        self,
        g2p_model,
        word_list_file,
        variants_mass=1.0,
        variants_number=1,
        g2p_path=None,
        g2p_python=None,
        filter_empty_words=False,
        concurrent=1,
    ):
        """
        :param Path g2p_model:
        :param Path word_list_file: text file with a word each line
        :param float variants_mass:
        :param int variants_number:
        :param DelayedBase|str|None g2p_path:
        :param DelayedBase|str|None g2p_python:
        :param bool filter_empty_words: if True, creates a new lexicon file with no empty translated words
        :param int concurrent: split up word list file to parallelize job into this many instances
        """

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

        self.g2p_model = g2p_model
        self.g2p_path = g2p_path
        self.g2p_python = g2p_python
        self.variants_mass = variants_mass
        self.variants_number = variants_number
        self.word_list = word_list_file
        self.filter_empty_words = filter_empty_words
        self.concurrent = concurrent

        self.out_g2p_lexicon = self.output_path("g2p.lexicon")
        self.out_g2p_untranslated = self.output_path("g2p.untranslated")

        self.rqmt = {"cpu": 1, "mem": 1, "time": 2}

    def tasks(self):
        yield Task("split_word_list", mini_task=True)
        yield Task("run", rqmt=self.rqmt, args=range(1, self.concurrent + 1))
        yield Task("merge", mini_task=True)
        if self.filter_empty_words:
            yield Task("filter", mini_task=True)

    def split_word_list(self):
        num_digits = len(str(self.concurrent))
        sp.check_call(
            [
                "split",
                f"--number=l/{self.concurrent}",
                "--numeric-suffixes=1",
                f"--suffix-length={num_digits}",
                self.word_list.get_path(),
                "words.",
            ]
        )

    def run(self, task_id):
        g2p_lexicon_path = f"g2p.lexicon.{task_id}"
        g2p_untranslated_path = f"g2p.untranslated.{task_id}"
        num_digits = len(str(self.concurrent))
        word_list_path = f"words.{task_id:0{num_digits}d}"

        with uopen(g2p_lexicon_path, "wt") as out:
            with uopen(g2p_untranslated_path, "wt") as err:
                sp.check_call(
                    [
                        str(self.g2p_python),
                        str(self.g2p_path),
                        "-e",
                        "utf-8",
                        "-V",
                        str(self.variants_mass),
                        "--variants-number",
                        str(self.variants_number),
                        "-m",
                        self.g2p_model.get_path(),
                        "-a",
                        word_list_path,
                    ],
                    stdout=out,
                    stderr=err,
                )

    def merge(self):
        with uopen(self.out_g2p_lexicon, "wt") as f:
            sp.check_call(
                ["cat"] + [f"g2p.lexicon.{i}" for i in range(1, self.concurrent + 1)],
                stdout=f,
            )

        with uopen(self.out_g2p_untranslated, "wt") as f:
            sp.check_call(
                ["cat"]
                + [f"g2p.untranslated.{i}" for i in range(1, self.concurrent + 1)],
                stdout=f,
            )

    def filter(self):
        handle, tmp_path = mkstemp(dir=".", text=True)
        with uopen(self.out_g2p_lexicon, "rt") as lex, os.fdopen(
            handle, "wt"
        ) as fd_out:
            for line in lex:
                if len(line.strip().split("\t")) == 4:
                    fd_out.write(line)
        fd_out.close()

        os.remove(self.out_g2p_lexicon)
        os.rename(tmp_path, self.out_g2p_lexicon)

    @classmethod
    def hash(cls, kwargs):
        kwargs_copy = dict(**kwargs)
        kwargs_copy.pop("concurrent", None)
        return super().hash(kwargs_copy)
