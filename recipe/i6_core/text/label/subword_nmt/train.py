__all__ = ["TrainBPEModelJob", "ReturnnTrainBpeJob"]

import gzip
import subprocess as sp
import os
import sys

from sisyphus import *

import i6_core.util as util

Path = setup_path(__package__)


class TrainBPEModelJob(Job):
    """
    Create a bpe codes file using the official subword-nmt repo, either installed from pip
    or https://github.com/rsennrich/subword-nmt

    This job is deprecated, to create BPE codes that are compatible with legacy (non-sisyphus) RETURNN setups
    using e.g. language models from Kazuki, please use the `ReturnnTrainBpeJob`.

    Otherwise, please consider using the `sentencepiece` implementation.
    """

    def __init__(
        self,
        text_corpus,
        symbols=1000,
        min_frequency=2,
        dict_input=False,
        total_symbols=False,
        subword_nmt_repo=None,
    ):
        """
        :param Path text_corpus:
        :param int symbols:
        :param int min_frequency:
        :param bool dict_input:
        :param bool total_symbols:
        :param Path|str|None subword_nmt_repo:
        """
        self.text_corpus = text_corpus
        self.symbols = symbols
        self.min_frequency = min_frequency
        self.dict_input = dict_input
        self.total_symbols = total_symbols

        self.subword_nmt_repo = (
            subword_nmt_repo if subword_nmt_repo is not None else gs.SUBWORD_NMT_PATH
        )

        self.out_code_file = self.output_path("codes")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 4}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        train_binary = os.path.join(
            tk.uncached_path(self.subword_nmt_repo), "subword_nmt/learn_bpe.py"
        )
        args = [
            sys.executable,
            train_binary,
            "-o",
            self.out_code_file.get_path(),
            "-s",
            str(self.symbols),
            "--min-frequency",
            str(self.min_frequency),
        ]
        if self.dict_input:
            args += ["--dict-input"]
        if self.total_symbols:
            args += ["--total-symbols"]

        text_corpus = self.text_corpus.get_path()

        with util.uopen(text_corpus, "rb") as f:
            p = sp.Popen(args, stdin=sp.PIPE, stdout=sys.stdout, stderr=sys.stderr)
            while True:
                data = f.read(4096)
                if len(data) > 0:
                    p.stdin.write(data)
                else:
                    break

            p.communicate()
            assert p.returncode == 0


class ReturnnTrainBpeJob(Job):
    """
    Create Bpe codes and vocab files compatible with RETURNN BytePairEncoding
    Repository:
        https://github.com/albertz/subword-nmt

    This job can be used to produce BPE codes compatible to legacy (non-sisyphus) RETURNN setups.
    """

    def __init__(self, text_file, bpe_size, unk_label="UNK", subword_nmt_repo=None):
        """
        :param Path text_file: corpus text file
        :param int bpe_size: number of BPE merge operations
        :param str unk_label: unknown label
        :param Path|str|None subword_nmt_repo: subword nmt repository path. see also `CloneGitRepositoryJob`
        """
        self.text_file = text_file
        self.bpe_size = bpe_size
        self.subword_nmt_repo = (
            subword_nmt_repo if subword_nmt_repo is not None else gs.SUBWORD_NMT_PATH
        )
        self.unk_label = unk_label

        self.out_bpe_codes = self.output_path("bpe.codes")
        self.out_bpe_vocab = self.output_path("bpe.vocab")
        self.out_vocab_size = self.output_var("vocab_size")

    def run(self):
        bpe_codes_cmd = [
            sys.executable,
            os.path.join(tk.uncached_path(self.subword_nmt_repo), "learn_bpe.py"),
            "--output",
            self.out_bpe_codes.get_path(),
            "--symbols",
            str(self.bpe_size),
        ]

        util.create_executable("create_bpe_codes.sh", bpe_codes_cmd)

        with util.uopen(self.text_file, "rb") as f:
            p = sp.Popen(
                bpe_codes_cmd, stdin=sp.PIPE, stdout=sys.stdout, stderr=sys.stderr
            )
            while True:
                data = f.read(4096)
                if len(data) > 0:
                    p.stdin.write(data)
                else:
                    break
            p.communicate()
            assert p.returncode == 0

        bpe_vocab_cmd = [
            sys.executable,
            os.path.join(tk.uncached_path(self.subword_nmt_repo), "create-py-vocab.py"),
            "--txt",
            self.text_file.get_path(),
            "--bpe",
            self.out_bpe_codes.get_path(),
            "--unk",
            self.unk_label,
            "--out",
            self.out_bpe_vocab.get_path(),
        ]

        util.create_executable("create_bpe_vocab.sh", bpe_vocab_cmd)
        sp.run(bpe_vocab_cmd, check=True)

        with util.uopen(self.out_bpe_vocab) as f:
            num_labels = max(list(eval(f.read()).values())) + 1  # 0-based index
            self.out_vocab_size.set(num_labels)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 4})
