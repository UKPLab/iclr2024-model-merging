from enum import Enum
import logging
import shutil
import subprocess

from sisyphus import *

try:
    import sentencepiece
except ImportError:
    if (
        not hasattr(gs, "WARNING_NO_SENTENCEPIECE")
        or gs.WARNING_NO_SENTENCEPIECE is True
    ):
        logging.warning(
            "The package 'sentencepiece' is not installed in the manager python env. Please make sure it is installed "
            "in the python environment running the Sisyphus worker. To suppress this warning set "
            "'WARNING_NO_SENTENCEPIECE=False' in the settings.py"
        )


class SentencePieceType(Enum):
    UNIGRAM = "unigram"
    BPE = "bpe"
    CHAR = "char"
    WORD = "word"


class TrainSentencePieceJob(Job):
    """
    Train a sentence-piece model to be used with RETURNN

    See also `https://returnn.readthedocs.io/en/latest/api/datasets.util.vocabulary.html#returnn.datasets.util.vocabulary.SentencePieces`_
    """

    def __init__(
        self,
        training_text,
        vocab_size,
        model_type,
        character_coverage=1.0,
        additional_options=None,
    ):
        """

        :param tk.Path training_text: raw text or gzipped text
        :param int vocab_size: target vocabulary size for the created model
        :param SentencePieceType model_type: which sentence model to use, use "UNIGRAM" for "typical" SPM
        :param float character_coverage: official default is 0.9995, but this caused the least used character to be dropped entirely
        :param dict|None additional_options: additional trainer options, see `https://github.com/google/sentencepiece/blob/master/doc/options.md`_
        """

        self.training_text = training_text
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.additional_options = additional_options or {}

        self.out_model = self.output_path("spm_out.model")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sentencepiece

        training_text_path = self.training_text.get_path()
        if training_text_path.endswith(".gz"):
            local_training_text_path = "unzipped_training_text.txt"
            outfile = open(local_training_text_path, "wt")
            subprocess.check_call(["gzip", "-dc", training_text_path], stdout=outfile)
            training_text_path = local_training_text_path

        sentencepiece.SentencePieceTrainer.Train(
            input=training_text_path,
            model_prefix="spm_out",
            model_type=self.model_type.value,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            **self.additional_options,
        )

        shutil.move("spm_out.model", self.out_model.get_path())
