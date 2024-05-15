"""
This is an old location of bpe jobs kept for backwards compatibility, for new setups using the subword-nmt based BPE,
please use i6_core.label.bpe, for other setups please switch to the sentencepiece implementation
"""
__all__ = ["TrainBPEModelJob", "ReturnnTrainBpeJob"]

from i6_core.text.label.subword_nmt.train import TrainBPEModelJob as _TrainBPEModelJob
from i6_core.text.label.subword_nmt.train import (
    ReturnnTrainBpeJob as _ReturnnTrainBpeJob,
)


class TrainBPEModelJob(_TrainBPEModelJob):
    """
    Create a bpe codes file using the official subword-nmt repo, either installed from pip
    or https://github.com/rsennrich/subword-nmt
    """

    pass


class ReturnnTrainBpeJob(_ReturnnTrainBpeJob):
    """
    Create Bpe codes and vocab files compatible with RETURNN BytePairEncoding
    Repository:
        https://github.com/albertz/subword-nmt
    """

    pass
