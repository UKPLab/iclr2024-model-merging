"""
This is an old location of bpe jobs kept for backwards compatibility, for new setups using the subword-nmt based BPE,
please use i6_core.label.bpe, for other setups please switch to the sentencepiece implementation
"""
__all__ = ["ApplyBPEModelToLexiconJob", "ApplyBPEToTextJob"]


from i6_core.text.label.subword_nmt.apply import (
    ApplyBPEModelToLexiconJob as _ApplyBPEModelToLexiconJob,
)
from i6_core.text.label.subword_nmt.apply import ApplyBPEToTextJob as _ApplyBPEToTextJob


class ApplyBPEModelToLexiconJob(_ApplyBPEModelToLexiconJob):
    """
    Apply BPE codes to a Bliss lexicon file
    """

    pass


class ApplyBPEToTextJob(_ApplyBPEToTextJob):
    """
    Apply BPE codes on a text file
    """

    pass
