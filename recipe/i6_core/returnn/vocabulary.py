import pickle
from typing import Set, Optional, Union

from sisyphus import Job, Task, tk

from i6_core.lib import lexicon
from i6_core.util import uopen


class ReturnnVocabFromPhonemeInventory(Job):
    """
    Create a pickled vocab for RETURNN based on the phoneme-inventory of a bliss lexicon

    The lexicon.phonemes are an ordered dict, so the index table should always be stable.

    This job can be used in the case that the lexicon is "applied" on a corpus, meaning
    that the Bliss corpus XML contains the phoneme representation instead of the original orthography.

    Outputs:

    tk.Path out_vocab: path to the pickled returnn vocabulary (``vocab.pkl``)
    tk.Variable out_vocab_size: integer variable containing the vocabulary size (``vocab_size``)
    """

    __sis_hash_exclude__ = {"blacklist": None}

    def __init__(
        self,
        bliss_lexicon: tk.Path,
        blacklist: Optional[Union[Set, tk.Path]] = None,
    ):
        """
        :param bliss_lexicon: a bliss lexicon xml file containg a phoneme inventory
        :param blacklist: Exclude phonemes in blacklist from vocab, one phoneme per line if as file.
        """
        self.bliss_lexicon = bliss_lexicon
        self.blacklist = blacklist

        self.out_vocab = self.output_path("vocab.pkl")
        self.out_vocab_size = self.output_var("vocab_size")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())
        if isinstance(self.blacklist, tk.Path):
            blacklist = uopen(self.blacklist.get_path()).readlines()
            blacklist = {phoneme.strip() for phoneme in blacklist}
        elif isinstance(self.blacklist, Set):
            blacklist = self.blacklist
        else:
            blacklist = set()
        for k in blacklist:
            if k in lex.phonemes.keys():
                del lex.phonemes[k]
        vocab = {k: v for v, k in enumerate(lex.phonemes.keys())}
        pickle.dump(vocab, uopen(self.out_vocab, "wb"))

        self.out_vocab_size.set(len(vocab))
