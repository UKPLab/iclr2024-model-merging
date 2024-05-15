__all__ = ["WriteLexiconJob", "MergeLexiconJob"]

import copy
from collections import OrderedDict, defaultdict
import itertools

from sisyphus import Job, Task

from i6_core.lib import lexicon
from i6_core.util import write_xml


class WriteLexiconJob(Job):
    """
    Create a bliss lexicon file from a static Lexicon.

    Supports optional sorting of phonemes and lemmata.

    Example for a static lexicon:

    .. code: python

        static_lexicon = lexicon.Lexicon()
        static_lexicon.add_lemma(
            static_lexiconicon.Lemma(
                orth=["[SILENCE]", ""],
                phon=["[SILENCE]"],
                synt=[],
                special="silence",
                eval=[[]],
            )
        )
        # set synt and eval carefully
        # synt == None   --> nothing                 no synt element
        # synt == []     --> "<synt />"              meant to be empty synt token sequence
        # synt == [""]   --> "<synt><tok /></synt>"  incorrent
        # eval == []     --> nothing                 no eval element
        # eval == [[]]   --> "<eval />"              meant to be empty eval token sequence
        # eval == [""]   --> "<eval />"              equivalent to [[]], but not encouraged
        # eval == [[""]] --> "<eval><tok /></eval>"  incorrect
        static_lexicon.add_lemma(
            static_lexiconicon.Lemma(
                orth=["[UNKNOWN]"],
                phon=["[UNKNOWN]"],
                synt=["<UNK>"],
                special="unknown",
            )
        )
        static_lexicon.add_phoneme("[SILENCE]", variation="none")
        static_lexicon.add_phoneme("[UNKNOWN]", variation="none")
    """

    def __init__(
        self, static_lexicon, sort_phonemes=False, sort_lemmata=False, compressed=True
    ):
        """
        :param lexicon.Lexicon static_lexicon: A Lexicon object
        :param bool sort_phonemes: sort phoneme inventory alphabetically
        :param bool sort_lemmata: sort lemmata alphabetically based on first orth entry
        :param bool compressed: compress final lexicon
        """
        self.static_lexicon = static_lexicon
        self.sort_phonemes = sort_phonemes
        self.sort_lemmata = sort_lemmata

        self.out_bliss_lexicon = self.output_path(
            "lexicon.xml.gz" if compressed else "lexicon.xml"
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()
        if self.sort_phonemes:
            sorted_phoneme_list = [
                (k, self.static_lexicon.phonemes[k])
                for k in sorted(self.static_lexicon.phonemes.keys())
            ]
            for phoneme_tuple in sorted_phoneme_list:
                lex.add_phoneme(symbol=phoneme_tuple[0], variation=phoneme_tuple[1])
        else:
            lex.phonemes = self.static_lexicon.phonemes

        if self.sort_lemmata:
            lemma_dict = {}
            for lemma in self.static_lexicon.lemmata:
                # sort by first orth entry
                lemma_dict[lemma.orth[0]] = lemma
            lex.lemmata = [lemma_dict[key] for key in sorted(lemma_dict.keys())]
        else:
            lex.lemmata = self.static_lexicon.lemmata

        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())

    @classmethod
    def _fix_hash_for_lexicon(cls, new_lexicon):
        """
        The "old" lexicon had an incorrect "synt" type, after fixing
        the hashes for the lexicon changed, so this job here
        needs to revert the lexicon to the old "synt" type.

        :param lexicon.Lexicon new_lexicon:
        :return: lexicon in the legacy format
        :type: lexicon.Lexicon
        """
        lex = lexicon.Lexicon()
        lex.phonemes = new_lexicon.phonemes
        lex.lemmata = []
        for new_lemma in new_lexicon.lemmata:
            lemma = copy.deepcopy(new_lemma)
            lemma.synt = [new_lemma.synt] if new_lemma.synt is not None else []
            lex.lemmata.append(lemma)

        return lex

    @classmethod
    def hash(cls, parsed_args):
        parsed_args = parsed_args.copy()
        parsed_args["static_lexicon"] = cls._fix_hash_for_lexicon(
            parsed_args["static_lexicon"]
        )
        return super().hash(parsed_args)


class MergeLexiconJob(Job):
    """
    Merge multiple bliss lexica into a single bliss lexicon.

    Phonemes and lemmata can be individually sorted alphabetically or kept as is.

    When merging a lexicon with a static lexicon, putting the static lexicon first
    and only sorting the phonemes will result in the "typical" lexicon structure.

    Please be aware that the sorting or merging of lexica that were already used
    will create a new lexicon that might be incompatible to previously generated alignments.
    """

    def __init__(
        self, bliss_lexica, sort_phonemes=False, sort_lemmata=False, compressed=True
    ):
        """
        :param list[Path] bliss_lexica: list of bliss lexicon files (plain or gz)
        :param bool sort_phonemes: sort phoneme inventory alphabetically
        :param bool sort_lemmata: sort lemmata alphabetically based on first orth entry
        :param bool compressed: compress final lexicon
        """
        self.lexica = bliss_lexica
        self.sort_phonemes = sort_phonemes
        self.sort_lemmata = sort_lemmata

        self.out_bliss_lexicon = self.output_path(
            "lexicon.xml.gz" if compressed else "lexicon.xml"
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        merged_lex = lexicon.Lexicon()

        lexica = []
        for lexicon_path in self.lexica:
            lex = lexicon.Lexicon()
            lex.load(lexicon_path.get_path())
            lexica.append(lex)

        # combine the phonemes
        merged_phonemes = OrderedDict()
        for lex in lexica:
            for symbol, variation in lex.phonemes.items():
                if symbol in merged_phonemes.keys():
                    assert variation == merged_phonemes[symbol], (
                        "conflicting phoneme variant for phoneme: %s" % symbol
                    )
                else:
                    merged_phonemes[symbol] = variation

        if self.sort_phonemes:
            sorted_phoneme_list = [
                (k, merged_phonemes[k]) for k in sorted(merged_phonemes.keys())
            ]
            for phoneme_tuple in sorted_phoneme_list:
                merged_lex.add_phoneme(
                    symbol=phoneme_tuple[0], variation=phoneme_tuple[1]
                )
        else:
            merged_lex.phonemes = merged_phonemes

        # combine the lemmata
        if self.sort_lemmata:
            lemma_dict = defaultdict(list)
            for lex in lexica:
                for lemma in lex.lemmata:
                    # sort by first orth entry
                    orth_key = lemma.orth[0] if lemma.orth else ""
                    lemma_dict[orth_key].append(lemma)
            merged_lex.lemmata = list(
                itertools.chain(*[lemma_dict[key] for key in sorted(lemma_dict.keys())])
            )
        else:
            for lex in lexica:
                # check for existing orths to avoid overlap
                merged_lex.lemmata.extend(lex.lemmata)

        write_xml(self.out_bliss_lexicon.get_path(), merged_lex.to_xml())
