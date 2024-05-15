import collections
import gzip
import os.path
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from sisyphus import *

Path = setup_path(__package__)

import i6_core.lib.lexicon as lexicon
from i6_core.util import uopen, write_xml


class LexiconToWordListJob(Job):
    def __init__(self, bliss_lexicon, apply_filter=True):
        self.set_vis_name("Lexicon to Word List")

        self.bliss_lexicon = bliss_lexicon
        self.apply_filter = apply_filter

        self.out_word_list = self.output_path("words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(tk.uncached_path(self.bliss_lexicon), "r") as lexicon_file:
            lexicon = ET.fromstring(lexicon_file.read())
            words = set()
            for e in lexicon.findall("./lemma/orth"):
                if (
                    e.text is not None
                    and len(e.text) > 0
                    and not (e.text.startswith("[") and self.apply_filter)
                ):
                    words.add(e.text)

        with open(self.out_word_list.get_path(), "w") as word_file:
            for w in sorted(words):
                word_file.write("%s\n" % w)


class FilterLexiconByWordListJob(Job):
    """
    Filter lemmata to given word list.
    Warning: case_sensitive parameter does the opposite. Kept for backwards-compatibility.
    """

    __sis_hash_exclude__ = {"check_synt_tok": False}

    def __init__(
        self, bliss_lexicon, word_list, case_sensitive=False, check_synt_tok=False
    ):
        """
        :param tk.Path bliss_lexicon: lexicon file to be handeled
        :param tk.Path word_list: filter lexicon by this word list
        :param bool case_sensitive: filter lemmata case-sensitive. Warning: parameter does the opposite.
        :param bool check_synt_tok: keep also lemmata where the syntactic token matches word_list
        """
        self.set_vis_name("Filter Lexicon by Word List")

        self.bliss_lexicon = bliss_lexicon
        self.word_list = word_list
        self.case_sensitive = case_sensitive
        self.check_synt_tok = check_synt_tok

        self.out_bliss_lexicon = self.output_path(
            os.path.basename(tk.uncached_path(bliss_lexicon)), cached=True
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        transform = (lambda s: s.lower()) if self.case_sensitive else (lambda s: s)

        with uopen(tk.uncached_path(self.bliss_lexicon), "r") as lexicon_file:
            old_lexicon = ET.fromstring(lexicon_file.read())

        with uopen(tk.uncached_path(self.word_list), "r") as words_file:
            words = set([transform(w.strip()) for w in words_file.readlines()])

        root = ET.Element("lexicon")
        root.append(old_lexicon.find("phoneme-inventory"))
        for lemma in old_lexicon.findall("lemma"):
            if any(
                transform(orth.text) in words
                or "special" in lemma.attrib
                or (orth.text is not None and orth.text.startswith("["))
                for orth in lemma.findall("orth")
            ) or (
                self.check_synt_tok
                and all(
                    [
                        transform(tok.text) in words
                        for tok in lemma.findall("./synt/tok")
                    ]
                )
            ):
                root.append(lemma)

        with uopen(self.out_bliss_lexicon.get_path(), "wt") as lexicon_file:
            lexicon_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lexicon_file.write(ET.tostring(root, "unicode"))


class LexiconUniqueOrthJob(Job):
    """Merge lemmata with the same orthography."""

    __sis_hash_exclude__ = {"merge_multi_orths_lemmata": False}

    def __init__(self, bliss_lexicon, merge_multi_orths_lemmata=False):
        """
        :param tk.Path bliss_lexicon: lexicon file to be handeled
        :param bool merge_multi_orths_lemmata: if True, also lemmata containing
            multiple orths are merged based on their primary orth. Otherwise
            they are ignored.

            Merging strategy
            - orth/phon/eval
                all orth/phon/eval elements are merged together
            - synt
                synt element is only copied to target lemma when
                    1) the target lemma does not already have one
                    2) and the rest to-be-merged-lemmata have any synt
                       element.
                    ** having a synt <=> synt is not None
                this could lead to INFORMATION LOSS if there are several
                different synt token sequences in the to-be-merged lemmata
        """
        self.set_vis_name("Make Lexicon Orths Unique")

        self.bliss_lexicon = bliss_lexicon
        self.merge_multi_orths_lemmata = merge_multi_orths_lemmata

        self.out_bliss_lexicon = self.output_path(
            os.path.basename(tk.uncached_path(bliss_lexicon)), cached=True
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()
        lex.load(self.bliss_lexicon.get_path())

        orth2lemmata = collections.defaultdict(list)

        for lemma in lex.lemmata:
            if lemma.special:
                continue
            num_orths = len(lemma.orth)
            if num_orths < 1:
                continue
            if num_orths > 1 and not self.merge_multi_orths_lemmata:
                continue
            orth2lemmata[lemma.orth[0]].append(lemma)

        for orth, lemmata in orth2lemmata.items():
            if len(lemmata) < 2:
                continue
            final_lemma = lemmata[0]
            for lemma in lemmata[1:]:
                for orth in lemma.orth:
                    if orth not in final_lemma.orth:
                        final_lemma.orth.append(orth)
                for phon in lemma.phon:
                    if phon not in final_lemma.phon:
                        final_lemma.phon.append(phon)
                if final_lemma.synt is None and lemma.synt is not None:
                    final_lemma.synt = lemma.synt
                for eval in lemma.eval:
                    if eval not in final_lemma.eval:
                        final_lemma.eval.append(eval)
                lex.lemmata.remove(lemma)

        write_xml(self.out_bliss_lexicon, element_tree=lex.to_xml())


class LexiconFromTextFileJob(Job):
    """
    Create a bliss lexicon from a regular text file, where each line contains:
    <WORD> <PHONEME1> <PHONEME2> ...
    separated by tabs or spaces.
    The lemmata will be added in the order they appear in the text file,
    the phonemes will be sorted alphabetically.
    Phoneme variants of the same word need to appear next to each other.

    WARNING: No special lemmas or phonemes are added,
    so do not use this lexicon with RASR directly!

    As the splitting is taken from RASR and not fully tested,
    it might not work in all cases so do not use this job
    without checking the output manually on new lexica.
    """

    def __init__(self, text_file, compressed=True):
        """
        :param Path text_file:
        :param compressed: save as .xml.gz
        """
        self.text_file = text_file

        self.out_bliss_lexicon = self.output_path(
            "lexicon.xml.gz" if compressed else "lexicon.xml"
        )

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lex = lexicon.Lexicon()

        phonemes = set()
        seen_lemma = {}
        with uopen(self.text_file.get_path()) as f:
            for line in f:
                # splitting is taken from RASR
                # src/Tools/Bliss/blissLexiconLib.py#L185
                s = line.split(None, 1)
                orth = s[0].split("\\", 1)[0]
                phon_variants = [
                    tuple(p.split()) for p in s[1].split("\\") if p.strip()
                ]
                for phon_variant in phon_variants:
                    phonemes.update(phon_variant)
                phon = [" ".join(v) for v in phon_variants]
                if orth in seen_lemma:
                    lemma = seen_lemma[orth]
                    for p in phon:
                        if p not in lemma.phon:
                            lemma.phon.append(p)
                else:
                    lemma = lexicon.Lemma(orth=[orth], phon=phon)
                    seen_lemma[orth] = lemma
                    lex.add_lemma(lemma)

        for phoneme in sorted(phonemes):
            lex.add_phoneme(phoneme)

        write_xml(self.out_bliss_lexicon.get_path(), lex.to_xml())


class GraphemicLexiconFromWordListJob(Job):
    default_transforms = {".": "DOT", "+": "PLUS", "{": "LBR", "}": "RBR"}

    def __init__(
        self, word_list_file, add_unknown=False, add_noise=False, transforms=None
    ):
        self.add_unknown = add_unknown
        self.add_noise = add_noise
        self.transforms = (
            transforms if transforms is not None else self.default_transforms
        )
        self.word_list_file = word_list_file

        self.out_bliss_lexicon = self.output_path("grapheme.lexicon", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(tk.uncached_path(self.word_list_file), "rt") as f:
            words = [l.strip() for l in f]

        phonemes = set()
        for w in words:
            phonemes.update(w)
        phonemes.discard(" ")  # just in case

        lex = lexicon.Lexicon()
        lex.add_phoneme("sil", variation="none")
        for p in sorted(phonemes):
            p = self.transforms.get(p, p)
            lex.add_phoneme(p, "context")
        if self.add_unknown:
            lex.add_phoneme("unk", "none")
        if self.add_noise:
            lex.add_phoneme("noise", "none")

        # TODO: figure out requirements on synt/eval element for differnt types of lemmata
        # silence lemma, needs synt/eval element with empty token sequence
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SILENCE]", ""],
                phon=["sil"],
                synt=[],
                special="silence",
                eval=[[]],
            )
        )
        # sentence border lemmata, needs no eval element
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SENTENCE_BEGIN]"], synt=["<s>"], special="sentence-begin"
            )
        )
        lex.add_lemma(
            lexicon.Lemma(
                orth=["[SENTENCE_END]"], synt=["</s>"], special="sentence-end"
            )
        )
        # unknown lemma, needs no synt/eval element
        if self.add_unknown:
            lex.add_lemma(
                lexicon.Lemma(orth=["[UNKNOWN]"], phon=["unk"], special="unknown")
            )
            # TODO: synt = ["<UNK>"] ???
        # noise lemma, needs empty synt token sequence but no eval element?
        if self.add_noise:
            lex.add_lemma(
                lexicon.Lemma(
                    orth=["[NOISE]"],
                    phon=["noise"],
                    synt=[],
                    special="unknown",
                )
            )

        for w in words:
            l = lexicon.Lemma()
            l.orth.append(w)
            l.phon.append(" " + " ".join(self.transforms.get(p, p) for p in w) + " ")
            lex.add_lemma(l)

        with uopen(self.out_bliss_lexicon.get_path(), "w") as lexicon_file:
            lexicon_file.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lexicon_file.write(ET.tostring(lex.to_xml(), "unicode"))
