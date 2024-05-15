__all__ = ["BlissLexiconToG2PLexiconJob", "G2POutputToBlissLexiconJob"]

import itertools as it
import logging
import xml.etree.ElementTree as ET

from sisyphus import *

from i6_core.lib import lexicon
from i6_core.util import uopen, write_xml

Path = setup_path(__package__)


class BlissLexiconToG2PLexiconJob(Job):
    """
    Convert a bliss lexicon into a g2p compatible lexicon for training
    """

    __sis_hash_exclude__ = {
        "include_orthography_variants": False,
    }

    def __init__(
        self,
        bliss_lexicon,
        include_pronunciation_variants=False,
        include_orthography_variants=False,
    ):
        """
        :param Path bliss_lexicon:
        :param bool include_pronunciation_variants: In case of multiple phoneme representations for one lemma, when this is false it outputs only the first phoneme
        :param bool include_orthography_variants: In case of multiple orthographic representations for one lemma, when this is false it outputs only the first orth
        """
        self.bliss_lexicon = bliss_lexicon
        self.include_pronunciation_variants = include_pronunciation_variants
        self.include_orthography_variants = include_orthography_variants

        self.out_g2p_lexicon = self.output_path("g2p.lexicon")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.bliss_lexicon, "rt") as f:
            tree = ET.parse(f)
        with uopen(self.out_g2p_lexicon, "wt") as out:
            all_lemmas = tree.findall(".//lemma")
            assert (
                len(all_lemmas) > 0
            ), "No lemma tag found in the lexicon file! Wrong format file?"

            for lemma in all_lemmas:
                if lemma.get("special") is not None:
                    continue

                if self.include_orthography_variants:
                    orths = [o.text.strip() for o in lemma.findall("orth")]
                else:
                    orths = [lemma.find("orth").text.strip()]

                for orth in orths:
                    if self.include_pronunciation_variants:
                        phons = lemma.findall("phon")
                        phon_single = []
                        for phon in phons:
                            p = phon.text.strip()
                            if p not in phon_single:
                                phon_single.append(p)
                                out.write("%s %s\n" % (orth, p))
                    else:
                        phon = lemma.find("phon").text.strip()
                        out.write("%s %s\n" % (orth, phon))


class G2POutputToBlissLexiconJob(Job):
    """
    Convert a g2p applied word list file (g2p lexicon) into a bliss lexicon
    """

    def __init__(self, iv_bliss_lexicon, g2p_lexicon, merge=True):
        """
        :param Path iv_bliss_lexicon: bliss lexicon as reference for the phoneme inventory
        :param Path g2p_lexicon: from ApplyG2PModelJob.out_g2p_lexicon
        :param bool merge: merge the g2p lexicon into the iv_bliss_lexicon instead of
            only taking the phoneme inventory
        """
        self.iv_bliss_lexicon = iv_bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.merge = merge

        self.out_oov_lexicon = self.output_path("oov.lexicon.gz", cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.g2p_lexicon, "rt", encoding="utf-8") as f:
            oov_words = dict()
            for orth, data in it.groupby(
                map(lambda line: line.strip().split("\t"), f), lambda t: t[0]
            ):
                oov_words[orth] = []
                for d in data:
                    if len(d) == 4:
                        oov_words[orth].append(d[3])
                    elif len(d) < 4:
                        logging.warning(
                            'No pronunciation found for orthography "{}"'.format(orth)
                        )
                    else:
                        logging.warning(
                            'Did not fully parse entry for orthography "{}"'.format(
                                orth
                            )
                        )

        iv_lexicon = lexicon.Lexicon()
        iv_lexicon.load(self.iv_bliss_lexicon.get_path())

        g2p_lexicon = lexicon.Lexicon()
        # use phoneme inventory from existing lexicon
        g2p_lexicon.phonemes = iv_lexicon.phonemes

        if self.merge:
            # when we merge also copy over all existing lemmata
            g2p_lexicon.lemmata = iv_lexicon.lemmata

        for orth, prons in oov_words.items():
            if len(prons) > 0:
                lemma = lexicon.Lemma(orth=[orth], phon=prons)
                g2p_lexicon.add_lemma(lemma)

        write_xml(self.out_oov_lexicon.get_path(), g2p_lexicon.to_xml())
