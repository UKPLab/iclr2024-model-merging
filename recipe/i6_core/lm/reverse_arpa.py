__all__ = ["ReverseARPALmJob"]

import sys
import typing

from sisyphus import *

from i6_core.lib.lm import Lm
import i6_core.util as util

Path = setup_path(__package__)


class ReverseARPALmJob(Job):
    """
    Create a new LM in arpa format by reverting the n-grams of an existing Arpa LM.
    """

    def __init__(self, lm_path):
        """
        :param Path lm_path: Path to the existing arpa file
        """
        self.lm_path = lm_path

        self.out_reverse_lm = self.output_path("reverse.lm.gz", cached=True)

        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        # Copyright 2012 Mirko Hannemann BUT, mirko.hannemann@gmail.com
        # Copied from Kaldi: http://sourceforge.net/p/kaldi/code/1640/tree//trunk/egs/wsj/s5/utils/reverse_arpa.py
        # Adapted for python 3
        lm = Lm(self.lm_path.get())

        with util.uopen(
            self.out_reverse_lm.get_path(), "wt", encoding="utf-8"
        ) as outfile:

            # compute new reversed ARPA model
            all_ngrams = []
            for n in range(1, len(lm.ngram_counts) + 1):  # unigrams, bigrams, trigrams
                curr_ngrams = {name: prob for name, prob in lm.get_ngrams(n)}
                all_ngrams.append(curr_ngrams)
                for ngram in curr_ngrams:
                    words = ngram.split(" ")
                    ReverseARPALmJob.add_missing_backoffs(words, all_ngrams)

            outfile.write("\\data\\\n")
            for n in range(1, len(lm.ngram_counts) + 1):  # unigrams, bigrams, trigrams
                outfile.write("ngram %d=%d\n" % (n, len(all_ngrams[n - 1])))
            offset = 0.0
            for n in range(1, len(lm.ngram_counts) + 1):
                outfile.write("\\%d-grams:\n" % n)
                for ngram in sorted(all_ngrams[n - 1]):
                    prob = all_ngrams[n - 1][ngram]
                    # reverse word order
                    words = ngram.split()
                    rstr = " ".join(reversed(words))
                    # swap <s> and </s>
                    rev_ngram = (
                        rstr.replace("<s>", "<temp>")
                        .replace("</s>", "<s>")
                        .replace("<temp>", "</s>")
                    )

                    revprob = prob[0]
                    if prob[1] != float(
                        "inf"
                    ):  # only backoff weights from not newly created ngrams
                        revprob = revprob + prob[1]
                    # sum all missing terms in decreasing ngram order
                    for x in range(n - 1, 0, -1):
                        l_ngram = " ".join(words[:x])  # shortened ngram
                        if l_ngram not in all_ngrams[x - 1]:
                            sys.stderr.write(
                                "%s: not found %s\n" % (rev_ngram, l_ngram)
                            )
                        p_l = all_ngrams[x - 1][l_ngram][0]
                        revprob = revprob + p_l

                        r_ngram = " ".join(
                            words[1 : 1 + x]
                        )  # shortened ngram with offset one
                        if r_ngram not in all_ngrams[x - 1]:
                            sys.stderr.write(
                                "%s: not found %s\n" % (rev_ngram, r_ngram)
                            )
                        p_r = all_ngrams[x - 1][r_ngram][0]
                        revprob = revprob - p_r

                    if n != len(lm.ngram_counts):  # not highest order
                        back = 0.0
                        if (
                            rev_ngram[:3] == "<s>"
                        ):  # special handling since arpa2fst ignores <s> weight
                            if n == 1:
                                offset = revprob  # remember <s> weight
                                revprob = (
                                    lm.sentprob
                                )  # apply <s> weight from forward model
                                back = offset
                            elif n == 2:
                                revprob = (
                                    revprob + offset
                                )  # add <s> weight to bigrams starting with <s>
                        if prob[1] != float(
                            "inf"
                        ):  # only backoff weights from not newly created ngrams
                            outfile.write("%g %s %g\n" % (revprob, rev_ngram, back))
                        else:
                            outfile.write("%g %s -100000.0\n" % (revprob, rev_ngram))
                    else:  # highest order - no backoff weights
                        if (n == 2) and (rev_ngram[:3] == "<s>"):
                            revprob = revprob + offset
                        outfile.write("%g %s\n" % (revprob, rev_ngram))
            outfile.write("\\end\\\n")

    @staticmethod
    def add_missing_backoffs(
        words, ngrams: typing.List[typing.Dict[str, typing.Tuple]]
    ):
        n = len(ngrams)
        inf = float("inf")
        for x in range(n - 1, 0, -1):
            # add all missing backoff ngrams for reversed lm
            l_ngram = " ".join(words[:x])  # shortened ngram
            r_ngram = " ".join(words[1 : 1 + x])  # shortened ngram with offset one
            if l_ngram not in ngrams[x - 1]:  # create missing ngram
                ngrams[x - 1][l_ngram] = (0.0, inf)
            if r_ngram not in ngrams[x - 1]:  # create missing ngram
                ngrams[x - 1][r_ngram] = (0.0, inf)

            # add all missing backoff ngrams for forward lm
            h_ngram = " ".join(words[n - x :])  # shortened history
            if h_ngram not in ngrams[x - 1]:  # create missing ngram
                ngrams[x - 1][h_ngram] = (0.0, inf)
