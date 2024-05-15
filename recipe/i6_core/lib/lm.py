import logging

from sisyphus import *

import i6_core.util as util


class Lm:
    """
    Interface to access the ngrams of an LM. Currently supports only LMs in arpa format.
    """

    def __init__(self, lm_path):
        """
        :param str lm_path: Path to the LM file, currently supports only arpa files
        """
        self.lm_path = lm_path
        self.ngram_counts = []
        self.ngrams_start = []
        self.ngrams_end = []
        self.sentprob = 0.0
        self.load_arpa()

    def load_arpa(self):
        # read language model in ARPA format
        lm_path = self.lm_path
        with util.uopen(lm_path, "rt", encoding="utf-8") as infile:
            reader = {
                "infile": infile,
                "lineno": 0,
            }

            def read_increase_line():
                reader["lineno"] += 1
                return reader["infile"].readline()

            text = read_increase_line()
            while text and text[:6] != "\\data\\":
                text = read_increase_line()
            assert text, "Invalid ARPA file"

            while text and text[:5] != "ngram":
                text = read_increase_line()

            # get ngram counts
            n = 0
            while text and text[:5] == "ngram":
                ngram_count = text.split("=")
                counts = int(ngram_count[1].strip())
                order = ngram_count[0].split()
                read_n = int(order[1].strip())
                assert read_n == n + 1, "invalid ARPA file: %s %d %d" % (
                    text.strip(),
                    read_n,
                    n + 1,
                )
                n = read_n
                self.ngram_counts.append(counts)
                text = read_increase_line()

            # read through the file and find start and end lines for each ngrams order
            for n in range(
                1, len(self.ngram_counts) + 1
            ):  # unigrams, bigrams, trigrams
                while text and "-grams:" not in text:
                    text = read_increase_line()
                assert n == int(text[1]), "invalid ARPA file: %s" % text

                self.ngrams_start.append(
                    (reader["lineno"] + 1, reader["infile"].tell())
                )
                for ng in range(self.ngram_counts[n - 1]):
                    text = read_increase_line()
                    if not_ngrams(text):
                        break
                self.ngrams_end.append(reader["lineno"])
                logging.info(f"Read through the {n}grams")

            while text and text[:5] != "\\end\\":
                text = read_increase_line()
            assert text, "invalid ARPA file"

        assert (
            len(self.ngram_counts) == len(self.ngrams_start) == len(self.ngrams_end)
        ), f"{len(self.ngram_counts)} == {len(self.ngrams_start)} == {len(self.ngrams_end)} is False"
        for i in range(len(self.ngram_counts)):
            assert self.ngram_counts[i] == (
                self.ngrams_end[i] - self.ngrams_start[i][0] + 1
            ), "Stated %d-gram count is wrong %d != %d" % (
                i + 1,
                self.ngram_counts[i],
                (self.ngrams_end[i] - self.ngrams_start[i][0] + 1),
            )

    def get_ngrams(self, n):
        """
        returns all the ngrams of order n
        """
        yield from self._read_ngrams(n)

    def _read_ngrams(self, n):
        """
        Read the ngrams knowing start and end lines
        """
        with util.uopen(self.lm_path, "rt", encoding="utf-8") as infile:
            infile.seek(self.ngrams_start[n - 1][1])
            i = self.ngrams_start[n - 1][0] - 1
            while i < self.ngrams_end[n - 1]:
                i += 1
                text = infile.readline()
                entry = text.split()
                prob = float(entry[0])
                if len(entry) > n + 1:
                    back = float(entry[-1])
                    words = entry[1 : n + 1]
                else:
                    back = 0.0
                    words = entry[1:]
                ngram = " ".join(words)
                if (n == 1) and words[0] == "<s>":
                    self.sentprob = prob
                    prob = 0.0
                if i - (self.ngrams_start[n - 1][0] - 1) % 1000 == 0:
                    logging.info(f"Read 1000 {n}grams")
                yield ngram, (prob, back)


def not_ngrams(text: str):
    return (not text) or (
        (len(text.split()) == 1) and (("-grams:" in text) or (text[:5] == "\\end\\"))
    )
