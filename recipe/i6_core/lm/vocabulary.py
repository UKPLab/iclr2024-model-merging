from sisyphus import Job, Task

from i6_core.lib.lm import Lm


class VocabularyFromLmJob(Job):
    """
    Extract the vocabulary from an existing LM. Currently supports only arpa files for input.
    """

    def __init__(self, lm_file):
        """
        :param Path lm_file: path to the lm arpa file
        """
        self.lm_path = lm_file
        self.out_vocabulary = self.output_path("vocabulary.txt")
        self.out_vocabulary_size = self.output_var("vocabulary_size")

        self.rqmt = {"cpu": 1, "mem": 2, "time": 2}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        lm = Lm(self.lm_path.get())
        self.out_vocabulary_size.set(lm.ngram_counts[0])

        vocabulary = set()

        for n in range(len(lm.ngram_counts)):
            for words, _ in lm.get_ngrams(n + 1):
                for word in words.split(" "):
                    vocabulary.add(word)

        with open(self.out_vocabulary.get_path(), "w") as fout:
            for word in sorted(vocabulary):
                fout.write(f"{word}\n")
