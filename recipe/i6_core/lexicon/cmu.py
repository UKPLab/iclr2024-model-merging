import gzip
import itertools as it
import re
import shutil
import urllib.request

from sisyphus import *

Path = setup_path(__package__)


class DownloadCMUDictJob(Job):
    def __init__(
        self,
        url="https://raw.githubusercontent.com/cmusphinx/cmudict/master/",
        dict_file="cmudict.dict",
        phoneme_file="cmudict.phones",
    ):
        self.set_vis_name("Download CMU Lexicon")

        self.url = url
        self.dict_file = dict_file
        self.phoneme_file = phoneme_file

        self.out_phoneme_list = self.output_path(phoneme_file)
        self.out_cmu_lexicon = self.output_path(dict_file, cached=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        for name, path in zip(
            [self.dict_file, self.phoneme_file],
            [self.out_cmu_lexicon, self.out_phoneme_list],
        ):
            with urllib.request.urlopen(self.url + name) as src:
                with open(path.get_path(), "wb") as dst:
                    shutil.copyfileobj(src, dst)


class CMUDictToBlissJob(Job):
    def __init__(
        self,
        phoneme_list,
        cmu_lexicon,
        add_unknown=False,
        capitalize_words=True,
        noise_lemmas=None,
    ):
        self.set_vis_name("Convert CMU Lexicon to Bliss")

        self.phoneme_list = phoneme_list
        self.cmu_lexicon = cmu_lexicon
        self.add_unknown = add_unknown
        self.capitalize_words = capitalize_words
        self.noise_lemmas = noise_lemmas

        self.out_bliss_lexicon = self.output_path("lexicon.gz", cached=True)

    def tasks(self):
        yield Task(
            "run", mini_task=True, rqmt={"time": 10 / 60, "cpu": 1, "mem": "64M"}
        )

    def run(self):
        with open(
            tk.uncached_path(self.phoneme_list), "r", encoding="iso-8859-1"
        ) as input_phonemes:
            with open(
                tk.uncached_path(self.cmu_lexicon), "r", encoding="iso-8859-1"
            ) as input_lexicon:
                with gzip.open(self.out_bliss_lexicon.get_path(), "wt") as out:
                    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    out.write("<lexicon>\n")

                    # Phoneme Inventory
                    out.write("  <phoneme-inventory>\n")
                    out.write(
                        "    <phoneme><symbol>sil</symbol><variation>none</variation></phoneme>\n"
                    )
                    for line in input_phonemes:
                        out.write(
                            "    <phoneme><symbol>%s</symbol><variation>context</variation></phoneme>\n"
                            % line.split("\t")[0].lower()
                        )
                    if self.noise_lemmas is not None:
                        out.write(
                            "    <phoneme><symbol>noise</symbol><variation>none</variation></phoneme>\n"
                        )
                    out.write("  </phoneme-inventory>\n")

                    def write_lemma(
                        word, pronounciations, special="", synts=None, empty_eval=False
                    ):
                        synts = [] if synts is None else synts
                        if type(word) not in [list, tuple]:
                            word = [word]
                        out.write("  <lemma%s>\n" % special)
                        for w in word:
                            w = (
                                w.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                            )
                            out.write("    <orth>%s</orth>\n" % w)
                        for pron in sorted(pronounciations):
                            out.write("    <phon>%s</phon>\n" % pron)
                        for synt in synts:
                            out.write("    <synt>%s</synt>\n" % synt)
                        if empty_eval:
                            out.write("    <eval/>\n")
                        out.write("  </lemma>\n")

                    # special lemmas
                    write_lemma(
                        ("[SILENCE]", ""), {"sil"}, ' special="silence"', [""], True
                    )
                    write_lemma(
                        "[SENTANCE_BEGIN]",
                        set(),
                        ' special="sentence-begin"',
                        ["<tok>&lt;s&gt;</tok>"],
                    )
                    write_lemma(
                        "[SENTANCE_END]",
                        set(),
                        ' special="sentence-end"',
                        ["<tok>&lt;/s&gt;</tok>"],
                    )
                    if self.add_unknown:
                        write_lemma("[UNKNOWN]", set(), ' special="unknown"', [])
                    if self.noise_lemmas is not None:
                        write_lemma(
                            self.noise_lemmas, {"noise"}, synts=[""], empty_eval=True
                        )
                    last_word = None
                    pronounciations = None

                    # Pronounciations
                    for line in input_lexicon:
                        if line[0] == ";":
                            continue

                        s = list(
                            it.takewhile(lambda t: t != "#", line.strip().split(" "))
                        )
                        word = re.sub("^(.*)\\(\\d+\\)$", "\\1", s[0])
                        if self.capitalize_words:
                            word = word.upper()
                        pronounciation = " ".join(
                            [
                                (p.lower()[:-1] if p[-1] in "012" else p.lower())
                                for p in s[1:]
                                if len(p) > 0
                            ]
                        )

                        if word == last_word:
                            pronounciations.add(pronounciation)
                        else:
                            if last_word is not None:
                                write_lemma(last_word, pronounciations)
                            last_word = word
                            pronounciations = {pronounciation}

                    write_lemma(word, pronounciations)

                    out.write("</lexicon>")
                    out.close()
