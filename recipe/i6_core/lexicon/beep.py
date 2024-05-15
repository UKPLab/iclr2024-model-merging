import gzip
import io
import os
import shutil
import tempfile
import urllib.request

from sisyphus import *

Path = setup_path(__package__)


class DownloadBeepJob(Job):
    def __init__(
        self, url="ftp://svr-ftp.eng.cam.ac.uk/pub/comp.speech/dictionaries/beep.tar.gz"
    ):
        self.set_vis_name("Download Beep Lexicon")

        self.url = url

        self.out_archive = self.output_path("beep.tar.gz")
        self.out_phoneme_list = self.output_path("phone45.tab")
        self.out_beep_lexicon = self.output_path("beep-1.0")

        self.set_rqmt("run", {"time": 5 / 60, "cpu": 1, "mem": "64M"})

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with urllib.request.urlopen(self.url) as src:
            with open(self.out_archive.get_path(), "wb") as dst:
                shutil.copyfileobj(src, dst)
        with tempfile.TemporaryDirectory() as temp:
            self.sh("tar xz -C %s -f {archive}" % temp)
            shutil.copyfile(
                os.path.join(temp, "beep", "phone45.tab"),
                self.out_phoneme_list.get_path(),
            )
            shutil.copyfile(
                os.path.join(temp, "beep", "beep-1.0"), self.out_beep_lexicon.get_path()
            )


class BeepToBlissLexiconJob(Job):
    def __init__(
        self,
        phoneme_list,  # the phone45.tab file from the beep lexicon
        beep_lexicon,  # the pronounciation file from the beep lexicon
        add_unknown=False,
    ):
        self.set_vis_name("Convert Beep Lexicon to Bliss")

        self.phoneme_list = phoneme_list
        self.beep_lexicon = beep_lexicon
        self.add_unknown = add_unknown

        self.out_bliss_lexicon = self.output_path("lexicon.gz", cached=True)
        self.set_rqmt("run", {"time": 10 / 60, "cpu": 1, "mem": "64M"})

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with open(tk.uncached_path(self.phoneme_list), "r") as input_phonemes:
            with open(tk.uncached_path(self.beep_lexicon), "r") as input_lexicon:
                with gzip.open(self.out_bliss_lexicon.get_path(), "w") as out_binary:
                    out = io.TextIOWrapper(out_binary, encoding="utf-8")
                    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                    out.write("<lexicon>\n")

                    def write_lemma(word, pronounciations):
                        word = word.replace("&", "&amp;")
                        if word == "[SIL]":
                            return
                        special = ""
                        if (
                            word == "[PAUSE]"
                        ):  # pause appears first in the lexicon, but we want to name it silence
                            special = ' special="silence"'
                            word = "[SILENCE]"
                        out.write("  <lemma%s>\n" % special)
                        out.write("    <orth>%s</orth>\n" % word)
                        if word == "[SILENCE]":
                            out.write("    <orth></orth>\n")
                        for pron in pronounciations:
                            out.write("    <phon>%s</phon>\n" % pron)
                        if word == "[SILENCE]":
                            out.write("    <synt></synt>\n")
                            out.write("    <eval></eval>\n")
                        out.write("  </lemma>\n")

                    # Phoneme Inventory
                    out.write("  <phoneme-inventory>\n")
                    for line in input_phonemes:
                        line = line.strip()
                        variation = "none" if line == "sil" else "context"
                        out.write(
                            "    <phoneme><symbol>%s</symbol><variation>%s</variation></phoneme>\n"
                            % (line, variation)
                        )
                    out.write("  </phoneme-inventory>\n")

                    if self.add_unknown:
                        out.write('  <lemma special="unknown">\n')
                        out.write("    <orth>[UNKNOWN]</orth>\n")
                        out.write("  </lemma>\n")

                    out.write('  <lemma special="sentence-begin">\n')
                    out.write("    <orth>[SENTENCE_BEGIN]</orth>\n")
                    out.write("    <synt><tok>&lt;s&gt;</tok></synt>\n")
                    out.write("  </lemma>\n")

                    out.write('  <lemma special="sentence-end">\n')
                    out.write("    <orth>[SENTENCE_END]</orth>\n")
                    out.write("    <synt><tok>&lt;/s&gt;</tok></synt>\n")
                    out.write("  </lemma>\n")

                    # Pronounciations
                    last_word = ""
                    pronounciations = set()
                    for line in input_lexicon:
                        if line[0] == "#":
                            continue

                        s = list(filter(lambda s: len(s) > 0, line.split()))
                        if len(s) < 2:
                            continue

                        s[0] = s[0].replace("<", "[").replace(">", "]")

                        if s[0] == last_word:
                            pronounciations.add(" ".join(s[1:]))
                        else:
                            if len(last_word) > 0:
                                write_lemma(last_word, pronounciations)
                            pronounciations = set()
                            pronounciations.add(" ".join(s[1:]))
                            last_word = s[0]
                    if len(last_word) > 0:
                        write_lemma(last_word, pronounciations)

                    out.write("</lexicon>")
                    out.close()
