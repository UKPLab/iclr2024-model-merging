import os
import tempfile
import filecmp
from sisyphus import setup_path

from i6_core.corpus.convert import CorpusReplaceOrthFromReferenceCorpus

Path = setup_path(__package__)


def test_corpus_replace_orth_from_reference_corpus():

    with tempfile.TemporaryDirectory() as tmpdir:
        reference_corpus = Path("files/test_replace.corpus.xml")
        bliss_corpus_corrupt = Path("files/test_replace.corrupt.corpus.xml")

        replace_job = CorpusReplaceOrthFromReferenceCorpus(
            bliss_corpus=bliss_corpus_corrupt, reference_bliss_corpus=reference_corpus
        )
        replace_job.out_corpus = Path(os.path.join(tmpdir, "replaced.corpus.xml"))
        replace_job.run()

        assert filecmp.cmp(replace_job.out_corpus, reference_corpus, shallow=False)
