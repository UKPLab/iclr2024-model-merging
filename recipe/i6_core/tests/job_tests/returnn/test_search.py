import os
import tempfile
from sisyphus import setup_path

from i6_core.returnn.search import SearchBPEtoWordsJob

Path = setup_path(__package__)


def test_search_bpe_to_words_single():
    with tempfile.TemporaryDirectory() as tmpdir:
        search_out = Path("files/search_out_single")
        reference_word_search_results = Path("files/word_search_results_single.py")
        bpe_to_words_job = SearchBPEtoWordsJob(search_out)
        bpe_to_words_job.out_word_search_results = Path(
            os.path.join(tmpdir, "word_search_results_single.py")
        )
        bpe_to_words_job.run()

        reference_dict = eval(
            open(reference_word_search_results.get_path(), "rt").read()
        )
        job_dict = eval(
            open(bpe_to_words_job.out_word_search_results.get_path(), "rt").read()
        )
        assert reference_dict == job_dict


def test_search_bpe_to_words_nbest():
    with tempfile.TemporaryDirectory() as tmpdir:
        search_out = Path("files/search_out_nbest")
        reference_word_search_results = Path("files/word_search_results_nbest.py")
        bpe_to_words_job = SearchBPEtoWordsJob(search_out)
        bpe_to_words_job.out_word_search_results = Path(
            os.path.join(tmpdir, "word_search_results_nbest.py")
        )
        bpe_to_words_job.run()

        reference_dict = eval(
            open(reference_word_search_results.get_path(), "rt").read()
        )
        job_dict = eval(
            open(bpe_to_words_job.out_word_search_results.get_path(), "rt").read()
        )
        assert reference_dict == job_dict
