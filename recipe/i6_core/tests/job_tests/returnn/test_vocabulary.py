import os
import tempfile
import pickle as pkl
import filecmp
from sisyphus import setup_path, tk

from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

Path = setup_path(__package__)


def test_returnn_vocab_from_phoneme_inventory():

    with tempfile.TemporaryDirectory() as tmpdir:
        lexicon = Path("files/test_lexicon.xml.gz")

        vocab_job = ReturnnVocabFromPhonemeInventory(bliss_lexicon=lexicon)
        vocab_job.out_vocab = Path(os.path.join(tmpdir, "vocab.pkl"))
        vocab_job.out_vocab_size = tk.Variable(os.path.join(tmpdir, "vocab_size"))
        vocab_job.run()

        vocab_path = Path("files/vocab.pkl")
        with open(vocab_path.get_path(), "rb") as f:
            reference_vocab = pkl.load(f)
        with open(os.path.join(tmpdir, "vocab.pkl"), "rb") as f:
            vocab = pkl.load(f)
        assert reference_vocab == vocab
        assert vocab_job.out_vocab_size.get() == 44


def test_returnn_vocab_from_phoneme_inventory_blacklist():

    with tempfile.TemporaryDirectory() as tmpdir:
        lexicon = Path("files/test_lexicon.xml.gz")

        vocab_job = ReturnnVocabFromPhonemeInventory(
            bliss_lexicon=lexicon, blacklist={"[SILENCE]"}
        )
        vocab_job.out_vocab = Path(os.path.join(tmpdir, "vocab.pkl"))
        vocab_job.out_vocab_size = tk.Variable(os.path.join(tmpdir, "vocab_size"))
        vocab_job.run()

        vocab_path = Path("files/blacklist_vocab.pkl")
        with open(vocab_path.get_path(), "rb") as f:
            reference_vocab = pkl.load(f)
        with open(os.path.join(tmpdir, "vocab.pkl"), "rb") as f:
            vocab = pkl.load(f)
        assert reference_vocab == vocab
        assert vocab_job.out_vocab_size.get() == 43


def test_returnn_vocab_from_phoneme_inventory_blacklist_file():

    with tempfile.TemporaryDirectory() as tmpdir:
        lexicon = Path("files/test_lexicon.xml.gz")
        blacklist = Path("files/blacklist")
        vocab_job = ReturnnVocabFromPhonemeInventory(
            bliss_lexicon=lexicon, blacklist=blacklist
        )
        vocab_job.out_vocab = Path(os.path.join(tmpdir, "vocab.pkl"))
        vocab_job.out_vocab_size = tk.Variable(os.path.join(tmpdir, "vocab_size"))
        vocab_job.run()

        vocab_path = Path("files/blacklist_file_vocab.pkl")
        with open(vocab_path.get_path(), "rb") as f:
            reference_vocab = pkl.load(f)
        with open(os.path.join(tmpdir, "vocab.pkl"), "rb") as f:
            vocab = pkl.load(f)
        assert reference_vocab == vocab
        assert vocab_job.out_vocab_size.get() == 42
