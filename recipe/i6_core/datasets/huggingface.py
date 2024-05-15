"""
https://huggingface.co/docs/datasets/
"""

from typing import Optional, Any, Union
from sisyphus import *
from sisyphus.delayed_ops import DelayedBase

from i6_core.util import instanciate_delayed


class DownloadAndPrepareHuggingFaceDatasetJob(Job):
    """
    https://huggingface.co/docs/datasets/
    https://huggingface.co/datasets

    pip install datasets

    Basically wraps ``datasets.load_dataset(...).save_to_disk(out_dir)``.

    Example for Librispeech:

    DownloadAndPrepareHuggingFaceDatasetJob("librispeech_asr", "clean")
    https://github.com/huggingface/datasets/issues/4179
    """

    def __init__(
        self,
        path: Union[str, DelayedBase],
        name: Optional[str] = None,
        *,
        data_files: Optional[Any] = None,
        revision: Optional[str] = None,
        time_rqmt: float = 1,
        mem_rqmt: float = 2,
        cpu_rqmt: int = 2,
        mini_task: bool = True,
    ):
        """
        :param path: Path or name of the dataset, parameter passed to `Dataset.load_dataset`
        :param name: Name of the dataset configuration, parameter passed to `Dataset.load_dataset`
        :param data_files: Path(s) to the source data file(s), parameter passed to `Dataset.load_dataset`
        :param revision: Version of the dataset script, parameter passed to `Dataset.load_dataset`
        :param float time_rqmt:
        :param float mem_rqmt:
        :param int cpu_rqmt:
        :param bool mini_task: the job should be run as mini_task
        """
        super().__init__()
        self.path = path
        self.name = name
        self.data_files = data_files
        self.revision = revision

        self.rqmt = {"cpu": cpu_rqmt, "mem": mem_rqmt, "time": time_rqmt}
        self.mini_task = mini_task

        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=self.mini_task)

    def run(self):
        import tempfile
        import datasets

        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:

            ds = datasets.load_dataset(
                instanciate_delayed(self.path),
                self.name,
                data_files=instanciate_delayed(self.data_files),
                revision=self.revision,
                cache_dir=tmp_dir,
            )
            print("Dataset:")
            print(ds)

            print("Saving...")
            ds.save_to_disk(self.out_dir.get())

            print("Done.")

    @classmethod
    def hash(cls, kwargs):
        # All other options are ignored for the hash, as they should not have an influence on the result.
        d = {
            "path": kwargs["path"],
            "name": kwargs["name"],
            "data_files": kwargs["data_files"],
            "revision": kwargs["revision"],
        }
        return super().hash(d)
