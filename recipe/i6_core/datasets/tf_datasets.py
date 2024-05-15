"""
This module adds jobs for TF datasets, as documented here:
https://www.tensorflow.org/datasets
"""


from typing import Optional
from sisyphus import *


class DownloadAndPrepareTfDatasetJob(Job):
    """
    This job downloads and prepares a TF dataset.
    The processed files are stored in a `data_dir` folder,
    from where it can be loaded again (see https://www.tensorflow.org/datasets/overview#load_a_dataset)

    Install the dependencies:

        pip install tensorflow-datasets

    It further needs some extra dependencies, for example for 'librispeech':

        pip install apache_beam
        pip install pydub
        # ffmpeg installed

    See here for some more:
    https://github.com/tensorflow/datasets/blob/master/setup.py

    Also maybe::

        pip install datasets  # for Huggingface community datasets
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        max_simultaneous_downloads: Optional[int] = None,
        max_workers: Optional[int] = None,
    ):
        """
        :param dataset_name: Name of the dataset in the official TF catalog or community catalog.
            Available datasets can be found here:
            https://www.tensorflow.org/datasets/overview
            https://www.tensorflow.org/datasets/catalog/overview
            https://www.tensorflow.org/datasets/community_catalog/huggingface
        :param max_simultaneous_downloads: simultaneous downloads for tfds.load,
            some datasets might not work with the internal defaults,
            so use e.g. 1 in the case of librispeech.
            (https://github.com/tensorflow/datasets/issues/3885)
        :param max_workers: max workers for download extractor and Apache Beam,
            the default (cpu core count) might cause high memory load,
            so reduce this to a number smaller than the number of cores.
            (https://github.com/tensorflow/datasets/issues/3887)
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.max_simultaneous_downloads = max_simultaneous_downloads
        self.max_workers = max_workers

        self.out_data_dir = self.output_path("data_dir", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import tensorflow_datasets as tfds
        from apache_beam.options.pipeline_options import PipelineOptions

        if self.max_simultaneous_downloads:
            orig_get_downloader = tfds.download.downloader.get_downloader

            # https://github.com/tensorflow/datasets/issues/3885
            def _patched_get_downloader(*args, **kwargs):
                kwargs.setdefault(
                    "max_simultaneous_downloads", self.max_simultaneous_downloads
                )
                return orig_get_downloader(*args, **kwargs)

            tfds.download.downloader.get_downloader = _patched_get_downloader

        if self.max_workers:
            orig_get_extractor = tfds.download.extractor.get_extractor

            # https://github.com/tensorflow/datasets/issues/3887
            def _patched_get_extractor(*args, **kwargs):
                kwargs.setdefault("max_workers", self.max_workers)
                return orig_get_extractor(*args, **kwargs)

            tfds.download.extractor.get_extractor = _patched_get_extractor

        # Just use tfds.load, which will download and prepare the data.
        # Additionally, we can then do a simple sanity check.
        beam_options = {}
        if self.max_workers:
            beam_options.update(
                dict(
                    # https://cloud.google.com/dataflow/docs/reference/pipeline-options
                    max_num_workers=1,
                    number_of_worker_harness_threads=1,
                    num_workers=1,
                )
            )
        ds, info = tfds.load(
            self.dataset_name,
            data_dir=self.out_data_dir.get(),
            download_and_prepare_kwargs=dict(
                download_config=tfds.download.DownloadConfig(
                    beam_options=PipelineOptions.from_dictionary(beam_options)
                ),
            ),
            with_info=True,
        )
        print("Info:")
        print(info)
        print("Dataset take(1):")
        entry = ds.take(1)
        print(entry)

    @classmethod
    def hash(cls, kwargs):
        # All other options except dataset_name are ignored, as they should not have an influence on the result.
        d = {
            "dataset_name": kwargs["dataset_name"],
        }
        return super().hash(d)
