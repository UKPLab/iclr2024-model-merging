# http://parl.ai/downloads/fits/fits_data_v0.1.tar.gz 

import abc

import datasets

_CITATION = ""
_DESCRIPTION = ""
_HOMEPAGE = ""

import json
import logging

import datasets
import numpy as np
import transformers
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SequenceClassificationDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Art."""

    def __init__(self, **kwargs):
        """BuilderConfig for Art.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SequenceClassificationDatasetConfig, self).__init__(**kwargs)

class SequenceClassificationDataset(object):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    BUILDER_CONFIGS = [
        SequenceClassificationDatasetConfig(
            name=name,
            version=datasets.Version("1.0.0"),
            description=""
        ) for name in ["classification", "generation"]
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "dataset_id": datasets.Value("string"),
                    "input": datasets.Value("string"),
                    "label": datasets.Value("string")
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, filepath):
        pass

    @abc.abstractmethod
    def _map_to_common_format(self, sample):
        pass

    def _download_files(self, urls, data_files, dl_manager):
        if data_files is not None:
            raise NotImplementedError()
        return dl_manager.download_and_extract(urls)
