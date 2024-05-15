from typing import List

import datasets
from datasets import load_dataset
from .base import SequenceClassificationDataset

class IMBD(SequenceClassificationDataset, datasets.GeneratorBasedBuilder):
  
    _LABEL_MAP = {
        1: "positive",
        0: "negative",
        -1: "none"
    }

    def _map_to_common_format(self, sample):
        formatted_sample = {
            "dataset_id": "imdb",
            "input": sample["text"],
        }
        if self.config.name == "classification":
            formatted_sample["label"] = sample["label"]
        else:
            formatted_sample["label"] = self._LABEL_MAP[sample["label"]]

        return formatted_sample

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        dataset = load_dataset("imdb")

        splits = ["train", "test"]
        hf_splits = [datasets.Split.TRAIN, datasets.Split.TEST]
        split_data = {split: [] for split in splits}

        for split in splits:
            for sample in dataset[split]:
                split_data[split].append(self._map_to_common_format(sample))

        return [
            datasets.SplitGenerator(
                name=ds_split, gen_kwargs={
                    "data": split_data[split],
                })
            for ds_split, split in zip(hf_splits, splits)
        ]

    def _generate_examples(self, data):
        for idx, sample in enumerate(data):
            if not "id" in sample:
                sample["id"] = str(idx)
            yield idx, sample
