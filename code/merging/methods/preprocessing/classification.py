from merging.methods.preprocessing.base import Preprocessor

class SequenceClassificationPreprocessor(Preprocessor):

    def preprocess(self, features):
        sequences, labels = [], []
        for source, target in zip(features["input"], features["label"]):
            tokenized_source = self.tokenizer(
                self.model_args.prompt_prefix + source, 
                max_length=self.model_args.max_input_length,
                truncation=True
            )["input_ids"]
            target = int(target)
            sequences.append(tokenized_source)
            labels.append(target)

        return sequences, labels
