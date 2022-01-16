import numpy as np
from collections import defaultdict
from reader.helper import convert_tokens_to_ids
from reader.helper import load_vocab


class KGDataReader(object):
    def __init__(self,
                 vocab_path,
                 data_path,
                 batch_size=4096,
                 is_training=True):
        self.vocab = load_vocab(vocab_path)
        self.mask_id = self.vocab["[MASK]"]
        self.batch_size = batch_size
        self.is_training = is_training
        self.seq_len = -1
        self.examples = self.read_example(data_path)

    def read_example(self, input_file):
        examples = []
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens[-1].startswith("MASK"):
                    token_seq_ids = convert_tokens_to_ids(self.vocab, tokens[:-1])
                    self.seq_len = max(self.seq_len, len(token_seq_ids))
                    token_seq_ids.append(tokens[-1])
                else:
                    token_seq_ids = convert_tokens_to_ids(self.vocab, tokens[:])
                    self.seq_len = max(self.seq_len, len(token_seq_ids))
                examples.append(token_seq_ids)
        return examples

    def data_generator(self):
        range_list = [i for i in range(len(self.examples))]
        if self.is_training:
            np.random.shuffle(range_list)
        for i in range(0, len(self.examples), self.batch_size):
            batch = []
            for j in range_list[i:i + self.batch_size]:
                batch.append(self.examples[j])
            yield batch
