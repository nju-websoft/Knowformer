from reader.helper import load_vocab
from reader.helper import convert_tokens_to_ids


class EADataReader(object):
    def __init__(self, vocab_path, data_path):
        self.vocab = load_vocab(vocab_path)
        self.ent_pairs = self.read_ent_pairs(data_path)

    def read_ent_pairs(self, input_file):
        ent_pairs = []
        with open(input_file, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split("\t")
                ent_pair_id = convert_tokens_to_ids(self.vocab, tokens)
                ent_pairs.append(ent_pair_id)
        return ent_pairs
