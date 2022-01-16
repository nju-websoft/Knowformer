import collections


def write_vocab_path(vocab_path, entities_list, relations_list):
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("[PAD]\n")
        for i in range(0, 95):
            f.write("[unused{}]\n".format(i))
        f.write("[UNK]\n")
        f.write("[CLS]\n")
        f.write("[SEP]\n")
        f.write("[MASK]\n")
        for entity in entities_list:
            f.write(entity + "\n")
        for relation in relations_list:
            f.write(relation + "\n")


def load_vocab(vocab_path):
    vocab = collections.OrderedDict()
    with open(vocab_path, encoding="utf-8") as f:
        for num, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 1
            token = items[0].strip()
            vocab[token] = num
    return vocab


def convert_tokens_to_ids(vocab, tokens):
    output = []
    for item in tokens:
        output.append(vocab[item])
    return output


def read_triples(triples_path):
    res = []
    with open(triples_path, encoding="utf-8") as f:
        for line in f:
            l = line.strip().split()
            res.append(tuple(l))
    return res


def write_triples(triples_path, triples, is_add_mask=True):
    with open(triples_path, "w", encoding="utf-8") as f:
        for triple in triples:
            res = "\t".join(triple)
            if is_add_mask:
                f.write(res + "\tMASK_0\n")
                f.write(res + "\tMASK_2\n")
            else:
                f.write(res + "\n")


def read_entity_paris(pairs_path):
    res = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            e1, e2 = line.strip().split()
            res.append((e1, e2))
    return res
