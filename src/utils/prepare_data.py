import os
from reader.helper import read_triples
from reader.helper import write_triples
from reader.helper import read_entity_paris
from reader.helper import write_vocab_path
from reader.helper import load_vocab


def prepare_entity_alignment_data(args):
    sup_ents_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_sup_file)
    ref_ents_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_ref_file)
    ent_ILLs_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_all_file)
    s_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_source_triples_file)
    t_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_target_triples_file)
    train_triples_path = os.path.join(args.dataset_root_path, args.dataset, args.ea_train_triples_file)
    vocab_path = os.path.join(args.dataset_root_path, args.dataset, args.vocab_file)

    assert os.path.exists(s_triples_path) and os.path.exists(t_triples_path) and os.path.exists(
        ref_ents_path) and os.path.exists(ent_ILLs_path)
    if not os.path.exists(sup_ents_path):
        def create_sup_ents(sup_ents_path_, ref_ents_path_, ent_ILLs_path_):
            ent_ILLs = read_entity_paris(ent_ILLs_path_)
            ref_ents = read_entity_paris(ref_ents_path_)
            sup_ents = []
            for pair in ent_ILLs:
                if pair not in ref_ents:
                    sup_ents.append(pair)
            assert len(ent_ILLs) == len(ref_ents) + len(sup_ents)
            with open(sup_ents_path_, "w", encoding='utf-8') as f:
                for pair in sup_ents:
                    f.write("\t".join(pair) + "\n")

        create_sup_ents(sup_ents_path, ref_ents_path, ent_ILLs_path)

    if not os.path.exists(train_triples_path):
        def merge(train_triples_path_, s_triples_path_, t_triples_path_, sup_ents_path_):
            s_triples = read_triples(s_triples_path_)
            t_triples = read_triples(t_triples_path_)
            sup_ents = read_entity_paris(sup_ents_path_)
            t_s_map = {}
            for s, t in sup_ents:
                assert t not in t_s_map.keys()
                t_s_map[t] = s
            train_triples = s_triples.copy()
            for t_triple in t_triples:
                t_triple_h, t_triple_r, t_triple_t = t_triple
                t_triple_h = t_s_map.get(t_triple_h, t_triple_h)
                t_triple_t = t_s_map.get(t_triple_t, t_triple_t)
                train_triples.append((t_triple_h, t_triple_r, t_triple_t))
            assert len(train_triples) == len(s_triples) + len(t_triples)
            with open(train_triples_path_, "w", encoding='utf-8') as f:
                for triple in train_triples:
                    f.write("\t".join(triple) + "\n")

        merge(train_triples_path, s_triples_path, t_triples_path, sup_ents_path)

    entities_set = set()
    relations_set = set()
    train_triples = read_triples(train_triples_path)
    for triple in train_triples:
        for i, label in enumerate(triple):
            if label.startswith("MASK"):
                continue
            if i % 2 == 0:
                entities_set.add(label)
            else:
                relations_set.add(label)
    entities_list = sorted(list(entities_set))
    relations_list = sorted(list(relations_set))
    args.vocab_size = 100 + len(entities_list) + len(relations_list)
    args.num_relations = len(relations_list)

    if not os.path.exists(vocab_path):
        write_vocab_path(vocab_path, entities_list, relations_list)
    else:
        assert args.vocab_size == len(load_vocab(vocab_path))


def prepare_link_prediction_data(args):
    train_origin_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_train_origin_file)
    valid_origin_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_valid_origin_file)
    test_origin_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_test_origin_file)
    assert os.path.exists(train_origin_path) and os.path.exists(valid_origin_path) and os.path.exists(test_origin_path)

    vocab_path = os.path.join(args.dataset_root_path, args.dataset, args.vocab_file)

    true_triple_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_all_true_file)

    train_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_train_file)
    valid_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_valid_file)
    test_path = os.path.join(args.dataset_root_path, args.dataset, args.lp_test_file)
    if not os.path.exists(train_path):
        train_triples = read_triples(train_origin_path)
        write_triples(train_path, train_triples, is_add_mask=True)
    for write_path, read_path in zip([valid_path, test_path], [valid_origin_path, test_origin_path]):
        if not os.path.exists(write_path):
            tmp_triples = read_triples(read_path)
            write_triples(write_path, tmp_triples, is_add_mask=True)

    entities_set = set()
    relations_set = set()
    train_origin_triples = read_triples(train_origin_path)
    valid_origin_triples = read_triples(valid_origin_path)
    test_origin_triples = read_triples(test_origin_path)
    for triples in [train_origin_triples, valid_origin_triples, test_origin_triples]:
        for h, r, t in triples:
            entities_set.add(h)
            entities_set.add(t)
            relations_set.add(r)
    entities_list = sorted(list(entities_set))
    relations_list = sorted(list(relations_set))
    args.vocab_size = 100 + len(entities_list) + len(relations_list)
    args.num_relations = len(relations_list)

    if not os.path.exists(vocab_path):
        write_vocab_path(vocab_path, entities_set, relations_set)
    else:
        assert args.vocab_size == len(load_vocab(vocab_path))

    all_triples = train_origin_triples + valid_origin_triples + test_origin_triples
    if not os.path.exists(true_triple_path):
        vocab = load_vocab(vocab_path)
        with open(true_triple_path, "w", encoding="utf-8") as f:
            for h, r, t in all_triples:
                f.write(str(vocab[h]) + "\t" + str(vocab[r]) + "\t" + str(vocab[t]) + "\n")
    else:
        assert len(read_triples(true_triple_path)) == len(all_triples)
