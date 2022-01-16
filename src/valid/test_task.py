import torch
import os
import numpy as np
from reader.kg_reader import KGDataReader
from reader.ea_reader import EADataReader
from reader.helper import load_vocab
from valid.evaluate_kbc import kbc_predict
from valid.evaluate_ea import ea_evaluation
from utils.tools import device


def link_prediction_test(args, my_model, logger):
    print("testing...")
    test_data_reader = KGDataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                    data_path=os.path.join(args.dataset_root_path, args.dataset, args.lp_test_file),
                                    batch_size=args.eval_batch_size,
                                    is_training=False)
    my_model.eval()
    with torch.no_grad():
        kbc_predict(args, logger, my_model, test_data_reader)
    logger.info("Finish test linking prediction task")


def entity_alignment_test(args, my_model, logger):
    top_k = [1, 3, 10]
    test_data_reader = EADataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                    data_path=os.path.join(args.dataset_root_path, args.dataset, args.ea_ref_file))
    test_data = np.array(test_data_reader.ent_pairs)

    kg1_entities = test_data[:, 0]
    kg2_entities = test_data[:, 1]

    add_kg2_entities_index = set()
    with open(os.path.join(args.dataset_root_path, args.dataset, args.ea_target_triples_file), encoding="utf-8") as f:
        this_vocab = load_vocab(os.path.join(args.dataset_root_path, args.dataset, args.vocab_file))
        for line in f:
            h, r, t = line.strip().split()
            if h in this_vocab.keys():
                add_kg2_entities_index.add(this_vocab[h])
            if t in this_vocab.keys():
                add_kg2_entities_index.add(this_vocab[t])
    add_kg2_entities_index -= set(list(kg2_entities))
    add_kg2_entities_index = list(add_kg2_entities_index)

    kg1_entities_index = list(kg1_entities)
    kg2_entities_index = list(kg2_entities) + add_kg2_entities_index

    assert len(kg2_entities_index) == len(set(kg2_entities_index))

    my_model.eval()
    with torch.no_grad():
        kg1_entities_ids = torch.LongTensor(kg1_entities_index).to(device)
        kg2_entities_ids = torch.LongTensor(kg2_entities_index).to(device)
        embeds1 = my_model.ele_embedding.lut(kg1_entities_ids)
        embeds2 = my_model.ele_embedding.lut(kg2_entities_ids)
        embeds1 = embeds1.cpu().numpy()
        embeds2 = embeds2.cpu().numpy()
    alignment_rest_12, hits1_12, mrr_12, msg = ea_evaluation(embeds1, embeds2, None, top_k, threads_num=4,
                                                             metric='inner', normalize=False, csls_k=0, accurate=True)
    logger.info("{}\n".format(msg))
    _, _, _, msg = ea_evaluation(embeds1, embeds2, None, top_k, threads_num=4,
                                                             metric='inner', normalize=False, csls_k=2, accurate=True)
    logger.info("csls_k=2 : {}\n".format(msg))
    return hits1_12
