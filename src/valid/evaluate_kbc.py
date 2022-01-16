import collections
import torch
import os
from utils.tools import device
from reader.batching import prepare_batch_data
from utils.tools import print_results
import numpy as np
import timeit
import json
from reader.kg_reader import KGDataReader


# input_file: "to_predict_blank.txt"
# output_file: "result.json"
def predict_blank(args, my_model, input_file, output_file):
    myid2entity = {}
    with open(os.path.join(args.dataset_root_path, args.dataset, args.vocab_file)) as f:
        for i, line in enumerate(f):
            myid2entity[i] = line.strip()
    myentity2id = {v:k for k, v in myid2entity.items()}
        
    start = timeit.default_timer()
    true_triples_dict = load_kbc_eval_dict(os.path.join(args.dataset_root_path, args.dataset, args.lp_all_true_file))
    blank_data_reader = KGDataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                    data_path=os.path.join(args.dataset_root_path, args.dataset, input_file),
                                    batch_size=args.eval_batch_size,
                                    is_training=False)
    dump_dict = {}
    dump_dict["results"] = []
    examples = blank_data_reader.examples
    all_length = len(examples)
    examples_index = 0
    for batch in blank_data_reader.data_generator():
        batch_data = prepare_batch_data(batch, -1, blank_data_reader.mask_id)
        src_ids, input_mask, _, mask_pos, mask_pos_2, r_flag = batch_data
        src_ids = torch.LongTensor(src_ids).to(device)
        input_mask = torch.LongTensor(input_mask).to(device)
        mask_pos = torch.LongTensor(mask_pos).to(device)
        fc_out, fc_out_other = my_model(src_ids, input_mask, mask_pos, mask_pos_2=None, r_flag=None)
        
        # fc_out = torch.nn.functional.softmax(fc_out, dim=1)

        batch_results = fc_out.cpu().numpy()
        # batch_results[:, 0:100] = -np.Inf
        # batch_results[:, my_model._voc_size - my_model._n_relation:] = -np.Inf
        # print(blank_data_reader.examples[0])
        # print(batch_results[0][200:214])
        # assert 0

        for one_sample_score in batch_results:
            score_index = np.argsort(one_sample_score)[::-1]
            this_predict = []
            for index in score_index:
                if len(this_predict) == 1000: # Todo fb15k237:100 WN18RR:1000
                    break
                h, r, t = examples[examples_index][0], examples[examples_index][1], examples[examples_index][2]
                if examples[examples_index][-1] == "MASK_0":
                    if index in true_triples_dict[r]['hs'][t] and index != h:
                        continue
                elif examples[examples_index][-1] == "MASK_2":
                    if index in true_triples_dict[r]['ts'][h] and index != t:
                        continue
                else:
                    assert 0
                this_predict.append(index)
            this_predict_map = []
            for my_index, i in enumerate(this_predict):
                this_predict_map.append((myid2entity[i], str(one_sample_score[i])))
            dump_dict["results"].append(this_predict_map)
            print('[{:.3f}s] #predict triple: {}/{}'.format(timeit.default_timer() - start, examples_index, all_length), end='\r')
            examples_index += 1
    print()
    with open(os.path.join(args.dataset_root_path, args.dataset, output_file), "w") as f:
        json.dump(dump_dict,f)
    # without_score_dump_dict = {}
    # without_score_dump_dict["results"] = []
    # for one_result in dump_dict["results"]:
    #     without_score_one_result = []
    #     for e, s in one_result:
    #         without_score_one_result.append(e)
    #     without_score_dump_dict["results"].append(without_score_one_result)
    # with open(os.path.join(args.dataset_root_path, args.dataset, "wo_score_" + output_file), "w") as f:
    #     json.dump(without_score_dump_dict,f)


def load_kbc_eval_dict(true_triple_file):
    def load_true_triples(triple_file):
        triples = []
        with open(triple_file, encoding="utf-8") as f:
            for line in f.readlines():
                tokens = line.strip("\r \n").split("\t")
                assert len(tokens) == 3
                triples.append(
                    (int(tokens[0]), int(tokens[1]), int(tokens[2])))
        return triples

    true_triples = load_true_triples(true_triple_file)
    true_triples_dict = collections.defaultdict(lambda: {'hs': collections.defaultdict(list),
                                                         'ts': collections.defaultdict(list)})
    for h, r, t in true_triples:
        true_triples_dict[r]['ts'][h].append(t)
        true_triples_dict[r]['hs'][t].append(h)
    return true_triples_dict


def kbc_predict(args, logger, my_model, test_data_reader):
    true_triplets_dict = load_kbc_eval_dict(os.path.join(args.dataset_root_path, args.dataset, args.lp_all_true_file))

    eval_i = 0
    step = 0

    batch_eval_rets = []
    # batch_eval_rets_head = []
    # batch_eval_rets_tail = []

    filtered_batch_eval_rets = []
    # filtered_batch_eval_rets_head = []
    # filtered_batch_eval_rets_tail = []
    for batch in test_data_reader.data_generator():
        batch_data = prepare_batch_data(batch, -1, test_data_reader.mask_id)
        src_ids, input_mask, _, mask_pos, mask_pos_2, r_flag = batch_data
        src_ids = torch.LongTensor(src_ids).to(device)
        input_mask = torch.LongTensor(input_mask).to(device)
        mask_pos = torch.LongTensor(mask_pos).to(device)
        # mask_pos_2 = torch.LongTensor(mask_pos_2).to(device)
        # r_flag = torch.LongTensor(r_flag).to(device)

        fc_out, fc_out_other = my_model(src_ids, input_mask, mask_pos, mask_pos_2=None, r_flag=None)
        #########################
        # fc_out = torch.sigmoid(fc_out)
        # fc_out_other = torch.sigmoid(fc_out_other)
        # fc_out = fc_out + args.addition_loss_w * fc_out_other
        # fc_out = fc_out
        #########################
        batch_results = fc_out.cpu().numpy()
        
        #########################
        batch_results[:, 0:100] = -np.Inf
        batch_results[:, my_model._voc_size - my_model._n_relation:] = -np.Inf
        #########################
        
        _batch_len = fc_out.shape[0]

        rank, filter_rank = kbc_batch_evaluation(eval_i, test_data_reader.examples, batch_results, true_triplets_dict, args.lp_eval_type)

        batch_eval_rets.extend(rank)
        # batch_eval_rets_head.extend(rank[:len(rank)//2])
        # batch_eval_rets_tail.extend(rank[len(rank)//2:])

        filtered_batch_eval_rets.extend(filter_rank)
        # filtered_batch_eval_rets_head.extend(filter_rank[:len(filter_rank)//2])
        # filtered_batch_eval_rets_tail.extend(filter_rank[len(filter_rank)//2:])
        if step % 10 == 0:
            logger.info("Processing kbc_predict step: %d examples:%d" % (step, eval_i))
        step += 1
        eval_i += _batch_len

    eval_performance = compute_kbc_metrics(batch_eval_rets, filtered_batch_eval_rets)
    outs = "%s\t%.3f\t%.3f\t%.3f\t%.3f" % (args.dataset,
                                           eval_performance['fmrr'],
                                           eval_performance['fhits1'],
                                           eval_performance['fhits3'],
                                           eval_performance['fhits10'])
    msg = "\n----------- Evaluation Performance -----------\n%s\n%s" % \
          ("\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@10"]), outs)
    print_results(args.dataset, eval_performance)

    logger.info(msg)
    return eval_performance


def kbc_batch_evaluation(eval_i, all_examples, batch_results, tt, lp_eval_type):
    r_hts_idx = collections.defaultdict(list)
    scores_head = collections.defaultdict(list)
    scores_tail = collections.defaultdict(list)
    batch_r_hts_cnt = 0
    b_size = len(batch_results)
    for j in range(b_size):
        result = batch_results[j]
        i = eval_i + j
        example = all_examples[i]
        test_sample = example
        assert len(test_sample) == 4

        h, r, t = test_sample[0:3]
        _mask_type = test_sample[-1]
        if i % 2 == 0:
            r_hts_idx[r].append((h, t))
            batch_r_hts_cnt += 1
        if _mask_type == "MASK_0":
            scores_head[(r, t)] = result
        elif _mask_type == "MASK_2":
            scores_tail[(r, h)] = result
        else:
            raise ValueError("Unknown mask type in prediction example:%d" % i)

    rank = {}
    f_rank = {}

    def get_index(scores, target, eval_type):
        target_score = scores[target]
        if eval_type != "ordinal":
            scores = np.delete(scores, target)
            if eval_type == "top":
                scores = np.insert(scores, 0, target_score)
                sorted_idx = np.argsort(scores, kind='stable')[::-1]
                return np.where(sorted_idx == 0)[0][0] + 1
            elif eval_type == "bottom":
                scores = np.concatenate([scores, [target_score]], 0)
                sorted_idx = np.argsort(scores, kind='stable')[::-1]
                return np.where(sorted_idx == scores.shape[0]-1)[0][0] + 1
            elif eval_type == "random":
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, target_score)
                sorted_idx = np.argsort(scores, kind='stable')[::-1]
                return np.where(sorted_idx == rand)[0][0] + 1
            else:
                assert 0
        else:
            sorted_idx = np.argsort(scores, kind='stable')[::-1]
            return np.where(sorted_idx == target)[0][0] + 1

    for r, hts in r_hts_idx.items():
        r_rank = {'head': [], 'tail': []}
        r_f_rank = {'head': [], 'tail': []}
        for h, t in hts:
            assert lp_eval_type == "top" or lp_eval_type == "bottom" or lp_eval_type == "random" or lp_eval_type == "ordinal" 
            
            scores_t = scores_tail[(r, h)].copy()
            r_rank['tail'].append(get_index(scores_t, t, lp_eval_type))
            # scores_t = scores_tail[(r, h)].copy()
            # sorted_idx_t = np.argsort(scores_t)[::-1]
            # r_rank['tail'].append(np.where(sorted_idx_t == t)[0][0] + 1)

            rm_idx = tt[r]['ts'][h]
            rm_idx = [i for i in rm_idx if i != t]
            scores_t_filter = scores_tail[(r, h)].copy()
            for i in rm_idx:
                scores_t_filter[i] = -np.Inf
            r_f_rank['tail'].append(get_index(scores_t_filter, t, lp_eval_type))
            # rm_idx = tt[r]['ts'][h]
            # rm_idx = [i for i in rm_idx if i != t]
            # scores_t_filter = scores_tail[(r, h)].copy()
            # for i in rm_idx:
            #     scores_t_filter[i] = -np.Inf
            # sorted_idx_t = np.argsort(scores_t_filter)[::-1]
            # r_f_rank['tail'].append(np.where(sorted_idx_t == t)[0][0] + 1)

            scores_h = scores_head[(r, t)].copy()
            r_rank['head'].append(get_index(scores_h, h, lp_eval_type))
            # scores_h = scores_head[(r, t)].copy()
            # sorted_idx_h = np.argsort(scores_h)[::-1]
            # r_rank['head'].append(np.where(sorted_idx_h == h)[0][0] + 1)

            rm_idx = tt[r]['hs'][t]
            rm_idx = [i for i in rm_idx if i != h]
            scores_h_filter = scores_head[(r, t)].copy()
            for i in rm_idx:
                scores_h_filter[i] = -np.Inf
            r_f_rank['head'].append(get_index(scores_h_filter, h, lp_eval_type))
            # rm_idx = tt[r]['hs'][t]
            # rm_idx = [i for i in rm_idx if i != h]
            # scores_h_filter = scores_head[(r, t)].copy()
            # for i in rm_idx:
            #     scores_h_filter[i] = -np.Inf
            # sorted_idx_h = np.argsort(scores_h_filter)[::-1]
            # r_f_rank['head'].append(np.where(sorted_idx_h == h)[0][0] + 1)
        rank[r] = r_rank
        f_rank[r] = r_f_rank

    h_pos = [p for k in rank.keys() for p in rank[k]['head']]
    t_pos = [p for k in rank.keys() for p in rank[k]['tail']]
    f_h_pos = [p for k in f_rank.keys() for p in f_rank[k]['head']]
    f_t_pos = [p for k in f_rank.keys() for p in f_rank[k]['tail']]

    ranks = np.asarray(h_pos + t_pos)
    f_ranks = np.asarray(f_h_pos + f_t_pos)
    return ranks, f_ranks


def compute_kbc_metrics(rank_li, frank_li):
    rank_rets = np.array(rank_li).ravel()
    frank_rets = np.array(frank_li).ravel()
    mrr = np.mean(1.0 / rank_rets)
    fmrr = np.mean(1.0 / frank_rets)

    hits1 = np.mean(rank_rets <= 1.0)
    hits3 = np.mean(rank_rets <= 3.0)
    hits10 = np.mean(rank_rets <= 10.0)

    # filtered metrics
    fhits1 = np.mean(frank_rets <= 1.0)
    fhits3 = np.mean(frank_rets <= 3.0)
    fhits10 = np.mean(frank_rets <= 10.0)

    eval_result = {'mrr': np.float(mrr),
                   'hits1': np.float(hits1),
                   'hits3': np.float(hits3),
                   'hits10': np.float(hits10),
                   'fmrr': np.float(fmrr),
                   'fhits1': np.float(fhits1),
                   'fhits3': np.float(fhits3),
                   'fhits10': np.float(fhits10)}

    return eval_result
