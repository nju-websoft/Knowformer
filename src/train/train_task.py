import os
import time
import torch
import numpy as np
from reader.kg_reader import KGDataReader
from utils.loss_func import cross_entropy
# from utils.loss_func import bce_loss
from reader.batching import prepare_batch_data
from utils.tools import device
from valid.evaluate_kbc import kbc_predict
from utils.swa import swa
from valid.test_task import entity_alignment_test
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from contiguous_params import ContiguousParams
import torch.nn.functional as F


def link_prediction_train(args, my_model, logger, saved_model_folder):
    train_data_reader = KGDataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                     data_path=os.path.join(args.dataset_root_path, args.dataset, args.lp_train_file),
                                     batch_size=args.batch_size,
                                     is_training=True)
    valid_data_reader = KGDataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                     data_path=os.path.join(args.dataset_root_path, args.dataset, args.lp_valid_file),
                                     batch_size=args.eval_batch_size,
                                     is_training=False)
    test_data_reader = KGDataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                    data_path=os.path.join(args.dataset_root_path, args.dataset, args.lp_test_file),
                                    batch_size=args.eval_batch_size,
                                    is_training=False)
    
    criterion = cross_entropy
    optimizer = torch.optim.AdamW(ContiguousParams(my_model.parameters()).contiguous(), lr=args.learning_rate)
    
    ######## for early stop ###########
    max_hits1, times = 0, 0
    top_five_hits1_list = [0] * 20
    top_five_hits1_index_list = [-1] * 20
    ###################################

    is_relation = None
    one_hot_labels = None
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        epoch_loss = list()
        epoch_another_loss = list()
        my_model.train()
        
        for batch in train_data_reader.data_generator():
            batch_data = prepare_batch_data(batch, -1, train_data_reader.mask_id)
            src_ids, input_mask, mask_label, mask_pos, mask_pos_2, r_flag = batch_data
            src_ids = torch.LongTensor(src_ids).to(device)
            input_mask = torch.LongTensor(input_mask).to(device)
            mask_label = torch.LongTensor(mask_label).to(device)
            mask_pos = torch.LongTensor(mask_pos).to(device)
            if mask_pos_2 is not None:
                mask_pos_2 = torch.LongTensor(mask_pos_2).to(device)
            if r_flag is not None:
                r_flag = torch.LongTensor(r_flag).to(device)
            
            fc_out, fc_out_other = my_model(src_ids, input_mask, mask_pos, mask_index=-1, mask_pos_2=mask_pos_2, r_flag=r_flag)

            if one_hot_labels is None or one_hot_labels.shape[0] != mask_label.shape[0]:
                one_hot_labels = torch.zeros(mask_label.shape[0], args.vocab_size). \
                    to(device).scatter_(1, mask_label, 1)
            else:
                one_hot_labels.fill_(0).scatter_(1, mask_label, 1)
            

            if is_relation is None or is_relation.shape[0] != mask_label.shape[0]:
                entity_indicator = torch.zeros(mask_label.shape[0], args.vocab_size - args.num_relations).to(
                    device)
                relation_indicator = torch.ones(mask_label.shape[0], args.num_relations).to(device)
                is_relation = torch.cat((entity_indicator, relation_indicator), dim=-1)

            soft_labels = one_hot_labels * args.soft_label \
                            + (1.0 - one_hot_labels - is_relation) \
                            * ((1.0 - args.soft_label) / (args.vocab_size - 1 - args.num_relations))
            soft_labels.requires_grad = False
            
            loss = criterion(fc_out, soft_labels)
            epoch_loss.append(loss.item())

            if fc_out_other is not None:
                loss_other = criterion(fc_out_other, soft_labels)
                loss_other *= args.addition_loss_w
                epoch_another_loss.append(loss_other.item())
                loss += loss_other

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()       
        msg = "epoch: %d, epoch loss: %f, epoch another loss: %f, training time: %f" % (epoch, np.float(np.mean(epoch_loss)),
                                                                np.float(np.mean(epoch_another_loss)),
                                                                time.time() - start_time)
        print(msg)
        logger.info(msg)
        if epoch % args.eval_freq == 0:
            my_model.eval()
            with torch.no_grad():
                logger.info("before do validation, do test first")
                kbc_predict(args, logger, my_model, test_data_reader)
                logger.info("now do validation")
                eval_performance = kbc_predict(args, logger, my_model, valid_data_reader)
                # for early stop
                if eval_performance['fhits1'] > max_hits1:
                    max_hits1 = eval_performance['fhits1']
                    times = 0
                else:
                    times += 1
            if epoch >= args.start_save and epoch % args.save_freq == 0:
                ######## for early stop ###########
                min_top_five_hits1 = min(top_five_hits1_list)
                if eval_performance['fhits1'] > min_top_five_hits1:
                    replaced_idx = top_five_hits1_list.index(min_top_five_hits1)
                    top_five_hits1_list[replaced_idx] = eval_performance['fhits1']
                    origin_index = top_five_hits1_index_list[replaced_idx]
                    top_five_hits1_index_list[replaced_idx] = epoch
                    if os.path.exists(os.path.join(saved_model_folder, "params_epoch_{}.ckpt".format(origin_index))):
                        print("remove {} model".format(origin_index))
                        os.remove(os.path.join(saved_model_folder, "params_epoch_{}.ckpt".format(origin_index)))
                    print("save current model...")
                    saved_model_file = os.path.join(saved_model_folder, "params_epoch_{}.ckpt".format(epoch))
                    torch.save(my_model.state_dict(), saved_model_file)
                    logger.info("save model file: {} at epoch: {}".format(saved_model_file, epoch))
                ###################################
                # saved_model_file = os.path.join(saved_model_folder, "params_epoch_{}.ckpt".format(epoch))
                # torch.save(my_model.state_dict(), saved_model_file)
            # for early stop
            if times >= args.early_stop_max_times and epoch >= args.min_epochs:
                logger.info("early stop at this epoch")
                break
    logger.info("Finish training")


def entity_alignment_train(args, my_model, logger, saved_model_folder):
    train_data_reader = KGDataReader(vocab_path=os.path.join(args.dataset_root_path, args.dataset, args.vocab_file),
                                     data_path=os.path.join(args.dataset_root_path, args.dataset,
                                                            args.ea_train_triples_file),
                                     batch_size=args.batch_size,
                                     is_training=True)
    criterion = cross_entropy
    optimizer = torch.optim.Adam(my_model.parameters(), lr=args.learning_rate)

    max_hits1, times = 0, 0
    is_relation = None
    one_hot_labels = None
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        epoch_loss = list()
        epoch_another_loss = list()
        my_model.train()

        for batch in train_data_reader.data_generator():
            # for mask_index in range(0, train_data_reader.seq_len, 2):
            mask_index = -1
            batch_data = prepare_batch_data(batch, -1, train_data_reader.mask_id)
            # batch_data = prepare_batch_data(batch, mask_index, train_data_reader.mask_id)
            src_ids, input_mask, mask_label, mask_pos, mask_pos_2, r_flag = batch_data
            src_ids = torch.LongTensor(src_ids).to(device)
            input_mask = torch.LongTensor(input_mask).to(device)
            mask_label = torch.LongTensor(mask_label).to(device)
            mask_pos = torch.LongTensor(mask_pos).to(device)
            if mask_pos_2 is not None:
                mask_pos_2 = torch.LongTensor(mask_pos_2).to(device)
            if r_flag is not None:
                r_flag = torch.LongTensor(r_flag).to(device)

            fc_out, fc_out_other = my_model(src_ids, input_mask, mask_pos, mask_index=mask_index, mask_pos_2=mask_pos_2, r_flag=r_flag)

            if one_hot_labels is None or one_hot_labels.shape[0] != mask_label.shape[0]:
                one_hot_labels = torch.zeros(mask_label.shape[0], args.vocab_size). \
                    to(device).scatter_(1, mask_label, 1)
            else:
                one_hot_labels.fill_(0).scatter_(1, mask_label, 1)

            if is_relation is None or is_relation.shape[0] != mask_label.shape[0]:
                entity_indicator = torch.zeros(mask_label.shape[0], args.vocab_size - args.num_relations).to(
                    device)
                relation_indicator = torch.ones(mask_label.shape[0], args.num_relations).to(device)
                is_relation = torch.cat((entity_indicator, relation_indicator), dim=-1)

            soft_labels = one_hot_labels * args.soft_label \
                            + (1.0 - one_hot_labels - is_relation) \
                            * ((1.0 - args.soft_label) / (args.vocab_size - 1 - args.num_relations))
            soft_labels.requires_grad = False

            loss = criterion(fc_out, soft_labels)
            epoch_loss.append(loss.item())

            if fc_out_other is not None:
                loss_other = criterion(fc_out_other, soft_labels)
                loss_other *= args.addition_loss_w
                epoch_another_loss.append(loss_other.item())
                loss += loss_other

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        msg = "epoch: %d, epoch loss: %f, epoch another loss: %f, training time: %f" % (epoch, np.float(np.mean(epoch_loss)),
                                                                np.float(np.mean(epoch_another_loss)),
                                                                time.time() - start_time)
        print(msg)
        logger.info(msg)
        if epoch % args.eval_freq == 0:
            logger.info("do valid")
            my_model.eval()
            with torch.no_grad():
                eval_hits1_performance = entity_alignment_test(args, my_model, logger)
                if eval_hits1_performance > max_hits1:
                    max_hits1 = eval_hits1_performance
                    times = 0
                    saved_model_file = os.path.join(saved_model_folder, "params_best.ckpt")
                    torch.save(my_model.state_dict(), saved_model_file)
                    logger.info("save model file: {} at epoch: {}".format(saved_model_file, epoch))
                else:
                    times += 1
                if times >= args.early_stop_max_times and epoch >= args.min_epochs:
                    logger.info("early stop at this epoch")
                    break
    logger.info("Finish training")
