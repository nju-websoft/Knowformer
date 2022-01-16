import torch
import torch.nn as nn
from model.knowformer_encoder import Embeddings
from model.knowformer_encoder import Encoder
from utils.tools import truncated_normal_init
from utils.tools import norm_layer_init
from utils.tools import device


class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._input_dropout_prob = config['input_dropout_prob']
        self._attention_dropout_prob = config['attention_dropout_prob']
        self._hidden_dropout_prob = config['hidden_dropout_prob']
        self._residual_dropout_prob = config['residual_dropout_prob']
        self._initializer_range = config['initializer_range']
        self._intermediate_size = config['intermediate_size']
        self._residual_w = config['residual_w']
        self._relation_combine_dropout_prob = config['relation_combine_dropout_prob']
        self._use_gelu = config["use_gelu"]

        self._voc_size = config['vocab_size']
        self._n_relation = config['num_relations']

        self.ele_embedding = Embeddings(self._emb_size, self._voc_size, self._initializer_range)

        
        ####### for ablation study on our position embedding #######
        self.abs_pos_embedding = Embeddings(self._emb_size, 40, self._initializer_range)
        ############################################################
        
        self.h_fc = nn.Linear(self._emb_size, self._emb_size)
        truncated_normal_init(self.h_fc, [self._emb_size, self._emb_size], self._initializer_range)

        self.t_fc = nn.Linear(self._emb_size, self._emb_size)
        truncated_normal_init(self.t_fc, [self._emb_size, self._emb_size], self._initializer_range)

        self.encoder = Encoder(config, self.h_fc, self.t_fc)

        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.input_norm_layer = nn.LayerNorm(self._emb_size)
        norm_layer_init(self.input_norm_layer)

        self.output_fc_entity = nn.Linear(self._emb_size, self._emb_size)
        truncated_normal_init(self.output_fc_entity, [self._emb_size, self._emb_size], self._initializer_range)

        self.output_fc_act = nn.functional.gelu

        self.output_norm_layer_entity = nn.LayerNorm(self._emb_size)
        norm_layer_init(self.output_norm_layer_entity)
        
        self.bias_0 = torch.nn.Parameter(torch.zeros(self._voc_size))
        self.bias_1 = torch.nn.Parameter(torch.zeros(self._voc_size))

    def forward(self, src_ids, input_mask, mask_pos, mask_index=-1, mask_pos_2=None, r_flag=None):
        # mask_pos_2, r_flag is used in adding another loss in triples with length 3
        # training data with length more than 3 can not use this
        if mask_pos_2 is not None or r_flag is not None:
            assert mask_pos.shape == mask_pos_2.shape
            assert mask_pos_2.shape == r_flag.shape
        if src_ids.shape[1] != 3:
            assert mask_pos_2 is None and r_flag is None

        # src_ids.shape (batch_size, seq_size, 1)
        # input_mask shape = (batch_size, seq_size, seq_size)
        # mask_pos shape = (batch_size, 1)

        emb_out = self.ele_embedding(src_ids)
        emb_out = self.input_norm_layer(emb_out)
        emb_out = self.input_dropout_layer(emb_out)

        # self_attn_mask.shape (batch_size, seq_size, seq_size)
        self_attn_mask = input_mask

        # n_head_self_attn_mask.shape (batch_size, head_size, seq_size, seq_size)
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)
        n_head_self_attn_mask.requires_grad = False

        # out.shape (batch_size, seq_size, d_model)
        out = self.encoder(emb_out, n_head_self_attn_mask)
        
        assert mask_index == -1 or mask_index % 2 == 0
        assert mask_index < src_ids.shape[1]
        
        if mask_pos_2 is None or r_flag is None:
            fc_out_other = None
        else:
            r_embeddings = out[:, 1, :] * r_flag
            other_embeddings = out.reshape(-1, self._emb_size)[mask_pos_2.squeeze(-1)]
            other_plus_or_minus_r_embeddings = r_embeddings + other_embeddings
            fc_out_other = torch.matmul(other_plus_or_minus_r_embeddings, self.ele_embedding.lut.weight.transpose(0, 1))
            fc_out_other += self.bias_1.expand_as(fc_out_other)
                
        reshaped_out = out.reshape(-1, self._emb_size)
        mask_feat = reshaped_out[mask_pos.squeeze(-1)]

        mask_trans_feat = mask_feat
        mask_trans_feat = self.output_fc_entity(mask_feat)
        mask_trans_feat = self.output_fc_act(mask_trans_feat)
        mask_trans_feat = self.output_norm_layer_entity(mask_trans_feat)
        
        fc_out = torch.matmul(mask_trans_feat, self.ele_embedding.lut.weight.transpose(0, 1))
        fc_out += self.bias_0.expand_as(fc_out)
        # fc_out.shape (batch_size, self._voc_size)
        # fc_out_other.shape (batch_size, self._voc_size) or none
        return fc_out, fc_out_other
