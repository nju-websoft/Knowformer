import torch
import torch.nn as nn
import copy
import math
from utils.tools import truncated_normal_init
from utils.tools import truncated_normal
from utils.tools import norm_layer_init
from utils.tools import device


def clones(module, n):
    module_list = nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
    if isinstance(module, nn.Linear):
        for i in range(0, n):
            module_list[i].__init__(module.in_features, module.out_features)
    elif isinstance(module, SublayerConnection):
        for i in range(0, n):
            module_list[i].__init__(module.size, module.residual_dropout_prob)
    return module_list


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, initializer_range):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        truncated_normal_init(self.lut, [vocab, d_model], initializer_range)

    def forward(self, x):
        # x.shape: (batch_size, seq_len=3, 1) -> (batch_size, seq_len=3)
        x = x.squeeze(-1)
        # out.shape: (batch_size, seq_len=3, d_model)
        out = self.lut(x)
        return out


def attention(query, key, value, mask, attention_dropout_layer,
                q_dropout_layer, k_dropout_layer, v_dropout_layer, delta=0.2):
    d_k = query.size(-1)
    
    query = q_dropout_layer(query)
    key = k_dropout_layer(key)
    value = v_dropout_layer(value)

    # scores.shape: (batch_size, head_size, seq_size=3, seq_size=3)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn.shape: (batch_size, head_size, seq_size, seq_size)
    p_attn = nn.functional.softmax(scores, dim=-1)
    
    p_attn = attention_dropout_layer(p_attn)
    # value.shape: (batch_size, head_size, seq_size, dim)
    # out.shape: (batch_size, head_size, seq_size, dim)
    out = torch.matmul(p_attn, value)
    return out, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, attention_dropout_prob=0.1, initializer_range=0.02, delta=0.2,
                    q_dropout_prob=0, k_dropout_prob=0, v_dropout_prob=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        for linear in self.linears:
            truncated_normal_init(linear, [d_model, d_model], initializer_range)
        self.attn = None
        self.attention_dropout_layer = nn.Dropout(p=attention_dropout_prob)
        self.q_dropout_layer = nn.Dropout(p=q_dropout_prob)
        self.k_dropout_layer = nn.Dropout(p=k_dropout_prob)
        self.v_dropout_layer = nn.Dropout(p=v_dropout_prob)
        self.delta = delta

    def forward(self, query, key, value, mask):
        # query, key, value shape: (batch_size,seq_size,d_model)
        # mask.shape: (batch_size,head_size,seq_size,seq_size)

        nbatches = query.size(0)

        # l(x) shape: (batch_size,seq_size,head_size,feature_size=d_model//head_size)
        # query, key, value shape: (batch_size,head_size,seq_size,feature_size=d_model//head_size)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).permute(0, 2, 1, 3)
             for l, x in zip(self.linears, (query, key, value))]

        # x.shape: (batch_size,head_size,seq_size,feature_size)
        # self.attn: (batch_size,head_size,seq_size,seq_size)
        x, self.attn = attention(query, key, value, mask=mask,
                                 attention_dropout_layer=self.attention_dropout_layer, 
                                 q_dropout_layer=self.q_dropout_layer, 
                                 k_dropout_layer=self.k_dropout_layer, 
                                 v_dropout_layer=self.v_dropout_layer,
                                 delta=self.delta)

        # x.shape: (batch_size,head_size,seq_size,feature_size) -> (batch_size,seq_size,d_model)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # out.shape: (batch_size,seq_size,d_model)
        out = self.linears[-1](x)
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, hidden_dropout_prob=0.1, initializer_range=0.02):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        truncated_normal_init(self.w_1, [d_ff, d_model], initializer_range)

        self.w_2 = nn.Linear(d_ff, d_model)
        truncated_normal_init(self.w_2, [d_model, d_ff], initializer_range)

        self.hidden_dropout_layer = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        # x.shape: (batch_size,seq_size=3,d_model)
        # return.shape: (batch_size,seq_size=3,d_model)
        return self.w_2(self.hidden_dropout_layer(nn.functional.gelu(self.w_1(x))))
        

class SublayerConnection(nn.Module):
    def __init__(self, size, residual_dropout_prob):
        super(SublayerConnection, self).__init__()
        self.size = size
        self.residual_dropout_prob = residual_dropout_prob
        self.norm = nn.LayerNorm(size)
        norm_layer_init(self.norm)
        self.residual_dropout_layer = nn.Dropout(self.residual_dropout_prob)

    def forward(self, x, sublayer):
        return x + self.residual_dropout_layer(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, size, residual_dropout_prob):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        self.sublayer = clones(SublayerConnection(size, residual_dropout_prob), 2)

    def forward(self, x, mask):
        # input x_q.shape: (batch_size,seq_size,d_model)
        # input x_k.shape: (batch_size,seq_size,d_model)
        # input x_v.shape: (batch_size,seq_size,d_model)
        # input mask.shape: (batch_size,head_size,seq_size,seq_size)
        # output x.shape: (batch_size,seq_size,d_model)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        out = self.sublayer[1](x, self.feed_forward)
        # out.shape (batch_size,seq_size,d_model)
        return out


class Encoder(nn.Module):
    def __init__(self, config, h_fc, t_fc):
        super(Encoder, self).__init__()
        self.config = config
        self.use_gelu = config["use_gelu"]
        self.residual_w = config['residual_w']
        layers = []
        for i in range(config['num_hidden_layers']):
            attn = MultiHeadedAttention(config['num_attention_heads'], config['hidden_size'],
                                        config['attention_dropout_prob'], config['initializer_range'], config['residual_w'],
                                        config['q_dropout_prob'], config['k_dropout_prob'], config['v_dropout_prob'])
            ff = PositionwiseFeedForward(config['hidden_size'], config['intermediate_size'],
                                         config['hidden_dropout_prob'], config['initializer_range'])
            layer = EncoderLayer(attn, ff, config['hidden_size'], config['residual_dropout_prob'])
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(self.layers[0].size)
        norm_layer_init(self.norm)

        self.h_fc_layers = clones(h_fc, config['num_hidden_layers'])
        self.t_fc_layers = clones(t_fc, config['num_hidden_layers'])

        self.hr_dropout_layer = nn.Dropout(p=config["relation_combine_dropout_prob"])
        self.tr_dropout_layer = nn.Dropout(p=config["relation_combine_dropout_prob"])
        
    def forward(self, x, mask):
        # x.shape: (batch_size, seq_size=3, d_model)
        # mask.shape: (batch_size, seq_size=3, seq_size=3)
        input_x = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            h_fc = self.h_fc_layers[i]
            t_fc = self.t_fc_layers[i]

            x_clone = x.clone()
            for col in range(0, x.shape[1], 2):
                if col == 0:
                    tmp = self.hr_dropout_layer(h_fc(input_x[:,2,:] - input_x[:,1,:]))
                    if self.use_gelu:
                        tmp = nn.functional.gelu(tmp)
                    x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp
                elif col == x.shape[1]-1:
                    tmp = self.tr_dropout_layer(t_fc(input_x[:,col-2,:] + input_x[:,col-1,:]))
                    if self.use_gelu:
                        tmp = nn.functional.gelu(tmp)
                    x[:,col,:] = x_clone[:,col,:] + self.residual_w * tmp
                else:
                    tmp1 = self.tr_dropout_layer(t_fc(input_x[:,col-2,:] + input_x[:,col-1,:]))
                    if self.use_gelu:
                        tmp1 = nn.functional.gelu(tmp1)
                    tmp2 = self.hr_dropout_layer(h_fc(input_x[:,col+2,:] - input_x[:,col+1,:]))
                    if self.use_gelu:
                        tmp2 = nn.functional.gelu(tmp2)
                    x[:,col,:] = x_clone[:,col,:] + self.residual_w / 2 * tmp1 + self.residual_w / 2 * tmp2
            
            input_x = x
            x = layer(x, mask)

        return self.norm(x)
