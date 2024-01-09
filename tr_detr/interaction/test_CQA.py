from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

# VSL的吗？对
class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.1):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.cqa_linear= nn.LSTM(4 * dim,
        #                         dim ,
        #                         num_layers=1,
        #                         bidirectional=False,
        #                         dropout=drop_rate,
        #                         batch_first=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output) # (batch_size, c_seq_len, dim)
        # output = self.cqa_linear(output)[0] 使用LSTM
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class VSLFuser(nn.Module):

    def __init__(self, dim=128, drop_rate=0.1, **kwargs):
        super().__init__()
        self.cq_attention = CQAttention(dim=dim, drop_rate=drop_rate)
        self.cq_concat = CQConcatenate(dim=dim)

    def forward(self, vfeats=None, qfeats=None, vmask=None, qmask=None, **kwargs):
        assert None not in [vfeats, qfeats, vmask, qmask]
        # if vmask == None:
        #     vmask = torch.ones(vfeats.shape[:2]).cuda()
        #     qmask = torch.ones(qfeats.shape[:2]).cuda()
        feats = self.cq_attention(vfeats, qfeats, vmask, qmask)
        feats = self.cq_concat(feats, qfeats, qmask)
        return F.relu(feats)

# - src_txt: [batch_size, L_txt, D_txt]
# - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
#     will convert to 1 as padding later for transformer
# - src_vid: [batch_size, L_vid, D_vid]
# - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
#     will convert to 1 as padding later for transformer

# src_txt_mask = torch.ones((32, 14))
# src_vid_mask = torch.ones((32, 75))
# input_txt = torch.ones((32, 14, 512))
# input_vis = torch.ones((32, 75, 512))
# # 注意此处传入对应的隐藏维度
# model = VSLFuser(dim = 512)
# output = model(input_vis, input_txt, src_vid_mask, src_txt_mask)
# print(output.shape)