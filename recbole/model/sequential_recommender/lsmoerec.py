import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.modules import MultiHeadAttention, LinearAttnExperEncoder, GRUExpertEncoder


class LSMoERec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(LSMoERec, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config["hidden_size"]
        self.max_seq_len = config['MAX_ITEM_LIST_LENGTH']
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.n_heads = config["n_heads"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        # init embedding layer
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # init Expert
        self.experts = nn.ModuleList(
            [
                LinearAttnExperEncoder(config),
                GRUExpertEncoder(config),
            ]
        )
        # init MoE layers
        self.gate_dense_0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Gelu = nn.GELU()
        self.moe_proj0 = nn.Linear(self.hidden_size, 1)
        self.moe_proj1 = nn.Linear(self.max_seq_len, 1)
        self.out_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.moe_out_drop = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, self.layer_norm_eps)
        # init Output Gate Layer
        self.output_gate_proj0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gate_proj1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gate_projF = nn.Linear(self.hidden_size, self.hidden_size)
        self.final_output_drop = nn.Dropout(self.hidden_dropout_prob)
        # init loss func
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        # 用于快速负采样评估
        self.NEG_FIELD = config['eval_args']['neg_field']

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, item_seq, item_seq_len):
        seq_embedding = self.item_embedding(item_seq)
        seq_embedding = self.emb_dropout(seq_embedding)
        filtered_resp = []
        # ------ experts filter------
        for expert in self.experts:
            # 通过每个专家的频域滤波网络对输入序列进行频域滤波，然后转回时域
            filter_out = expert.filter_layer(seq_embedding)  #(batch,seq_len_hidden_size)
            filtered_resp.append(filter_out)
        # 多专家网络滤波结果
        filtered_resp = torch.stack(filtered_resp, dim=0, ).permute(1, 0, 2, 3)  #(batch,M,seq_len,hidden_size)
        # ------ calculate moe gate -------
        moe_gate = self.moe_proj0(filtered_resp).squeeze(-1)  #(batch,M,seq_len)
        moe_gate = self.moe_proj1(moe_gate).squeeze(-1)  #(batch,M)
        moe_gate = torch.softmax(moe_gate, dim=1).unsqueeze(-1).unsqueeze(-1)  #(batch,M,1,1)
        # ------ calculate moe output ----
        expert_outputs = []
        for i, expert_encoder in enumerate(self.experts):
            filtered_input_tensor = filtered_resp[:, i, :, :]  #(batch,seq_len,hidden_size)
            expert_output = expert_encoder(filtered_input_tensor)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=0).permute(1, 0, 2, 3)  #(batch,M,seq_len,hidden_size)
        moe_output = moe_gate * expert_outputs
        moe_output = torch.sum(moe_output, dim=1)
        moe_hidden_gate = self.gate_dense_0(seq_embedding)
        moe_hidden_gate = self.Gelu(moe_hidden_gate)
        moe_output = moe_hidden_gate * moe_output
        moe_output = self.out_dense(moe_output)
        """
        Todo
        专家计算得到一个多专家编码结果 (batch,M,seq_len,hidden_size)
        根据 filtered_resp 计算一个#(batch,M)的门控，沿着dim=1进行softmax
        然后拓展到 (batch,M,1,1) 与多专家编码结果相乘，最后对乘了门控的多专家编码结果
        进行sum得到一个(batch,seq_len,hidden_size)的混合专家网络输出
        """
        moe_output = self.moe_out_drop(moe_output)
        moe_output = self.LayerNorm(moe_output + seq_embedding)
        # ------ calculate output gate ------
        output_dense0 = self.output_gate_proj0(moe_output)
        output_dense1 = self.output_gate_proj1(moe_output)
        output_gate = self.Gelu(output_dense0)
        output_gated = output_gate * output_dense1
        output_tensor = self.final_output_drop(self.output_gate_projF(output_gated))
        output_tensor = self.LayerNorm(output_tensor + moe_output)
        seq_output = self.gather_indexes(output_tensor, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)  #(batch,hidden_size)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def fast_predict(self, interaction):
        """

        Args:
            interaction: (batch_size,) 一个batch大小的交互数据，包含了每个序列的负采样序列

        Returns:
            返回一个(batch_size,(neg_num+1))

        """
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_item_ids = interaction[self.ITEM_ID]  #(batch,)
        neg_item_ids = interaction[self.NEG_FIELD]  #(batch_size,neg_num)
        seq_output = self.forward(item_seq, item_seq_len)  #(batch,hidden_size)
        pos_item_embeds = self.item_embedding(pos_item_ids)  #(batch_size,hidden_size)
        neg_item_embeds = self.item_embedding(neg_item_ids)  #(batch_size,neg_num,hidden_size)
        pos_scores = torch.mul(seq_output, pos_item_embeds).sum(dim=1)
        neg_output = seq_output.unsqueeze(1).transpose(1, 2)  #(batch_size,hidden_size,1)
        neg_scores = torch.matmul(neg_item_embeds, neg_output).squeeze(2)  #(batch,neg_num)
        pos_scores = pos_scores.unsqueeze(1)  #(batch_size,1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)  #将正负样本的得分按列组合(batch,(neg_num+1))
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
