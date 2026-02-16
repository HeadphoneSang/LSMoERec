import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.modules import MLPExpertEncoder,LinearAttnExperEncoder, GRUExpertEncoder
from recbole.utils.encodeUtils import EventType, EventHandler


# @EventHandler(EventType.NOTICE_EVENT)
# def notice_mean_moe_gate(notice_dict, model):
#     notice_dict['moe_gate_avg'] = (
#             model.moe_records['moe_gate_avg'] / model.moe_records['accumulative_num']).cpu().tolist()
#     model.moe_records = None


class LSMoERec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(LSMoERec, self).__init__(config, dataset)

        # load parameters info
        self.moe_records = None
        self.hidden_size = config["hidden_size"]
        self.max_seq_len = config['MAX_ITEM_LIST_LENGTH']
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.n_heads = config["n_heads"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.moe_gate_t = config["moe_gate_t"]
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
        # shared expert
        self.shared_expert = MLPExpertEncoder(config)
        # self.shared_merge_dense = nn.Linear(self.hidden_size * 2, self.hidden_size)
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
        # 用于辅助损失的超参数
        self.mmd_lambda = config["mmd_lambda"]
        self.kernel_mul = config['kernel_mul']
        self.kernel_num = config['kernel_num']
        self.align_lambda = config["align_lambda"]

    # @EventHandler(EventType.NOTICE_EVENT)
    # def test_notification(self):
    #     print("success notification")

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

    def calculate_moe_gate_dense(self, filtered_resp):
        """
        calculate moe's information score by dense
        Args:
            filtered_resp: seq_embedding after filtering

        Returns: moe gate score
        """
        # (batch,M,seq_len) or (batch,seq_len)
        moe_gate = self.moe_proj0(filtered_resp).squeeze(-1)
        # (batch,M) or (batch,)
        moe_gate = self.moe_proj1(moe_gate).squeeze(-1)
        return moe_gate

    def expert_shared_FFN(self, expert_output, expert_hidden_gate, input_hidden_state):
        expert_output = expert_output * expert_hidden_gate
        expert_output = self.out_dense(expert_output)
        expert_output = self.moe_out_drop(expert_output)
        expert_output = self.LayerNorm(expert_output + input_hidden_state)
        return expert_output

    def output_hidden_filter(self, expert_output):
        """
        gating for each user_seq_hidden
        Args:
            expert_output: (batch,hidden_size)

        Returns: gated_output (batch,hidden_size)

        """
        left_dense_output = self.output_gate_proj0(expert_output)
        right_dense_output = self.output_gate_proj1(expert_output)
        output_hidden_gate = self.Gelu(left_dense_output)
        output_gated = output_hidden_gate * right_dense_output
        output_gated = self.output_gate_projF(output_gated)
        output_gated = self.final_output_drop(output_gated)
        return self.LayerNorm(output_gated + expert_output)

    def forward(self, item_seq, item_seq_len):
        """
        将所有专家的编码解耦的方法
        """
        # ----- embedding layer -----
        seq_embedding = self.item_embedding(item_seq)
        seq_embedding = self.emb_dropout(seq_embedding)
        moe_gates = []
        expert_hidden_gate = self.gate_dense_0(seq_embedding)
        # gate matrix for each expert's output
        expert_hidden_gate = self.Gelu(expert_hidden_gate)
        # MoE Encoding
        expert_last_res = []
        # (batch,hidden_size)
        item_seq_len -= 1
        batch_ids = torch.arange(item_seq.shape[0], device=item_seq.device)
        # (batch,1,hidden_size)
        # item_seq_len = item_seq_len.unsqueeze(-2)
        for expert in self.experts:
            # (batch,seq_len_hidden_size)
            filter_out = expert.filter_layer(seq_embedding)
            moe_gates.append(self.calculate_moe_gate_dense(filter_out))
            expert_output = expert(filter_out)
            expert_output = self.expert_shared_FFN(expert_output, expert_hidden_gate, seq_embedding)
            # (batch,hidden_size)
            last_expert_output = expert_output[batch_ids, item_seq_len]
            expert_last_res.append(last_expert_output)
        # MoE Gating and aggregation
        # (batch,M)
        moe_gates = torch.stack(moe_gates, dim=1)
        moe_gates = torch.softmax(moe_gates / (self.moe_gate_t + self.layer_norm_eps), dim=1)
        moe_gates = moe_gates.unsqueeze(-1)
        # (batch,M,hidden_size)
        expert_last_res = torch.stack(expert_last_res, dim=1)
        moe_output = moe_gates * expert_last_res
        # (batch,hidden_size)
        moe_output = moe_output.sum(dim=1)
        # shared_expert encoding
        shared_expert_output = self.shared_expert.filter_layer(seq_embedding)
        shared_expert_output = self.shared_expert(shared_expert_output)
        # (batch,hidden_size)
        shared_expert_output = shared_expert_output[batch_ids, item_seq_len]
        moe_output = moe_output + shared_expert_output
        # moe_output = self.shared_merge_dense(torch.cat([shared_expert_output, moe_output],dim=1))
        return self.output_hidden_filter(moe_output), moe_gates.squeeze(-1), expert_last_res

    def calculate_bal_loss(self, moe_gate):
        """
        calculate the balance loss for MoE's Gate
        Args:
            moe_gate: (batch,M) MoE gate vector for every sequence
        Returns:

        """
        E_gate = 1 / torch.tensor(len(self.experts), device=moe_gate.device)
        mean_gate = torch.mean(moe_gate, dim=0)  #(M,)
        return torch.sum(torch.pow(mean_gate - E_gate, 2), dim=0)

    def calculate_MMD(self, sample_a, sample_b, fix_bw=None):
        """
        calculate mmd from sample_a and sample_b
        Args:
            sample_a: (batch,hidden_size)
            sample_b: (batch,hidden_size)

        Returns: loss_item
        """
        batch_size = sample_a.shape[0]
        # distance matrices
        XX = torch.cdist(sample_a, sample_a, p=2) ** 2
        YY = torch.cdist(sample_b, sample_b, p=2) ** 2
        XY = torch.cdist(sample_a, sample_b, p=2) ** 2

        # remove diagonal for unbiased estimate
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=sample_a.device)
        XX = XX[mask]
        YY = YY[mask]
        XY = XY.view(-1)
        # adaptive bandwidth
        # mean bandwidth
        # bandwidth = ((XX.sum() + YY.sum() + XY.sum()) / (XX.numel() + YY.numel() + XY.numel())).detach()
        #median bandwidth
        if fix_bw:
            bandwidth = fix_bw
        else:
            bandwidth = torch.median(torch.cat([XX, YY, XY])).detach()
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        # multi-kernel RBF, mean over batch directly to save memory
        XX = sum(torch.exp(-XX / bw).mean() for bw in bandwidth_list)
        YY = sum(torch.exp(-YY / bw).mean() for bw in bandwidth_list)
        XY = sum(torch.exp(-XY / bw).mean() for bw in bandwidth_list)

        loss_item = XX + YY - 2 * XY
        return loss_item

    def calculate_semantic_ali_loss(self, expert_last_res, pos_items):
        """
        calculate the align loss of MoE
        Args:
            expert_last_res: (M,batch_size,hidden_size)
            pos_items: (batch)
        Returns: loss_item
        """
        test_item_emb = self.item_embedding.weight  #(item_n,hidden_size)
        M = expert_last_res.shape[0]
        # (M,batch,item_n)
        logits = torch.matmul(expert_last_res, test_item_emb.transpose(0, 1))
        logits = logits.view(-1, logits.shape[-1])
        pos_items = pos_items.unsqueeze(0).expand(M, -1).reshape(-1)
        loss = self.loss_fct(logits, pos_items)
        return loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, moe_gate, expert_last_res = self.forward(item_seq, item_seq_len)  #(batch,hidden_size),(batch,M)
        # bal_loss = self.calculate_bal_loss(moe_gate)
        expert_last_res = expert_last_res.permute(1, 0, 2)  # (M,batch,hidden_size)

        # record moe_gate_avg
        if self.moe_records is None:
            self.moe_records = {
                'moe_gate_avg': torch.mean(moe_gate, dim=0, keepdim=False).detach(),
                'accumulative_num': 1
            }
        else:
            self.moe_records['moe_gate_avg'] = self.moe_records['moe_gate_avg'] + torch.mean(moe_gate, dim=0,
                                                                                             keepdim=False).detach()
            self.moe_records['accumulative_num'] += 1
        mmd_loss = self.mmd_lambda * self.calculate_MMD(expert_last_res[0], expert_last_res[1])
        pos_items = interaction[self.POS_ITEM_ID]
        semantic_ali_loss = self.align_lambda * self.calculate_semantic_ali_loss(expert_last_res, pos_items)
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss, semantic_ali_loss, mmd_loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss, semantic_ali_loss, mmd_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output, _, _ = self.forward(item_seq, item_seq_len)
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
        seq_output, _, _ = self.forward(item_seq, item_seq_len)  #(batch,hidden_size)
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
        seq_output, _, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
