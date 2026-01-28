import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.modules import MultiHeadAttention


class GLINTRU(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GLINTRU, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.n_heads = config["n_heads"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]

        # define efficient mlp layers
        # self.EfficientMLP = EfficientMLP(self.hidden_size)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.dense1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.dense2 = nn.Linear(self.embedding_size, self.hidden_size)
        # 用于处理GRU输入的一维卷积层
        self.conv1d = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidden_size, self.hidden_size)
        # 选择性门控机制，用于控制GRU输出的权重
        # 通过两层线性变换和激活函数构建门控信号，决定哪些信息需要保留
        # 第一层：将隐藏层大小压缩至一半，减少参数量
        # SiLU激活：引入非线性变换，增强模型表达能力
        # 第二层：恢复至原隐藏层大小，生成门控权重
        # Dropout：防止过拟合，提高模型泛化能力
        self.selective_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.Dropout(0.3),
        )
        """
        这里的哈达玛矩阵门控，里面的值取值范围没有变成概率分布，试一下变成概率分布，怎么样？
        """
        # 用于处理GRU输出的一维卷积层
        self.conv1dforgru = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        #线性注意力专家 (当序列长度小于隐藏维度的时候，是不是普通的注意力更快)
        self.linearattention = MultiHeadAttention(
            self.n_heads,
            self.hidden_size,
            self.hidden_dropout_prob,
            self.attn_dropout_prob,
            self.layer_norm_eps,
        )
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        self.dense3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.denseout = nn.Linear(self.hidden_size, self.embedding_size)
        self.dropdense = nn.Dropout(0.3)
        self.dropmix = nn.Dropout(0.3)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.gelu = nn.GELU()

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
        self.gru_layers.flatten_parameters()
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        #-------------------------First Layer-------------------------------
        attention_output = self.linearattention(item_seq_emb_dropout)
        h1 = self.dense1(item_seq_emb_dropout)  #用于GRU专家的线性变换
        h2 = self.dense2(item_seq_emb_dropout)  #用于计算混合专家最终输出的门控矩阵的线性变换
        h1 = self.conv1d(h1.transpose(1, 2))  # 用于GRU输入的一维卷积层
        h1 = h1.transpose(1, 2)
        h2 = self.gelu(h2)

        #-------------------------Mixed Temporal Block----------------------
        gru_output, _ = self.gru_layers(h1)  #(batch,seq_len,hidden_Size)
        selective_gate = self.selective_gate(h1)  #(batch,seq_len,hidden_Size)
        gru_output = self.projection(gru_output)  #混合管道映射，就是对GRU的输出进行一次线性变换，特征融合和特征提取
        gru_output = selective_gate * gru_output  # 门控矩阵
        gru_output = self.conv1dforgru(gru_output.transpose(1, 2))  # 用于GRU输出的一维卷积层，聚合每个时间的隐藏状态的上下文
        gru_output = gru_output.transpose(1, 2)

        weights = F.softmax(self.weights, dim=0)
        expert_output = weights[0] * gru_output + weights[1] * attention_output
        h = expert_output * h2  #这里乘这个门控矩阵，这个门控值会让正值变成*2，而负值变小
        h = self.dense(h)
        h = self.dropmix(h)
        h = self.LayerNorm(h + item_seq_emb_dropout)  # 这里的混合专家的输出h和输入的item_seq_emb不在同一个尺度啊，感觉会有问题，由于
        # the embedding of the predicted item, shape of (batch_size, embedding_size)

        x1 = self.dense3(h)
        x2 = self.dense4(h)
        x2 = self.gelu(x2)
        x = x1 * x2
        x = self.denseout(x)
        x = self.dropdense(x)
        x = self.LayerNorm(x + h)

        seq_output = self.gather_indexes(x, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
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
