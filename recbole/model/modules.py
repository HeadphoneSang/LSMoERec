import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import MultiHeadAttention


class LinearMultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(LinearMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)  #row-wise
        self.softmax_col = nn.Softmax(dim=-2)  #column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor):
        """
        外部传来的tensor没有位置编码
        Args:
            input_tensor:

        Returns:

        """
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        # (batch,head_num,seq_len,hidden_size)
        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        # (batch,head_num,hidden_size,seq_len)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        # (batch,head_num,seq_len,hidden_size)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        # (L2 norm,计算每个向量的L2范数) (batch,head_num,seq_len)
        query_norm_inverse = 1 / torch.norm(elu_query, dim=3, p=2)
        # (L2 norm,计算每个向量的L2范数) (batch,head_num,seq_len)
        key_norm_inverse = 1 / torch.norm(elu_key, dim=2, p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij', elu_query, query_norm_inverse)
        # 将key_norm_inverse拓展成和elu_key一样形状的，让elu_key的每个向量都乘以对应的L2 norm进行标准化
        normalized_key_layer = torch.einsum('mnij,mnj->mnij', elu_key, key_norm_inverse)
        # (batch,head_num,seq_len,hidden_size)
        context_layer = torch.matmul(normalized_query_layer,
                                     torch.matmul(normalized_key_layer, value_layer)) / self.sqrt_attention_head_size
        # (batch,seq_len,head_num,hidden_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # hidden_states = context_layer
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    """
    The FFN module:
    x -> Linear mapping (out:hidden*4) -> ffn_act -> Liner mapping (out:hidden)
    -> dropout -> Add & Norm
    """

    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ffn_act = config['hidden_act']
        self.dense_1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        if self.ffn_act.lower() == 'gelu':
            self.intermediate_act_fn = nn.GELU()
        elif self.ffn_act.lower() == 'relu':
            self.intermediate_act_fn = nn.ReLU()
        else:
            raise ValueError(
                "{} is not supported as an activation function for FFN".format(self.ffn_act)
            )

        self.dense_2 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FrequencyAugExpert(nn.Module):
    """
    Frequency Expert:
    x -> Filter Layer -> Add & Norm -> FFN
    """

    def __init__(self, config):
        super(FrequencyAugExpert, self).__init__()
        self.max_seq_len = config['MAX_ITEM_LIST_LENGTH']
        self.hidden_size = config["hidden_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.complex_weight = nn.Parameter(
            torch.randn(1, self.max_seq_len // 2 + 1, self.hidden_size, 2, dtype=torch.float32) * 0.02)
        #-------------Layers for output---------------- -
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.ffn = Intermediate(config)

    def filter_layer(self, input_tensor):
        """
        对输入的向量进行傅里叶变换 -> 滤波 -> 逆傅里叶变换 -> dropout -> Add & Norm
        Args:
            input_tensor: (batch,seq_len,hidden_Size)

        Returns: 滤波后tensor

        """
        out_tensor = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        filter_mat = torch.view_as_complex(self.complex_weight)
        out_tensor = out_tensor * filter_mat
        out_tensor = torch.fft.irfft(out_tensor, n=self.max_seq_len, dim=1, norm='ortho')
        out_tensor = self.out_dropout(out_tensor)
        return self.LayerNorm(out_tensor + input_tensor)
        # return self.LayerNorm(out_tensor)


class AttnExperEncoder(FrequencyAugExpert):
    """
    Linear Attention Expert:
    x -> Filtered Attn Layer -> Add & Norm -> FFN
    """

    def __init__(self, config):
        super(AttnExperEncoder, self).__init__(config)
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.position_embedding = nn.Embedding(self.max_seq_len + 1, self.hidden_size)
        #-------------Layers for encode---------------- -
        self.attn_encoder = MultiHeadAttention(
            self.n_heads,
            self.hidden_size,
            self.hidden_dropout_prob,
            self.attn_dropout_prob,
            self.layer_norm_eps,
        )

    def forward(self, input_tensor, attention_mask):
        seq_len = input_tensor.shape[1]
        position_ids = torch.arange(seq_len, device=input_tensor.device)
        position_embedding = self.position_embedding(position_ids)  #(seq_len,hidden_size)
        position_embedding = position_embedding.unsqueeze(0)  #(1,seq_len,hidden_size)
        input_tensor = input_tensor + position_embedding
        # 创建注意力掩码矩阵
        attn_output = self.attn_encoder(input_tensor, attention_mask)
        return self.ffn(attn_output)


class LinearAttnExperEncoder(FrequencyAugExpert):
    """
    Linear Attention Expert:
    x -> Filtered Attn Layer -> Add & Norm -> FFN
    """

    def __init__(self, config):
        super(LinearAttnExperEncoder, self).__init__(config)
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.position_embedding = nn.Embedding(self.max_seq_len + 1, self.hidden_size)
        #-------------Layers for encode---------------- -
        self.attn_encoder = LinearMultiHeadAttention(
            self.n_heads,
            self.hidden_size,
            self.hidden_dropout_prob,
            self.attn_dropout_prob,
            self.layer_norm_eps,
        )

    def forward(self, input_tensor):
        seq_len = input_tensor.shape[1]
        position_ids = torch.arange(seq_len, device=input_tensor.device)
        position_embedding = self.position_embedding(position_ids)  #(seq_len,hidden_size)
        position_embedding = position_embedding.unsqueeze(0)  #(1,seq_len,hidden_size)
        input_tensor = input_tensor + position_embedding
        attn_output = self.attn_encoder(input_tensor)
        return self.ffn(attn_output)


class GRUExpertEncoder(FrequencyAugExpert):
    """

     """

    def __init__(self, config):
        super(GRUExpertEncoder, self).__init__(config)
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.time_b = config["time_b"]
        self.eps = config["layer_norm_eps"]
        # 时间嵌入
        self.time_embedding = nn.Embedding(64, self.hidden_size)
        # 用于处理GRU输入的一维卷积层
        self.in_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.kernel_size = config["conv_kernel_size"]
        self.left_pad_num = self.kernel_size - 1
        self.conv1d = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, padding=0)
        self.selective_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.3),
        )
        self.gru_layers = nn.GRU(
            input_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.gru_dense = nn.Linear(self.hidden_size, self.hidden_size)
        # 用于处理GRU输出的一维卷积层
        self.conv1dforgru = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, padding=0)

        self.gru_layernorm = nn.LayerNorm(self.hidden_size, eps=self.eps)

    def bukkit_time_gap(self, time_gap):
        gap = time_gap.clone()
        zero_mask = gap == 0
        inf = float("inf")
        gap[zero_mask] = inf
        min_gap = torch.min(gap, dim=1, keepdim=True)[0]
        min_gap[min_gap == inf] = 1
        gap[zero_mask] = 0
        bucket = torch.log1p(gap / (min_gap + self.eps)) / math.log(self.time_b)
        return torch.floor(bucket).long()

    def forward(self, input_tensor, time_list_seq):
        self.gru_layers.flatten_parameters()
        x = self.in_dense(input_tensor)

        # input casual conv1d
        x = x.transpose(1, 2)  #(batch,hidden_size,seq_len)
        x = F.pad(x, (self.left_pad_num, 0))  #(batch,hidden_size,seq_len+2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  #(batch,seq_len,hidden_size)

        # concat time_information
        time_gap_seq = time_list_seq[:, 1:] - time_list_seq[:, :-1]
        time_gap_seq = torch.clamp_min(time_gap_seq, 0)
        time_gap_seq = torch.cat([torch.zeros_like(time_gap_seq[:, :1]), time_gap_seq], dim=1)
        gap_bukkit_seq = self.bukkit_time_gap(time_gap_seq)
        time_embedding = self.time_embedding(gap_bukkit_seq)
        conv_input = torch.cat([x, time_embedding], dim=-1)  #(batch,seq_len,hidden_size*2)
        #---- calculate gate ----
        gate = self.selective_gate(conv_input)
        #---- GRU ----
        gru_output, _ = self.gru_layers(conv_input)
        gru_output = self.gru_dense(gru_output)
        res = gru_output
        G = gru_output * gate

        # res casual conv1d
        G = G.transpose(1, 2)
        G = F.pad(G, (self.left_pad_num, 0))
        G = self.conv1dforgru(G)
        G = G.transpose(1, 2)

        return self.ffn(self.gru_layernorm(G + res))

    # def forward(self, input_tensor):
    #     self.gru_layers.flatten_parameters()
    #     # x = self.in_dense(input_tensor)
    #     # x = self.conv1d(x.transpose(1, 2))
    #     # conv_input = x.transpose(1, 2)
    #     #---- calculate gate ----
    #     gate = self.selective_gate(input_tensor)
    #     #---- GRU ----
    #     gru_output, _ = self.gru_layers(input_tensor)
    #     # gru_output = self.gru_dense(gru_output)
    #     G = gru_output * gate
    #     # G = self.conv1dforgru(G.transpose(1, 2))
    #     # G = G.transpose(1, 2)
    #     return self.ffn(G)


class MLPExpertEncoder(FrequencyAugExpert):
    def __init__(self, config):
        super(MLPExpertEncoder, self).__init__(config)
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        # self.kernel_size = config["uaf_kernel_size"]
        # conv1d for frequency_input_embeddings
        # # UAF
        # self.freq_conv_encoder = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=self.hidden_size,
        #         out_channels=self.hidden_size,
        #         kernel_size=self.kernel_size,
        #         padding=self.kernel_size // 2,
        #     ),
        #     nn.BatchNorm1d(self.hidden_size),
        # )

        # def filter_layer(self, input_tensor):
        #     """
        #     对输入的向量进行傅里叶变换 -> 滤波 -> 逆傅里叶变换 -> dropout -> Add & Norm
        #     Args:
        #         input_tensor: (batch,seq_len,hidden_Size
        #     Returns: 滤波后tensor
        #     """
        # out_tensor = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        # filter_mat = torch.view_as_complex(self.complex_weight)
        # frequency filter
        #calculate user_adaptive filter
        # pure_fre_output = torch.abs(out_tensor)  # (batch,seq_len/2,hidden_size)
        # pure_fre_output = pure_fre_output.transpose(1, 2)
        # pure_fre_output = self.freq_conv_encoder(pure_fre_output)
        # user_adaptive_filter = torch.sigmoid(pure_fre_output).transpose(1, 2)
        # filter_mat = user_adaptive_filter * filter_mat
        #filter
        # out_tensor = out_tensor * filter_mat
        # out_tensor = torch.fft.irfft(out_tensor, n=self.max_seq_len, dim=1, norm='ortho')
        # out_tensor = self.out_dropout(out_tensor)
        # return self.LayerNorm(out_tensor + input_tensor)

    def forward(self, input_tensor):
        return self.ffn(input_tensor)


class UserAdaptiveEncoder(nn.Module):
    def __init__(self, config):
        super(UserAdaptiveEncoder, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.kernel_size = config["uaf_kernel_size"]
        self.max_seq_len = config["max_seq_len"]
        self.freq_conv_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm1d(self.hidden_size),
        )
        self.layer_norm_eps = config["layer_norm_eps"]
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, input_tensor):
        out_tensor = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        pure_fre_output = torch.abs(out_tensor)  # (batch,seq_len/2,hidden_size)
        pure_fre_output = pure_fre_output.transpose(1, 2)
        pure_fre_output = self.freq_conv_encoder(pure_fre_output)
        user_adaptive_filter = torch.sigmoid(pure_fre_output).transpose(1, 2)
        out_tensor = out_tensor * user_adaptive_filter
        out_tensor = torch.fft.irfft(out_tensor, n=self.max_seq_len, dim=1, norm='ortho')
        return self.layer_norm(out_tensor + input_tensor)
