# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self, kernel_mul, kernel_num):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul

    def forward(self, sample_a, sample_b, fix_bw=None):
        """
        calculate mmd from sample_a and sample_b
        Args:
            fix_bw: static bandwidth
            sample_a: (batch,hidden_size)
            sample_b: (batch,hidden_size
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
        # median bandwidth
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


class GateBalanceLoss(nn.Module):
    def __init__(self, gate_num):
        super(GateBalanceLoss, self).__init__()
        self.gate_num = gate_num

    def forward(self, moe_gate):
        """
        calculate the balance loss for MoE's Gate
        Args:
            moe_gate: (batch,M) MoE gate vector for every sequence
        Returns
        """
        E_gate = 1 / torch.tensor(self.gate_num, device=moe_gate.device)
        mean_gate = torch.mean(moe_gate, dim=0)  # (M,)
        return torch.sum(torch.pow(mean_gate - E_gate, 2), dim=0)


class ExpertsSemanticAlignLoss(nn.Module):
    def __init__(self, item_embedding):
        super(ExpertsSemanticAlignLoss, self).__init__()
        self.item_embedding = item_embedding
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, expert_last_res, pos_items):
        """
        calculate the align loss of MoE
        Args:
            expert_last_res: (M,batch_size,hidden_size)
            pos_items: (batch)
        Returns: loss_item
        """
        test_item_emb = self.item_embedding.weight  # (item_n,hidden_size)
        M = expert_last_res.shape[0]
        # (M,batch,item_n)
        logits = torch.matmul(expert_last_res, test_item_emb.transpose(0, 1))
        logits = logits.view(-1, logits.shape[-1])
        pos_items = pos_items.unsqueeze(0).expand(M, -1).reshape(-1)
        loss = self.loss_fct(logits, pos_items)
        return loss


class KLInfoNCE(nn.Module):
    """
    calculate InfoNCE for experts
    """

    def __init__(self, temp=0.5):
        super(KLInfoNCE, self).__init__()
        self.temp = temp

    def forward(self, experts_filters):
        """
        sum
        calculate experts division InfoNCE by KL function
        Args:
            experts_filters: (K,seq_len//2,hidden_size)

        Returns: loss_item

        """
        # (K,seq_len//2)
        K = experts_filters.shape[0]
        H = torch.mean(experts_filters, dim=-1)
        A = torch.softmax(H, dim=1)
        A = A + 1e-12
        log_A = torch.log(A)
        # (K,1)
        KL_0 = torch.sum(A * log_A, dim=1, keepdim=True)
        KL_1 = torch.matmul(A, log_A.T)
        # (K,K) KL matrix
        KL = KL_0 - KL_1
        mask = ~torch.eye(K, dtype=torch.bool, device=KL.device)
        masked_KL = KL[mask].view(K, -1)  #(K,K-1)
        log_KL = torch.log(masked_KL + 1)  # 防止单个KL对数值爆炸
        loss = 1.0 / torch.sum(log_KL, dim=1)
        loss = torch.mean(loss, dim=0)
        return loss

    # def forward(self, experts_filters):
    #     """
    #     info nce
    #     calculate experts division InfoNCE by KL function
    #     Args:
    #         experts_filters: (K,seq_len//2,hidden_size)
    #
    #     Returns: loss_item
    #
    #     """
    #     # (K,seq_len//2)
    #     H = torch.mean(experts_filters, dim=-1)
    #     A = torch.softmax(H, dim=1)
    #     A = A + 1e-12
    #     log_A = torch.log(A)
    #     # (K,1)
    #     KL_0 = torch.sum(A * log_A, dim=1, keepdim=True)
    #     KL_1 = torch.matmul(A, log_A.T)
    #     # (K,K)
    #     KL = KL_0 - KL_1
    #     KL_N = torch.exp(-1.0 * (KL / self.temp))
    #     self_mask = torch.eye(KL.shape[0], dtype=torch.bool, device=KL.device)
    #     # (K,)
    #     KL_N = torch.sum(torch.masked_fill(KL_N, self_mask, 0.), dim=1) + 1
    #     log_KL = -1.0 * torch.log(1.0 / KL_N).mean()
    #     return log_KL


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
