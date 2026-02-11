import torch


def calculate_MMD(sample_a, sample_b, kernel_mul=2.0, kernel_num=2, fix_bw=None):
    """
    calculate mmd from sample_a and sample_b
    Args:
        sample_a: (batch,hidden_size)
        sample_b: (batch,hidden_size)

    Returns: loss_item
    """
    batch_size = sample_a.shape[0]
    # sample_a = F.normalize(sample_a, dim=-1)
    # sample_b = F.normalize(sample_b, dim=-1)
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
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # multi-kernel RBF, mean over batch directly to save memory
    XX = sum(torch.exp(-XX / bw).mean() for bw in bandwidth_list)
    YY = sum(torch.exp(-YY / bw).mean() for bw in bandwidth_list)
    XY = sum(torch.exp(-XY / bw).mean() for bw in bandwidth_list)

    loss_item = XX + YY - 2 * XY
    return loss_item


if __name__ == '__main__':
    import torch

    # 设置随机种子保证可复现
    torch.manual_seed(42)

    # 样本参数
    batch_size = 64
    hidden_size = 128  # 特征维度

    # 高斯分布参数
    mean = 0.0
    std = 1.0

    # 生成样本 A 和 B，分布相同
    sample_a = torch.normal(mean=mean, std=std, size=(batch_size, hidden_size))
    sample_b = torch.normal(mean=mean, std=std, size=(batch_size, hidden_size))
    sample_b = sample_b + 10

    # optional: 打印均值方差确认
    print("Sample A mean:", sample_a.mean().item(), "std:", sample_a.std().item())
    print("Sample B mean:", sample_b.mean().item(), "std:", sample_b.std().item())
    print("MMD:", calculate_MMD(sample_a, sample_b))

