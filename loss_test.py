from recbole.model.loss import KLInfoNCE
import torch
if __name__ == "__main__":
    # ============= 1. 生成测试样本 =============
    # 定义输入维度（模拟真实场景：K个专家，seq_len//2=8，hidden_size=16）
    K = 4  # 专家数量
    seq_half = 8  # seq_len//2
    hidden_size = 16  # 专家特征维度

    # 场景1：随机样本（模拟训练中的普通情况）
    torch.manual_seed(42)  # 固定随机种子，保证结果可复现
    random_experts = torch.randn(K, seq_half, hidden_size)  # 正态分布随机数

    # 场景2：极端样本1（所有专家分布完全相同，预期损失接近0）
    same_experts = torch.ones(K, seq_half, hidden_size) * 0.5  # 所有值相同，softmax后分布一致

    # 场景3：极端样本2（专家分布完全不同，预期损失较大）
    diff_experts = torch.zeros(K, seq_half, hidden_size)
    for i in range(K):
        diff_experts[i, i, :] = 10.0  # 每个专家只在第i列有大值，softmax后分布完全分离

    # ============= 2. 初始化损失函数 =============
    loss_fn = KLInfoNCE(temp=0.5)  # 温度系数设0.5（行业常用值）

    # ============= 3. 计算并验证损失 =============
    print("=" * 50)
    # 测试随机样本
    loss_random = loss_fn(random_experts)
    print(f"【随机样本】损失值：{loss_random.item():.4f}")

    # 测试所有专家分布相同的样本
    loss_same = loss_fn(same_experts)
    print(f"【所有专家分布相同】损失值：{loss_same.item():.4f}")

    # 测试所有专家分布完全不同的样本
    loss_diff = loss_fn(diff_experts)
    print(f"【所有专家分布完全不同】损失值：{loss_diff.item():.4f}")

    # ============= 4. 反向传播测试（验证梯度有效性） =============
    print("=" * 50)
    # 随机样本需要计算梯度，所以克隆并设requires_grad=True
    test_input = random_experts.clone().requires_grad_(True)
    loss = loss_fn(test_input)
    loss.backward()  # 反向传播

    # 检查梯度是否存在（非None且非全0）
    grad_norm = torch.norm(test_input.grad)
    print(f"反向传播验证：梯度范数 = {grad_norm.item():.4f}")
    if grad_norm > 0:
        print("✅ 梯度计算正常，可用于训练")
    else:
        print("❌ 梯度为0，损失函数可能存在问题")