import torch

# 创建两个形状为 (2, 3) 的张量
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

print("原始张量 a:")
print(a)
print("形状:", a.shape)
print("\n原始张量 b:")
print(b)
print("形状:", b.shape)

# dim=0: 在第0维堆叠（最外层维度）
result_dim0 = torch.stack([a, b], dim=0)
print("\n" + "="*50)
print("torch.stack([a, b], dim=0) 结果:")
print(result_dim0)
print("形状:", result_dim0.shape)
print("解释: 在最外层添加新维度，变成 (2, 2, 3)")

# dim=1: 在第1维堆叠（中间维度）  
result_dim1 = torch.stack([a, b], dim=1)
print("\n" + "="*50)
print("torch.stack([a, b], dim=1) 结果:")
print(result_dim1)
print("形状:", result_dim1.shape)
print("解释: 在中间维度插入，变成 (2, 2, 3)")

# dim=2: 在第2维堆叠（最内层维度）
result_dim2 = torch.stack([a, b], dim=2)
print("\n" + "="*50)
print("torch.stack([a, b], dim=2) 结果:")
print(result_dim2)
print("形状:", result_dim2.shape)
print("解释: 在最内层维度堆叠，变成 (2, 3, 2)")

# 验证不同dim的结果确实不同
print("\n" + "="*50)
print("验证三个结果互不相同:")
print("dim=0 和 dim=1 相同吗?", torch.equal(result_dim0, result_dim1))
print("dim=0 和 dim=2 相同吗?", torch.equal(result_dim0, result_dim2))
print("dim=1 和 dim=2 相同吗?", torch.equal(result_dim1, result_dim2))

# 实际应用场景示例
print("\n" + "="*50)
print("实际应用示例:")

# 假设这是两个时间步的特征
time_step_1 = torch.randn(32, 128)  # batch_size=32, feature_dim=128
time_step_2 = torch.randn(32, 128)

# 按时间维度堆叠
sequence_data = torch.stack([time_step_1, time_step_2], dim=1)
print(f"时间序列数据形状: {sequence_data.shape}")  # (32, 2, 128)

# 如果要在batch维度堆叠不同的样本
sample_1 = torch.randn(10, 128)  # 10个特征向量
sample_2 = torch.randn(10, 128)
batch_data = torch.stack([sample_1, sample_2], dim=0)
print(f"批次数据形状: {batch_data.shape}")  # (2, 10, 128)