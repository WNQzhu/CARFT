import torch

# 定义批量大小 B、维度 M、N 和 K
B = 2
M = 5
N = 3
K = 2

# 创建形状为 (B, M, N) 的随机张量 T1
T1 = torch.randn(B, M, N)
print("T1: ", T1)
print("原始张量 T1 形状:", T1.shape)

# 创建形状为 (B, K) 的索引张量 T2
T2 = torch.randint(0, M, (B, K))
print("T2: ", T2)
print("索引张量 T2 形状:", T2.shape)

# 创建批量索引
batch_indices = torch.arange(B).unsqueeze(1).expand(-1, K)
print("batch_indices: ", batch_indices)
print("batch_indices.shape: ", batch_indices.shape)


# 使用高级索引选取子张量
selected_sub_tensor = T1[batch_indices, T2]
print("selected_sub_tensor: ", selected_sub_tensor)
print("选取的子张量形状:", selected_sub_tensor.shape)    
