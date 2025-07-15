import torch
import numpy as np
from functools import reduce
# import MinkowskiEngine as ME

# # 创建 Minkowski 3D 卷积层，确保坐标不变
# conv_layer = ME.MinkowskiConvolution(
#     in_channels=16,   # 输入通道数
#     out_channels=32,  # 输出通道数
#     kernel_size=3,    # 3x3x3 卷积核
#     stride=1,         # 步长 1（不改变坐标）
#     dimension=3       # 3D 卷积
# )

# # 生成稀疏坐标和特征
# coords = torch.randint(0, 100, (1000, 4)).int()  # 1000 个点 (x, y, z, batch index)
# feats = torch.randn(1000, 16)  # 16 通道输入
# coords[:, 0] = 0
# # 创建 SparseTensor
# sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords)

# # 通过卷积层
# output_tensor = conv_layer(sparse_tensor)

# # 对比输入和输出坐标
# print("输入坐标:\n", sparse_tensor.C[:5])  # 打印前 5 个坐标
# print("输出坐标:\n", output_tensor.C[:5])  # 打印前 5 个坐标
# print("输出特征:\n", output_tensor.F[:5])
# assert torch.equal(sparse_tensor.C, output_tensor.C), "坐标发生了变化！"

# tensor = torch.zeros(76796)
# print(tensor.shape)  # torch.Size([76796])
# indices = np.random.randint(low=0, high=76796, size=(10, 2))
# print(indices.dtype)
# indices = tensor[indices]
# print(indices)

# print(indices.shape)  # (77087, 2)

# print("✅ 坐标保持不变！")

# N = 10
# M = 5
# x = torch.randn(N)
# # idx = torch.randint(0, N, (N, 2))
# idx = torch.randint(0, N, (M, 5))
# # tmp_idx = idx[[0, 2, 3]].numpy()
# x_new = x[idx]
# print(x)
# print(idx)
# print(x_new)

# # 创建一个 Nx3 的张量
# N = 5  # 假设 N=5
# tensor_Nx3 = torch.randn(N, 3)  # 随机生成 Nx3 的张量
# print("Nx3 Tensor:")
# print(tensor_Nx3)

# # 创建一个 Mx2 的行索引张量
# M = 3  # 假设 M=3
# indices_Mx2 = torch.randint(0, N, (M, 2))  # 随机生成 Mx2 的行索引张量
# print("\nMx2 Indices Tensor:")
# print(indices_Mx2)

# # 根据行索引选取 Mx2x3 的张量
# # 扩展维度到 Mx2x3，选取每行的 3 个元素
# selected_tensor = tensor_Nx3[indices_Mx2]  # 索引选择

# print("\nSelected Mx2x3 Tensor:")
# print(selected_tensor)

# # 创建一个布尔张量，全是 False
# mask = torch.zeros(5, dtype=torch.bool)

# # 设置两个位置为 True
# mask[1] = True
# mask[3] = True

# # 获取 True 的位置
# indices = mask.nonzero(as_tuple=True)[0]

# print(mask)       # tensor([False, True, False, True, False])
# print(indices)    # tensor([1, 3])
# print(indices[:5])
# print(mask.sum().item())

# remove_duplicates_list = []
# remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
# print(remove_duplicates)

refer = torch.tensor([0, 1, 2, 3, 4])

# query_1 = np.array([[2, 1],
#                     [4, 3]
# ])
query_2 = torch.tensor([[2, 1],
                        [4, 3]
])

# print(refer[query_1])
print(refer[query_2])