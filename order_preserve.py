import torch
import numpy as np

# def order_preserving_transform(tensor):
#     # 获取排序后的值和排序后的索引
#     sorted_values, sorted_indices = torch.sort(tensor)

#     # 初始化一个张量来存储每个元素的排名
#     rank_tensor = torch.empty_like(tensor, dtype=torch.long)

#     # 使用布尔掩码避免循环
#     unique_values, inverse_indices = torch.unique(sorted_values, return_inverse=True)
#     rank_tensor[sorted_indices] = inverse_indices

#     return rank_tensor


# tensor = torch.tensor([40.5, 10, 30, 10, 50, 10, 10, 50, 40.5])
# rank = order_preserving_transform(tensor)
# print(rank)


def get_anchor_sign(compact_point):

    spatial_anchor_sign = (((compact_point[:, 0] % 2 == 0) & (compact_point[:, 1] % 2 == 0) & (compact_point[:, 2] % 2 == 0))|
                            ((compact_point[:, 0] % 2 == 0) & (compact_point[:, 1] % 2 == 1) & (compact_point[:, 2] % 2 == 1))|
                            ((compact_point[:, 0] % 2 == 1) & (compact_point[:, 1] % 2 == 0) & (compact_point[:, 2] % 2 == 1))|
                            ((compact_point[:, 0] % 2 == 1) & (compact_point[:, 1] % 2 == 1) & (compact_point[:, 2] % 2 == 0))
                            )
    
    return spatial_anchor_sign

# 目标数量
num_points = 160_000

# 创建一个比目标数量稍大的立方体空间，然后随机选择
range_limit = int(np.ceil(num_points ** (1/3))) + 10  # 立方根 + buffer
all_coords = np.array(np.meshgrid(
    np.arange(range_limit),
    np.arange(range_limit),
    np.arange(range_limit)
)).reshape(3, -1).T

# 打乱顺序并取前 num_points 个
np.random.shuffle(all_coords)
unique_coords = all_coords[:num_points]

spatial_anchor_sign = get_anchor_sign(unique_coords)

anchor = unique_coords[spatial_anchor_sign]
nonanchor = unique_coords[~spatial_anchor_sign]

print(anchor.shape[0])
print(nonanchor.shape[0])