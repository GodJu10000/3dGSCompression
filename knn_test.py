import faiss
import numpy as np
import torch
import time


# 生成示例数据
num_points = 500000  # 假设有100万个3D点
num_queries = 500000 # 1万个查询点
k = 5  # 取最近的5个邻居

# 生成随机整数3D点
# points = np.random.randint(0, 1000, size=(num_points, 3)).astype(np.float32)
# queries = np.random.randint(0, 1000, size=(num_queries, 3)).astype(np.float32)
points = torch.randint(0, 1000, (num_points, 3), dtype=torch.float32)
queries = torch.randint(0, 1000, (num_queries, 3), dtype=torch.float32)
points = points.cpu().numpy()
queries = queries.cpu().numpy()
# # 构建 FAISS 索引，并将其移到 GPU
torch.cuda.synchronize(); t0 = time.time()
index = faiss.IndexFlatL2(3)  # 3D 欧几里得距离
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)


# 添加点云数据
# torch.cuda.synchronize(); t0 = time.time()
gpu_index.add(points)
# torch.cuda.synchronize(); t_1 = time.time() - t0
# print("index time:", t_1)


# 查询最近邻
distances, indices = gpu_index.search(queries, k)
torch.cuda.synchronize(); t_1 = time.time() - t0
print("query time:", t_1)
print(type(indices))
print(indices.shape)


# IVF
# torch.cuda.synchronize(); t0 = time.time()
# dim, measure = 3, faiss.METRIC_L2 
# description =  'IVF4096,Flat'
# index = faiss.index_factory(dim, description, measure)
# res = faiss.StandardGpuResources()
# gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
# gpu_index.train(points)
# gpu_index.add(points)

# gpu_index.nprobe = 10
# distances, indices = gpu_index.search(queries, k)

# torch.cuda.synchronize(); t_1 = time.time() - t0
# print("query time:", t_1)

# HNSW
# torch.cuda.synchronize(); t0 = time.time()
# dim, max_nodes, measure = 3, 64, faiss.METRIC_L2   
# param =  'HNSW64' 
# # index = faiss.index_factory(dim, param, measure)
# index = faiss.IndexHNSWFlat(dim, max_nodes)
# res = faiss.StandardGpuResources()
# # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
# gpu_index = index
# gpu_index.add(points)
# gpu_index.hnsw_efSearch = 8

# distances, indices = gpu_index.search(queries, k)
# torch.cuda.synchronize(); t_1 = time.time() - t0
# print("query time:", t_1)
