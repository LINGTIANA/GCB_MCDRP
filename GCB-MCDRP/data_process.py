import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data



class my_dataloader(Dataset):
    def __init__(self, drug_graphs, cell_expression, cell_mutation, cell_methylation, cell_pathway, pair, position):
        self.drug_graphs = drug_graphs  # 药物图数据
        self.cell_expression = cell_expression
        self.cell_mutation = cell_mutation
        self.cell_methylation = cell_methylation
        self.cell_pathway = cell_pathway
        self.pair = pair  # 药物和细胞系的IC50配对
        self.position = position  # 训练/验证/测试数据的索引位置

    def __len__(self):
        return len(self.position)

    def __getitem__(self, index):
        # 获取当前的药物-细胞系配对
        index = self.pair[self.position[index]]
        cell_idx, drug_idx, ic50_value = index

        # 获取药物的图数据
        drug_graph = self.drug_graphs[drug_idx]

        # 获取细胞系的特征
        expression = self.cell_expression[cell_idx]
        mutation = self.cell_mutation[cell_idx]
        methylation = self.cell_methylation[cell_idx]
        pathway = self.cell_pathway[cell_idx]

        # 返回药物图数据、细胞系特征和标签
        return drug_graph, expression, mutation, methylation, pathway, ic50_value

# 自定义 collate_fn 函数

def collate_fn(batch):
    # 分开图数据和其他特征（如细胞系特征和标签）
    drug_graphs = [item[0] for item in batch]  # 提取药物图数据

    # 处理细胞系特征，避免不必要的复制
    def to_tensor(item):
        if isinstance(item, torch.Tensor):
            return item.clone().detach()  # 如果已经是tensor，直接.detach()，避免复制
        return torch.tensor(item, dtype=torch.float).clone().detach()  # 否则转为tensor

    expressions = [to_tensor(item[1]) for item in batch]
    mutations = [to_tensor(item[2]) for item in batch]
    methylations = [to_tensor(item[3]) for item in batch]
    pathways = [to_tensor(item[4]) for item in batch]
    labels = [to_tensor(item[5]) for item in batch]

    # 使用 torch_geometric 的 Batch 来合并图数据
    batch_data = Batch.from_data_list(drug_graphs)

    # 将其他数据（细胞系特征和标签）作为元组返回
    return batch_data, expressions, mutations, methylations, pathways, labels


def data_process(drug_feature, mut_feature, exp_feature, methy_feature, pathway_feature, pair, cellline_id, drug_id, drug_graphs):
    """
    处理数据并返回训练、验证、测试集以及全部数据集的数据加载器。
    """
    # 对细胞系ID和药物ID进行排序
    cellline_id.sort()
    drug_id.sort()

    # 创建细胞系和药物的映射字典
    cell_map = list(zip(cellline_id, list(range(len(cellline_id)))))
    drug_map = list(zip(drug_id, list(range(len(drug_id)))))
    cell_dict = {i[0]: i[1] for i in cell_map}  # 细胞系ID映射到索引
    drug_dict = {i[0]: i[1] for i in drug_map}  # 药物ID映射到索引

    # 处理药物-细胞系对（药物和细胞系的IC50反应）
    all_pairs = []
    for i in pair:
        all_pairs.append([cell_dict[i[0]], drug_dict[i[1]], i[2]])

    # 处理药物特征数据
    drug_feature_num = len(drug_feature[drug_id[0]])  # 每个药物的特征数
    drug_feature_df = pd.DataFrame(index=drug_id, columns=list(range(drug_feature_num)))
    for index in drug_id:
        for j in range(drug_feature_num):
            drug_feature_df.loc[index, j] = drug_feature[index][j]

    # 将药物特征转换为torch张量
    drug_data = [torch.from_numpy(np.array(list(drug_feature_df.iloc[:, i]), dtype='float32')) for i in
                 range(drug_feature_num)]

    # 处理细胞系的基因表达、突变、甲基化和通路数据
    mutation = mut_feature.loc[cellline_id]
    expression = exp_feature.loc[cellline_id]
    methylation = methy_feature.loc[cellline_id]
    pathway = pathway_feature.loc[cellline_id]

    # 将细胞系数据转换为torch张量
    mutation = torch.from_numpy(np.array(mutation, dtype='float32'))
    expression = torch.from_numpy(np.array(expression, dtype='float32'))
    methylation = torch.from_numpy(np.array(methylation, dtype='float32'))
    pathway = torch.from_numpy(np.array(pathway, dtype='float32'))

    # 划分训练集、验证集和测试集
    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False}

    # 将数据集划分为训练集、验证集和测试集
    train_index, temp_index = train_test_split(range(len(pair)), test_size=0.2, random_state=42)
    val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=42)

    # 创建数据加载器
    train_set = DataLoader(
        my_dataloader(drug_graphs, expression, mutation, methylation, pathway, all_pairs, train_index),
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn  # 使用自定义的 collate_fn
    )

    test_set = DataLoader(
        my_dataloader(drug_graphs, expression, mutation, methylation, pathway, all_pairs, test_index),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn  # 使用自定义的 collate_fn
    )

    val_set = DataLoader(
        my_dataloader(drug_graphs, expression, mutation, methylation, pathway, all_pairs, val_index),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn  # 使用自定义的 collate_fn
    )

    # 创建完整数据集的数据加载器
    full_data_set = DataLoader(
        my_dataloader(drug_graphs, expression, mutation, methylation, pathway, all_pairs, range(len(pair))),
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn  # 使用自定义的 collate_fn
    )

    # 返回训练集、验证集、测试集和完整数据集的数据加载器
    return train_set, test_set, val_set, full_data_set

