import pandas as pd
import os

# data path
rootpath = os.path.dirname(os.path.abspath(__file__))
file_path  = rootpath + "/data/cid_smiles.csv"
df = pd.read_csv(file_path)

# 查看数据的前几行，确认格式
# print(df.head())

from rdkit import Chem
from torch_geometric.data import Data
import torch


# 将 SMILES 转换为 RDKit 分子对象
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)


# 将 RDKit 分子对象转化为图（Graph）
def mol_to_graph(mol):
    # 获取节点特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),  # 原子的元素符号（整数表示）
            atom.GetDegree(),  # 原子的连接度
            atom.GetFormalCharge(),  # 原子的电荷
            atom.GetHybridization(),  # 原子的杂化类型
            atom.GetNumImplicitHs(),  # 原子的隐式氢原子数
            atom.GetIsAromatic()  # 是否是芳香原子
        ])
    x = torch.tensor(atom_features, dtype=torch.float)  # 转换为张量

    # 获取边的信息（化学键）
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 由于是无向图，加入双向边
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 对于每条边，设置边特征（例如，化学键类型）
    edge_attr = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()  # 获取化学键类型
        edge_attr.append([bond_type])  # 可以根据需要添加更多边特征

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 转换为张量c

    # 创建一个图对象（Data）
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


invalid_smiles = []

# 生成图数据并保存到 drug_graphs 中
drug_graphs = []
for smiles in df['SMILES']:
    mol = smiles_to_mol(smiles)
    if mol:
        graph_data = mol_to_graph(mol)
        drug_graphs.append(graph_data)

# # 确保索引正确
# print(f"Total number of drug graphs: {len(drug_graphs)}")


# 查看未能成功解析的 SMILES
len(invalid_smiles), invalid_smiles[:10]  # 查看前 10 个无效 SMILES

# print(graph_data)

# 打印无效的 SMILES
# print(f"Total number of invalid SMILES: {len(invalid_smiles)}")
# print("Some invalid SMILES examples:", invalid_smiles[:10])  # 查看前 10 个无效 SMILES
