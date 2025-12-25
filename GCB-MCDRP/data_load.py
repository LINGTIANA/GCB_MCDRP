import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
import torch

def smiles_to_mol(smiles):
    """将SMILES字符串转化为RDKit的分子对象"""
    return Chem.MolFromSmiles(smiles)

def mol_to_graph(mol):
    """将RDKit的分子对象转化为图对象"""
    # 获取节点特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum(),  # 原子的元素符号（整数表示）
                              atom.GetDegree(),    # 原子的连接度
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
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 设置边特征（例如，化学键类型）
    edge_attr = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr.append([bond_type])

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 创建并返回一个图对象
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def dataload(**cfg):
    """加载数据并处理药物图"""
    # 提取从配置文件传入的各个数据文件路径
    response = cfg['path']['response']        # 药物反应数据文件路径
    mutation = cfg['path']['mutation']        # 突变数据文件路径
    methylation = cfg['path']['methylation']  # 甲基化数据文件路径
    expression = cfg['path']['expression']    # 基因表达数据文件路径
    pathway_file = cfg['path']['pathway']     # 信号通路数据文件路径
    drug_fpFile_morgan = cfg['path']['morgan'] # Morgan药物指纹数据文件路径
    drug_fpFile_espf = cfg['path']['espf']    # ESPF药物指纹数据文件路径
    drug_fpFile_psfp = cfg['path']['psfp']    # PubChem药物指纹数据文件路径
    smiles_file = cfg['path']['smiles']       # SMILES药物文件路径

    # 读取药物反应数据文件（cell_line-drug pairs）
    response = pd.read_csv(response, index_col=0)  # 将反应数据加载为 DataFrame，并将第一列作为行索引
    drug_key = response.columns.values  # 获取药物的列名（即药物ID）

    # 创建一个空列表，用于存储细胞系和药物的反应数据对
    pair = []
    # 遍历所有细胞系的反应数据
    for index, row in response.iterrows():
        # 遍历每个药物，检查该药物是否有有效的IC50值
        for i in drug_key:
            if np.isnan(row[i]) == False:
                pair.append([index, i, row[i]])

    # 加载细胞系的突变数据、基因表达数据和甲基化数据
    mut_feature = pd.read_csv(mutation, index_col=0)   # 读取突变数据
    exp_feature = pd.read_csv(expression, index_col=0)  # 读取基因表达数据
    methy_feature = pd.read_csv(methylation, index_col=0)  # 读取甲基化数据

    # 读取信号通路数据
    pathway = pd.read_csv(pathway_file, index_col=0)   # 读取信号通路数据

    # 使用pickle加载药物的分子指纹数据
    with open(drug_fpFile_morgan, 'rb') as f:
        morgan_fp = pickle.load(f)  # 加载Morgan药物指纹
    with open(drug_fpFile_espf, 'rb') as f:
        espf_fp = pickle.load(f)   # 加载ESPF药物指纹
    with open(drug_fpFile_psfp, 'rb') as f:
        pubchem_fp = pickle.load(f)  # 加载PubChem药物指纹

    # 创建一个字典用于存储每个药物的指纹特征
    drug_feature = {}
    for i in drug_key:
        drug_feature[i] = [morgan_fp[int(i)], espf_fp[int(i)], pubchem_fp[int(i)]]

    # 加载药物的SMILES并生成图数据
    smiles_df = pd.read_csv(smiles_file)
    drug_graphs = []
    invalid_smiles = []

    for smiles in smiles_df['SMILES']:
        mol = smiles_to_mol(smiles)
        if mol:
            drug_graphs.append(mol_to_graph(mol))
        else:
            invalid_smiles.append(smiles)

    print(f"Total number of drug graphs: {len(drug_graphs)}")
    print(f"Total number of invalid SMILES: {len(invalid_smiles)}")

    # 返回加载的数据
    return drug_feature, mut_feature, exp_feature, methy_feature, pathway, pair, response.index.values, response.columns.values, drug_graphs


