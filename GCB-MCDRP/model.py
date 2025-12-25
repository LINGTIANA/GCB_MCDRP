import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch_geometric.data import Data
from BAN import BANLayer  # 自定义的BAN层，用于药物与细胞特征交互
from torch_geometric.nn import GCNConv, global_mean_pool  # GCN卷积和全局池化

# 特征扰动：对节点特征添加噪声
def augment_graph(graph, noise_level=0.1):
    """
    向节点特征中添加高斯噪声以增强模型的鲁棒性。
    :param graph: 图数据对象，包含节点特征 graph.x
    :param noise_level: 噪声的标准差，决定噪声的大小
    :return: 添加了噪声后的图
    """
    device = graph.x.device  # 获取节点特征的设备
    noise = torch.randn_like(graph.x) * noise_level  # 在原始特征上添加噪声
    noisy_features = graph.x + noise  # 将噪声加到原始特征上

    # 返回扰动后的图
    graph.x = noisy_features
    return graph


from torch_geometric.nn import ChebConv

class DrugGraphEmbeddingChebNet(nn.Module):
    def __init__(self, in_channels, out_dim, K=3):
        super(DrugGraphEmbeddingChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, 128, K)
        self.conv2 = ChebConv(128, 256, K)
        self.conv3 = ChebConv(256, 512, K)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, graph.batch)
        x = self.fc(x)
        return x

def info_nce_loss(z1, z2, temperature=0.5):
    # Normalize embeddings to unit vectors
    z1 = F.normalize(z1, dim=1)  # Normalize z1 along the feature dimension
    z2 = F.normalize(z2, dim=1)  # Normalize z2 along the feature dimension

    # Compute similarity matrix (cosine similarity)
    similarity_matrix = torch.matmul(z1, z2.T)  # (batch_size, batch_size)

    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature

    # Create labels (positive samples on the diagonal)
    labels = torch.arange(z1.size(0)).to(z1.device)  # (batch_size)

    # Compute the loss using cross-entropy
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

# 主模型，支持对比学习模式
class GCBMCDRP(nn.Module):
    def __init__(self, cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, **config):
        super(GCBMCDRP, self).__init__()

        # 读取配置的超参数
        drug_out_dim = config['drug']['drug_out_dim']
        cell_out_dim = config['cell']['cell_out_dim']
        ban_heads = config['ban']['ban_heads']
        mlp_in_dim = config['mlp']['mlp_in_dim']
        mlp_hidden_dim = config['mlp']['mlp_hidden_dim']

        # 药物图嵌入层
        self.drug_embedding = DrugGraphEmbeddingChebNet(in_channels=6, out_dim=drug_out_dim)
        # 细胞特征嵌入层
        self.cell_embedding = CellEmbedding(cell_exp_dim, cell_mut_dim, cell_meth_dim, cell_path_dim, cell_out_dim,
                                            use_exp=True, use_mut=True, use_meth=True, use_path=True)

        # BAN模块，融合药物与细胞信息
        self.ban = weight_norm(
            BANLayer(v_dim=drug_out_dim, q_dim=cell_out_dim, h_dim=mlp_in_dim, h_out=ban_heads,
                     dropout=config['ban']['dropout_rate']), name='h_mat', dim=None)

        # MLP预测层
        self.mlp = MLP(mlp_in_dim, mlp_hidden_dim, out_dim=1)

    def forward(self, drug_graph, cell_data, contrastive=False, view2_graph=None):
        # 对比学习模式：编码两个视图，返回对比学习的向量
        if contrastive and view2_graph is not None:
            z1 = self.drug_embedding(drug_graph)
            z2 = self.drug_embedding(view2_graph)
            return z1, z2

        # 正常模式：药物图和细胞数据嵌入，经过BAN和MLP，输出预测值和注意力
        v_d = self.drug_embedding(drug_graph)
        v_c = self.cell_embedding(*cell_data)
        f, att = self.ban(v_d, v_c)
        predict = self.mlp(f)
        predict = torch.squeeze(predict)
        return predict, att


# 药物图嵌入层，使用两层GCN和全局池化
class DrugGraphEmbedding(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(DrugGraphEmbedding, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = nn.Linear(128, out_dim)

    def forward(self, drug_graph):
        x, edge_index = drug_graph.x, drug_graph.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, drug_graph.batch)
        x = self.fc(x)
        return x


# 细胞特征嵌入层，融合多种组学信息
class CellEmbedding(nn.Module):
    def __init__(self, exp_in_dim, mut_in_dim, meth_in_dim, path_in_dim, out_dim,
                 use_exp=True, use_mut=True, use_meth=True, use_path=True):
        super(CellEmbedding, self).__init__()
        self.use_exp = use_exp
        self.use_mut = use_mut
        self.use_meth = use_meth
        self.use_path = use_path

        self.gexp_fc1 = nn.Linear(exp_in_dim, 256)
        self.gexp_bn = nn.BatchNorm1d(256)
        self.gexp_fc2 = nn.Linear(256, out_dim)

        self.mut_fc1 = nn.Linear(mut_in_dim, 256)
        self.mut_bn = nn.BatchNorm1d(256)
        self.mut_fc2 = nn.Linear(256, out_dim)

        self.methylation_fc1 = nn.Linear(meth_in_dim, 256)
        self.methylation_bn = nn.BatchNorm1d(256)
        self.methylation_fc2 = nn.Linear(256, out_dim)

        self.pathway_fc1 = nn.Linear(path_in_dim, 256)
        self.pathway_bn = nn.BatchNorm1d(256)
        self.pathway_fc2 = nn.Linear(256, out_dim)

    def forward(self, expression_data, mutation_data, methylation_data, pathway_data):
        x_cell = []
        if self.use_exp:
            x_exp = self.gexp_fc1(expression_data)
            x_exp = F.relu(self.gexp_bn(x_exp))
            x_exp = F.relu(self.gexp_fc2(x_exp))
            x_cell.append(x_exp)

        if self.use_mut:
            x_mut = self.mut_fc1(mutation_data)
            x_mut = F.relu(self.mut_bn(x_mut))
            x_mut = F.relu(self.mut_fc2(x_mut))
            x_cell.append(x_mut)

        if self.use_meth:
            x_meth = self.methylation_fc1(methylation_data)
            x_meth = F.relu(self.methylation_bn(x_meth))
            x_meth = F.relu(self.methylation_fc2(x_meth))
            x_cell.append(x_meth)

        if self.use_path:
            x_path = self.pathway_fc1(pathway_data)
            x_path = F.relu(self.pathway_bn(x_path))
            x_path = F.relu(self.pathway_fc2(x_path))
            x_cell.append(x_path)

        # 堆叠所有细胞嵌入，shape: (batch_size, 特征类型数, out_dim)
        x_cell = torch.stack(x_cell, dim=1)
        return x_cell


# 多层感知机，用于最后预测
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.hidden = nn.Sequential(*layers)
        self.fc2 = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.hidden(x)
        x = self.fc2(x)
        return x
