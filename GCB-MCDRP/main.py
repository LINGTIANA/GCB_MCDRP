import os
import torch
from config import get_cfg_defaults
from data_load import *
from data_process import *
from model import *
from utils import *
# from drug import *
import torch.nn as nn
import pandas as pd

# 预测完整数据集的 IC50 值
def predict_full_data_set(model, full_data_set, depmap_id, drug_id, device, criterion):
    model.eval()
    predictions = []
    seen_pairs = set()  # 用来跟踪已经预测过的药物-细胞配对

    with torch.no_grad():
        # 获取所有数据集的药物图、细胞系特征和标签
        all_drug_graphs = [item[0] for item in full_data_set.dataset]
        all_expressions = [item[1] for item in full_data_set.dataset]
        all_mutations = [item[2] for item in full_data_set.dataset]
        all_methylations = [item[3] for item in full_data_set.dataset]
        all_pathways = [item[4] for item in full_data_set.dataset]
        all_labels = [torch.tensor(item[5], dtype=torch.float).to(device) for item in full_data_set.dataset]  # 转换label为Tensor

        # 将数据移到指定设备
        all_drug_graphs = [graph.to(device) for graph in all_drug_graphs]
        all_expressions = [exp.to(device) for exp in all_expressions]  # 直接将exp移到设备
        all_mutations = [mut.to(device) for mut in all_mutations]  # 同理
        all_methylations = [meth.to(device) for meth in all_methylations]
        all_pathways = [path.to(device) for path in all_pathways]

        # 确保所有数据的维度为2D（如果是1D则通过unsqueeze添加维度）
        all_expressions = [exp.unsqueeze(0) if exp.ndimension() == 1 else exp for exp in all_expressions]
        all_mutations = [mut.unsqueeze(0) if mut.ndimension() == 1 else mut for mut in all_mutations]
        all_methylations = [meth.unsqueeze(0) if meth.ndimension() == 1 else meth for meth in all_methylations]
        all_pathways = [path.unsqueeze(0) if path.ndimension() == 1 else path for path in all_pathways]

        # 预测
        for i in range(len(all_drug_graphs)):
            drug_graph = all_drug_graphs[i]
            exp = all_expressions[i]
            mut = all_mutations[i]
            meth = all_methylations[i]
            path = all_pathways[i]
            label = all_labels[i]

            # 检查 drug_graph 是否有边（避免空图）
            if drug_graph.edge_index.numel() == 0:  # 如果没有边，跳过该图
                # print(f"Skipping prediction for drug={drug_id[i % len(drug_id)]} because it has no edges.")
                continue  # 跳过该图

            # 正常预测
            predict, _ = model(drug_graph, [exp, mut, meth, path])

            # 获取 cell 和 drug 信息
            cell_idx, drug_idx, true_ic50 = full_data_set.dataset.pair[full_data_set.dataset.position[i]]
            cell_name = depmap_id[cell_idx]
            drug_name = drug_id[drug_idx]  # 使用 drug_idx 从 drug_id 中获取药物名称

            pair_key = (cell_name, drug_name)  # 用 (cell, drug) 作为唯一标识

            # 仅当该配对还未预测时进行预测
            if pair_key not in seen_pairs:
                # print(f"Predicting for cell={cell_name}, drug={drug_name}")
                predictions.append([cell_name, drug_name, true_ic50, predict.item()])
                seen_pairs.add(pair_key)  # 记录该配对已经预测过
            else:
                print(f"Skipped repeated prediction for cell={cell_name}, drug={drug_name}")

    return predictions

# 训练函数
def train(model, train_set, optimizer, myloss):
    device = next(model.parameters()).device
    model.train()
    predict_list = []
    label_list = []
    total_contrastive_loss = 0

    for batch, (drug_graph, exp, mut, meth, path, label) in enumerate(train_set):
        drug_graph = drug_graph.to(device)
        exp = torch.stack([e.to(device) for e in exp])
        mut = torch.stack([m.to(device) for m in mut])
        meth = torch.stack([m.to(device) for m in meth])
        path = torch.stack([p.to(device) for p in path])
        label = torch.stack([l.to(device) for l in label])

        # 生成药物图的增强视图，用于对比学习
        view2_graph = augment_graph(drug_graph)

        optimizer.zero_grad()
        # 正常预测
        predict, _ = model(drug_graph, [exp, mut, meth, path])
        # 对比学习药物嵌入
        z1, z2 = model(drug_graph, [exp, mut, meth, path], contrastive=True, view2_graph=view2_graph)
        cl_loss = info_nce_loss(z1, z2)

        # 总损失 = 预测损失 + 对比损失的加权和
        lambda_cl = 0.1  # 对比损失权重，可调节
        loss = myloss(predict, label) + lambda_cl * cl_loss
        loss.backward()
        optimizer.step()

        predict_list.extend(predict.detach().cpu().tolist())
        label_list.extend(label.detach().cpu().tolist())
        total_contrastive_loss += cl_loss.item()

    train_loss = myloss(torch.tensor(predict_list), torch.tensor(label_list)).item()
    avg_cl_loss = total_contrastive_loss / len(train_set)
    print(f"[Train] Prediction Loss: {train_loss:.4f}, Contrastive Loss: {avg_cl_loss:.4f}")
    return train_loss

# 测试函数
def test(model, test_set, myloss, depmap_id, drug_id):
    device = next(model.parameters()).device
    model.eval()
    predict_list = []
    label_list = []
    total_contrastive_loss = 0
    predictions = []

    seen_pairs = set()

    with torch.no_grad():
        for batch, (drug_graph, exp, mut, meth, path, label) in enumerate(test_set):
            drug_graph = drug_graph.to(device)
            exp = torch.stack([e.to(device) for e in exp])
            mut = torch.stack([m.to(device) for m in mut])
            meth = torch.stack([m.to(device) for m in meth])
            path = torch.stack([p.to(device) for p in path])
            label = torch.stack([l.to(device) for l in label])

            # 预测
            predict, _ = model(drug_graph, [exp, mut, meth, path])

            for i in range(len(predict)):
                cell_idx, drug_idx, true_ic50 = test_set.dataset.pair[test_set.dataset.position[batch]]
                cell_name = depmap_id[cell_idx]
                drug_name = drug_id[drug_idx]

                pair_key = (cell_name, drug_name)
                if pair_key not in seen_pairs:
                    predictions.append([cell_name, drug_name, true_ic50, predict[i].item()])
                    seen_pairs.add(pair_key)

            # 对比学习损失计算
            view2_graph = augment_graph(drug_graph)
            z1, z2 = model(drug_graph, [exp, mut, meth, path], contrastive=True, view2_graph=view2_graph)
            cl_loss = info_nce_loss(z1, z2)

            predict_list.extend(predict.detach().cpu().tolist())
            label_list.extend(label.detach().cpu().tolist())
            total_contrastive_loss += cl_loss.item()

    test_loss = myloss(torch.tensor(predict_list), torch.tensor(label_list)).item()
    avg_cl_loss = total_contrastive_loss / len(test_set)
    mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value = eval_predict(label_list, predict_list)
    print(f"[Test] Prediction MSE Loss: {test_loss:.4f}, Contrastive Loss: {avg_cl_loss:.4f}")

    return test_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value, avg_cl_loss, predictions

# 保存预测结果
def save_predictions(predictions, save_path):
    pred_df = pd.DataFrame(predictions, columns=["cell", "drug", "true_ic50", "predicted_ic50"])
    pred_df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")

def main():
    cfg = get_cfg_defaults()
    if not os.path.exists(cfg['path']['savedir']):
        os.makedirs(cfg['path']['savedir'])
    set_seed(2020)

    # 加载数据
    drug_feature, mut_feature, exp_feature, methy_feature, pathway_feature, pair, depmap_id, drug_id ,drug_graphs= dataload(**cfg)
    print(f"Loaded {len(drug_id)} drugs and {len(depmap_id)} cell lines. Total {len(pair)} pairs")

    device = torch.device(f'cuda:{cfg["model"]["cuda_id"]}' if torch.cuda.is_available() else "cpu")

    # 获取训练集、验证集和测试集
    train_set, test_set, val_set, full_data_set = data_process(drug_feature, mut_feature, exp_feature, methy_feature, pathway_feature,
                                                                pair, depmap_id, drug_id, drug_graphs)

    model = GCBMCDRP(cell_exp_dim=exp_feature.shape[-1], cell_mut_dim=mut_feature.shape[-1],
                   cell_meth_dim=methy_feature.shape[-1], cell_path_dim=pathway_feature.shape[-1], **cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['model']['lr'], weight_decay=cfg['model']['weight_decay'])
    criterion = nn.MSELoss()

    min_mae = float('inf')
    train_losses = []
    val_losses = []
    eval_results = []
    best_eval = []

    # 训练和验证
    for epoch in range(cfg['model']['epoch']):
        train_loss = train(model, train_set, optimizer, criterion)

        val_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value, cl_loss_val, val_predictions = test(
            model, val_set, criterion, depmap_id, drug_id)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        metrics = [train_loss, val_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value]
        row = [str(round(x, 6)) for x in metrics]
        row = [str(epoch)] + row
        print_table(row)
        eval_results.append(row)

        if mae < min_mae:
            min_mae = mae
            torch.save(model.state_dict(), os.path.join(cfg['path']['savedir'], 'model.pt'))
            print("save!")
            best_eval = row

    save_output(train_losses, val_losses, eval_results, cfg['path']['savedir'], "val", best_model_eval=best_eval)

    model.load_state_dict(torch.load(os.path.join(cfg['path']['savedir'], 'model.pt')))

    # 检查数据集的大小，查看有多少个细胞和药物
    print(f"Total number of drug graphs: {len(full_data_set.dataset.drug_graphs)}")
    print(f"Total number of drug cell: {len(full_data_set.dataset.cell_expression)}")
    print(f"Total number of pairs: {len(full_data_set.dataset.pair)}")

    # 对所有数据集进行预测
    predictions = predict_full_data_set(model, full_data_set, depmap_id, drug_id, device, criterion)

    print(f"Total number of predictions: {len(predictions)}")

    # 保存完整的预测结果
    save_predictions(predictions, os.path.join(cfg['path']['savedir'], "predicted_ic50_full.csv"))

    # 保存其他文件
    test_loss, mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value, avg_cl_loss, predictions = test(
        model, test_set, criterion, depmap_id, drug_id)
    metrics = [mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value]
    row = [str(round(x, 6)) for x in metrics]
    print_table(row, only_test=True)
    save_test_output(0, row, cfg['path']['savedir'], "test")
    draw_loss_curve(train_losses, val_losses, os.path.join(cfg['path']['savedir'], 'loss_curve.png'))

if __name__ == '__main__':
    main()
