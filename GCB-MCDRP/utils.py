import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def eval_predict(y_label, y_pred):
    # MAE MSE RMSE R^2
    mae = mean_absolute_error(y_label, y_pred)
    mse = mean_squared_error(y_label, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_label, y_pred)

    # pearson spearman
    pearson = pearsonr(y_label, y_pred)[0]
    pearson_p_value = pearsonr(y_label, y_pred)[1]
    spearman = spearmanr(y_label, y_pred)[0]
    spearman_p_value = spearmanr(y_label, y_pred)[1]

    return mse, rmse, mae, r2, pearson, pearson_p_value, spearman, spearman_p_value


import os
import numpy as np
import pandas as pd

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_output(train_loss, test_loss, eval_result, savedir, prefix, best_model_eval=None):
    """
    保存训练和测试损失以及评估结果的CSV文件，
    并保存最佳模型评估结果到txt，
    以及画训练测试损失曲线图（JPG格式）。

    参数:
    - train_loss: 训练损失列表
    - test_loss: 测试损失列表
    - eval_result: 评估结果列表，每个元素是多个指标的序列
    - savedir: 保存目录
    - prefix: 文件名前缀
    - best_model_eval: 最佳模型的评估结果（任意类型，可转字符串）
    """

    os.makedirs(savedir, exist_ok=True)

    # 列名
    columns = ['Epoch', 'Train Loss', 'Test Loss', 'MSE', 'RMSE', 'MAE', 'R2',
               'Pearson', 'Pearson p-value', 'Spearman', 'Spearman p-value',
               'Additional Metric 1', 'Additional Metric 2']

    data = []
    for epoch in range(len(train_loss)):
        row = [epoch, train_loss[epoch], test_loss[epoch]] + list(eval_result[epoch][1:])
        data.append(row)

    df = pd.DataFrame(data, columns=columns)

    # 保存CSV
    csv_file = os.path.join(savedir, f'{prefix}_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"保存结果到 {csv_file}")

    # 保存best_model.txt
    if best_model_eval is not None:
        best_model_file = os.path.join(savedir, 'best_model.txt')
        with open(best_model_file, 'w') as f:
            f.write("Best Model Evaluation Result:\n")
            f.write(str(best_model_eval) + '\n')
        print(f"保存最佳模型结果到 {best_model_file}")

    # 画训练测试损失曲线
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.grid(True)

    jpg_file = os.path.join(savedir, f'{prefix}_loss_curve.jpg')
    plt.savefig(jpg_file)
    plt.close()
    print(f"保存训练测试损失曲线到 {jpg_file}")


def save_test_output(test_loss, best_eval, savedir, prefix):
    os.makedirs(savedir, exist_ok=True)

    # 保存测试损失（.npy）
    np.save(os.path.join(savedir, f'{prefix}_test_loss.npy'), test_loss)
    print(f"保存测试损失到 {prefix}_test_loss.npy")

    # 保存最佳模型结果 txt
    best_model_file = os.path.join(savedir, f'best_model_{prefix}.txt')
    with open(best_model_file, 'w') as f:
        f.write("Best Model Evaluation Result:\n")
        f.write(str(best_eval) + '\n')
    print(f"保存最佳模型结果到 {best_model_file}")




def draw_loss_curve(train_loss, test_loss, savedir):
    plt.clf()
    plt.plot(np.arange(len(train_loss)), train_loss, label="train loss")
    plt.plot(np.arange(len(test_loss)), test_loss, label="test loss")
    plt.legend()  
    plt.xlabel('epoches')
    plt.title('Model loss')
    plt.show()
    plt.savefig(savedir ,
                dpi=330,
                facecolor='violet',
                edgecolor='lightgreen',
                bbox_inches='tight')
    return


def print_table(data, only_test=False, title=None, headers=None):
    table = PrettyTable()
    if title:
        table.title = title
    if not only_test:
        table.field_names = ['epoch', 'train_loss', 'test_loss', 'mse', 'rmse', 'mae', 'r2', 'pearson', 'pcc-p',
                             'spearman', 'scc-p']
    else:
        print('test_set')
        table.field_names = ['mse', 'rmse', 'mae', 'r2', 'pearson', 'pcc-p',
                             'spearman', 'scc-p']
    table.add_row(data)
    print(table)
    return table
