import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

def vis_tsne(X, T, refs):
    # X: (N, C) 的样本特征矩阵
    # refs: (N,) 的样本标签
    # T: (4, C) 的 prototype 特征矩阵

    # Step 1: 归一化 prototype 特征矩阵
    T_normalized = normalize(T, axis=1)

    # Step 2: 合并样本特征和 prototype 特征
    data = np.vstack((X, T_normalized))

    # Step 3: 执行 T-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # Step 4: 分离降维后的样本和 prototype
    N = X.shape[0]  # 样本数量
    X_2d = data_2d[:N]  # 样本的降维结果
    T_2d = data_2d[N:]  # Prototype 的降维结果

    # Step 5: 可视化
    plt.figure(figsize=(10, 8))

    # 用不同颜色绘制不同类别的样本
    num_classes = len(np.unique(refs))
    colors = plt.cm.get_cmap('tab10', num_classes)
    colors = ['c', 'r', 'g', 'b', 'y']

    for class_id in range(num_classes):
        idx = refs == class_id
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f'Class {class_id}', alpha=0.6, s=4, c=colors[class_id])#[colors(class_id)]

        # 用特殊标记绘制 prototype
        plt.scatter(T_2d[class_id, 0], T_2d[class_id, 1], marker='*', alpha=0.6, s=15, color=colors[class_id])#[colors(class_id)]

    # 添加图例和标题
    plt.legend()
    plt.title('T-SNE Visualization of Samples and Prototypes')
    plt.savefig('./temp1.png')