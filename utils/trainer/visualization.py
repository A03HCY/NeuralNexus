import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    绘制混淆矩阵并返回 matplotlib figure 对象。
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_score, class_names=None):
    """
    绘制 ROC 曲线并返回 matplotlib figure 对象。
    支持二分类和多分类 (One-vs-Rest)。
    """
    n_classes = len(class_names) if class_names else (y_score.shape[1] if y_score.ndim > 1 else 2)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 二分类情况 (y_score shape [N, 1] or [N])
    if n_classes == 2 or (y_score.ndim == 1) or (y_score.shape[1] == 1):
        # 假设 y_score 是正类的概率
        # 如果 y_score 是 [N, 1]，squeeze
        if y_score.ndim == 2: y_score = y_score.squeeze()
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        # 多分类情况
        # 需要将 y_true 二值化
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # 计算每一类的 ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        # 绘制微平均 ROC 曲线 (Micro-average)
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        # 绘制每一类的 ROC 曲线
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(n_classes), colors):
            label = f'ROC curve of class {class_names[i] if class_names else i}'
            label += f' (area = {roc_auc[i]:.2f})'
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig
