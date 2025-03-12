import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def load_data(file_path):
    print("正在读取...")

    data = pd.read_csv(file_path, header=None, delimiter=',')

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    print(f"数据读取完成，共有 {X.shape[0]} 个样本， {X.shape[1]} 个特征")
    return X, y, feature_names


def analyze_features(X, y, feature_names):
    print("正在分析特征相关性...")

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    correlation_with_target = df.corr()['target'].sort_values(ascending=False)
    print("\n与目标变量相关性最高的20个特征")
    print(correlation_with_target[1:21])

    correlation_matrix = df.corr()

    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300)

    top_features = correlation_with_target[1:21].index.tolist()
    top_features.append('target')

    plt.figure(figsize=(14, 12))
    sns.heatmap(df[top_features].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Top 20 Features Correlation with Target")
    plt.tight_layout()
    plt.savefig("top_features_heatmap.png", dpi=300)

    top_feature_names = correlation_with_target[1:21].index.tolist()
    top_feature_indices = [feature_names.index(feat) for feat in top_feature_names]

    return top_feature_names, top_feature_indices

def split_and_save_data(X, y, top_features_indices=None, test_size=0.2, random_state=42):
    print("正在划分数据集")

    if top_features_indices is not None:
        print(f"只使用相关性最高的 {len(top_features_indices)} 个特征")
        X = X[:, top_features_indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")


    unique_values = np.unique(y)
    if len(unique_values) > 20:
        print(f"Target has {len(unique_values)} unique values, binning into classes...")

        bins = pd.qcut(pd.Series(y), q=10, retbins=True)[1]

        y_train_binned = pd.cut(pd.Series(y_train), bins=bins, labels=False, include_lowest=True)
        y_test_binned = pd.cut(pd.Series(y_test), bins=bins, labels=False, include_lowest=True)

        y_train = y_train_binned
        y_test = y_test_binned

        print(f"Data has been binned into {len(np.unique(y_train))} classes")

    #os.makedirs("data", exist_ok=True)

    with open("test_datasets/train_data.dat", "wb") as f:
        pickle.dump((X_train, y_train), f)

    with open("test_datasets/test_data.dat", "wb") as f:
        pickle.dump((X_test, y_test), f)

    print("数据集已保存为.dat文件")
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train, X_test, y_test, max_depths=[10, 15, 20]):
    print("Training decision tree classification models...")
    results = []

    print("Converting target variable to classes if needed...")

    unique_values = np.unique(y_train)
    print(f"Training with {len(unique_values)} distinct classes")


    for depth in max_depths:
        print(f"\nTraining tree with max depth {depth} ...")
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_pred)

        print(f"Training set Accuracy: {train_accuracy:.4f}")
        print(f"Testing set Accuracy: {test_accuracy:.4f}")

        model_path = f"test_models/decision_tree_depth_{depth}.sav"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        tree = model.tree_
        with open(f"test_models/tree_structure_depth_{depth}.txt", "w") as f:
            f.write(f"决策树 (max_depth={depth}) 结构摘要:\n")
            f.write(f"最大深度: {tree.max_depth}\n")
            f.write(f"节点总数: {tree.node_count}\n")
            f.write(f"叶子节点数: {tree.n_leaves}\n")
            f.write(f"内部节点数: {tree.node_count - tree.n_leaves}\n")


        results.append({
            'depth': depth,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'model_path': model_path,
            'node_count': tree.node_count,
            'leaf_count': tree.n_leaves
        })
 
    depths = [r['depth'] for r in results]
    train_accuracies = [r['train_accuracy'] for r in results]
    test_accuracies = [r['test_accuracy'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(depths, test_accuracies, 'o-', label='Testing Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Performance vs. Max Depth')
    plt.legend()
    plt.grid(True)
    plt.savefig("decision_tree_performance.png", dpi=300)
    
    # 绘制模型复杂度（节点数）与性能的关系
    node_counts = [r['node_count'] for r in results]
    leaf_counts = [r['leaf_count'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(depths, node_counts, 'o-', color='blue')
    plt.xlabel('Max Depth')
    plt.ylabel('Total Node Count')
    plt.title('Tree Size vs. Max Depth')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(node_counts, test_accuracies, 'o-', color='green')
    plt.xlabel('Total Node Count')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs. Tree Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("tree_complexity_analysis.png", dpi=300)


    # 找出表现最好的模型（基于测试集准确率）
    best_model = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest model depth: {best_model['depth']}")
    print(f"Best test accuracy: {best_model['test_accuracy']:.4f}")
    print(f"Tree size: {best_model['node_count']} nodes ({best_model['leaf_count']} leaves)")
    
    # 详细评估最佳模型
    with open(best_model['model_path'], "rb") as f:
        best_model_loaded = pickle.load(f)
    
    y_pred = best_model_loaded.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 如果类别数量合适，绘制混淆矩阵
    if len(unique_values) <= 15:  # 最多显示15个类别的混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png", dpi=300)
        print("Confusion matrix saved as confusion_matrix.png")
    
    return best_model


def analyze_feature_importance(model_path, feature_names):
    print("\n正在分析特征重要性...")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # 获取特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 显示所有选择的特征的重要性
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")  # 使用英文标题
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    
    print("\n所有特征的重要性排序:")
    for i in range(len(feature_names)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")


def main(file_path, max_dpeths=[5, 10, 15, 20]):
    X, y, feature_names = load_data(file_path)

    top_features_names, top_feature_indices = analyze_features(X, y, feature_names)

    X_train, X_test, y_train, y_test = split_and_save_data(X, y, top_feature_indices)

    best_model = train_decision_tree(X_train, y_train, X_test, y_test, max_depths)

    analyze_feature_importance(best_model['model_path'], top_features_names)

    print("\nAnalysis completed!")
    print(f"Feature correlation heatmap saved as correlation_heatmap.png")
    print(f"Top 20 features correlation heatmap saved as top_features_heatmap.png")
    print(f"Decision tree performance chart saved as decision_tree_performance.png")
    print(f"Feature importance chart saved as feature_importance.png")    



if __name__ == "__main__":
    file_path = "test_datasets/YearPredictionMSD.txt"

    max_depths = [5, 10, 15, 20]

    main(file_path, max_depths)
