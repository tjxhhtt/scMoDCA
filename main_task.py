import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from model.predict import classify_cells
from model.util import setup_seed, get_encodings
from sklearn import metrics
from sklearn.cluster import KMeans
import scanpy as sc
import torch


def classification(args, model, dl, real_label, times):
    setup_seed(args.seed)
    if args.query:
        path = "query"
    else:
        path = "reference"
    if not os.path.exists('output/classification/{}/{}/{}/'.format(args.dataset, path, times)):
        os.makedirs('output/classification/{}/{}/{}/'.format(args.dataset, path, times))
    save_path = open('output/classification/{}/{}/{}/accuracy_each_cell.txt'.format(args.dataset, path, times), "w")
    classified_label, groundtruth_label, prob = classify_cells(model, dl, real_label)
    t = 0
    for j in range(len(groundtruth_label)):
        if classified_label[j] == groundtruth_label[j]:
            t = t + 1
        print('cell ID: ', j, '\t', '\t', 'real cell type:', groundtruth_label[j], '\t', '\t', 'predicted cell type:',
              classified_label[j], '\t', '\t', 'probability:', round(float(prob[j]), 2), file=save_path)

    accuracy = t / len(groundtruth_label)
    precision = precision_score(groundtruth_label, classified_label, average='macro')
    recall = recall_score(groundtruth_label, classified_label, average='macro')
    f1 = f1_score(groundtruth_label, classified_label, average='macro')

    print(f"--- 综合评估结果 ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision:  {precision:.4f}")
    print(f"Macro Recall:     {recall:.4f}")
    print(f"Macro F1-score:   {f1:.4f}")

    print("\n--- 详细分类报告 (Per-class Metrics) ---")
    report_dict = classification_report(groundtruth_label, classified_label, output_dict=True)
    print(classification_report(groundtruth_label, classified_label))
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv('output/classification/{}/{}/{}/classification_report.csv'.format(args.dataset, path, times),
                     index=True, index_label="Cell_Type")

    print("✅ 分类报告已成功保存为 classification_report.csv")
    print("finish classification", times)

    # plot_cm module not available — skipped confusion matrix plot
    return accuracy, f1, recall, precision


def dim_reduce(args, model, dl, real_label, times, test_batch=None):
    path_type = "query" if args.query else "reference"
    out_dir = f'output/dim_reduce/{args.dataset}/{path_type}/{times}/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    enc_dict = get_encodings(model, dl)
    latent_shared = enc_dict['shared']

    if isinstance(latent_shared, torch.Tensor):
        latent_np = latent_shared.detach().cpu().numpy()
        label_np = enc_dict['label'].detach().cpu().numpy()
    else:
        latent_np = latent_shared
        label_np = enc_dict['label']

    true_label_names = [real_label[int(l)] for l in label_np]

    num_samples = latent_np.shape[0]
    num_features = latent_np.shape[1]

    cell_name_real = [f'cell_{i}' for i in range(num_samples)]
    feature_index = [f'feature_{j}' for j in range(num_features)]

    pd.DataFrame(latent_np, index=cell_name_real, columns=feature_index).to_csv(
        os.path.join(out_dir, 'latent_space.csv')
    )

    df_labels = pd.DataFrame(true_label_names, index=cell_name_real, columns=["label"])
    if test_batch is not None:
        df_labels["batch"] = test_batch
    df_labels.to_csv(os.path.join(out_dir, 'latent_space_label.csv'))
    print(f"Features and labels saved strictly as original format in {out_dir}")

    print(f"Running K-means clustering with n_clusters={args.n_clusters}...")
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=30, random_state=args.seed)
    y_pred = kmeans.fit_predict(latent_np)

    ami = np.round(metrics.adjusted_mutual_info_score(label_np, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label_np, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label_np, y_pred), 5)
    print(f'Final Metrics (K-means) -> AMI: {ami:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')

    print("Generating Clustering Evaluation Tables...")
    crosstab_df = pd.crosstab(np.array(true_label_names), y_pred, rownames=['True Class'], colnames=['Cluster ID'])
    crosstab_df.to_csv(os.path.join(out_dir, 'kmeans_crosstab_1.csv'))

    df_purity = pd.DataFrame({'True_Label': true_label_names, 'Cluster_ID': y_pred})
    purity_list = []
    for cluster_id, group in df_purity.groupby('Cluster_ID'):
        total_size = len(group)
        dominant_class = group['True_Label'].value_counts().idxmax()
        dominant_count = group['True_Label'].value_counts().max()
        purity_list.append({
            'Cluster_ID': cluster_id, 'Size': total_size,
            'Dominant_Class': dominant_class, 'Purity': np.round(dominant_count / total_size, 4)
        })
    pd.DataFrame(purity_list).sort_values('Purity', ascending=False).to_csv(
        os.path.join(out_dir, 'kmeans_purity_1.csv'), index=False
    )

    adata = sc.AnnData(latent_np)
    adata.obs["Ground Truth"] = pd.Categorical(true_label_names)
    adata.obs["K-means Cluster"] = pd.Categorical(y_pred.astype(str))

    if test_batch is not None:
        adata.obs["Batch Effect"] = pd.Categorical(test_batch)

    sc.pp.neighbors(adata, use_rep='X', n_neighbors=30, metric='cosine')
    sc.tl.umap(adata, min_dist=0.3)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color='K-means Cluster', ax=ax, show=False,
               title=f"", palette='Set3', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'umap_kmeans_cluster.png'), dpi=300)
    plt.close()

    if "Batch Effect" in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata, color='Batch Effect', ax=ax, show=False,
                   title="", palette='Dark2', frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'umap_batch_effect.png'), dpi=300)
        plt.close()
        print("Batch Effect UMAP saved individually.")

    print("All UMAP graphs have been saved separately successfully!")
    print("finish dimension reduction")
    return [ami, nmi, ari]

