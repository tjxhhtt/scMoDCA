import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.model import scMoDCA
from model.train import train_model
from model.util import setup_seed, SingleCellDataset, normalize_adata, get_label_names, to_tensor
from main_task import classification, dim_reduce
from sklearn.model_selection import StratifiedKFold
from config import *

def main_model(annData_1, annData_2, dataset):
    args = args_rc
    if dataset == 'CITE':
        args.hidden_2 = 50
    setup_seed(args.seed)
    cuda = True if torch.cuda.is_available() else False
    args.device = "cuda" if cuda else "cpu"
    args.dataset = dataset
    index = np.array(annData_1.obs['label'])
    train_label = pd.Categorical(index).codes
    adata1 = normalize_adata(annData_1)
    adata2 = normalize_adata(annData_2)

    classify_dim = max(train_label) + 1
    args.n_clusters = classify_dim
    nfeatures_1 = adata1.X.shape[1]
    nfeatures_2 = adata2.X.shape[1]
    train_data = np.concatenate((adata1.X, adata2.X), axis=1)
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    times = 1
    temp_1 = []
    temp_2 = []

    for train_index, test_index in k_fold.split(train_data, train_label):
        X_train, X_test = train_data[train_index], train_data[test_index]
        Y_train, Y_test = train_label[train_index], train_label[test_index]

        X_train, X_test, Y_train, Y_test = to_tensor(X_train, X_test, Y_train, Y_test)
        train_dataset = SingleCellDataset(X_train, Y_train)
        test_dataset = SingleCellDataset(X_test, Y_test)
        train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              drop_last=False)
        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             drop_last=False)


        print("The dataset is", args.dataset)
        model_save_path = "trained_model/{}/{}/".format(args.dataset, times)
        transform_real_label = get_label_names(index, classify_dim)

        model = scMoDCA(args=args, nfeatures_modality1=nfeatures_1, nfeatures_modality2=nfeatures_2, z_dim=args.z_dim,
                        classify_dim=classify_dim)

        model = model.cuda()
        model = train_model(args, model, train_dl, test_dl, lr=args.lr, epochs=args.epochs, dataset=dataset,
                            classify_dim=classify_dim, save_path=model_save_path)

        if args.classification:
            accuracy, f1, recall, precision = classification(args, model, test_dl, transform_real_label, times)
            print(f"Time {times} - MLP Acc: {accuracy:.4f}")

        if args.dim_reduce:
            metrics = dim_reduce(args, model, test_dl, transform_real_label, times, test_batch)
        temp_2.append(metrics)
        times = times + 1

    average_accuracy = np.mean(temp_1)
    average_metrics = np.mean(np.array(temp_2), axis=0)

    print("average metrics:")
    print(average_accuracy, average_metrics)
