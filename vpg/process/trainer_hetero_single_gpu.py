import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime

sys.path.append('/home/jrj/postgraduate/vpg')

import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

from process.vpg_dataset import HeteroVPGDataset, N_FEATURES
from process.vpg_model import RGCN, RGAT_linear, RGCN_GAT, RSAGE, RSAGE_GAT
from utils.dataManager import check_dataset_cache, load, get_dataloaders, get_rel_names

import torch
from torch import nn
from tqdm import tqdm

os.environ["DGLBACKEND"] = "pytorch"


def evaluate(g, features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model(g, features)
        pred = torch.sigmoid(output).cpu().numpy()
        pred = np.where(pred >= 0.5, 1, 0)
        labels = labels.data.cpu().numpy()
        acc = accuracy_score(labels, pred)
        precision = precision_score(labels, pred, )
        recall = recall_score(labels, pred)
        f1 = f1_score(labels, pred)
        return acc, precision, recall, f1


def evaluate_in_batches(dataloader, model, device):
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata["feats"]
        labels = batched_graph.nodes["mapping_var"].data["labels"].unsqueeze(1).float()
        acc, precision, recall, f1 = evaluate(batched_graph, features, labels, model)
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    return total_acc / (batch_id + 1), total_precision / (batch_id + 1), total_recall / (batch_id + 1), total_f1 / (
            batch_id + 1)


def train(train_dataloader, val_dataloader, model, device, model_name, max_patience=2):
    loss_fcn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    patience_counter = 0
    best_f1 = 0
    best_model = None
    best_optimizer = None
    best_epoch = 0
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_id, batched_graph in enumerate(train_dataloader):
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata["feats"]
            labels = batched_graph.nodes["mapping_var"].data["labels"].unsqueeze(1).float()
            logits = model(batched_graph, features)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            "Epoch {:05d} | Loss {:.4f} |".format(
                epoch + 1, total_loss / (batch_id + 1)
            )
        )
        if (epoch + 1) % 5 == 0:
            avg_acc, avg_pre, avg_recall, avg_f1 = evaluate_in_batches(
                val_dataloader, model, device
            )  # evaluate F1-score instead of loss
            print(
                "                            Acc. {:.4f}, Pre. {:.4f}, Recall. {:.4f}, F1. {:.4f}".format(
                    avg_acc, avg_pre, avg_recall, avg_f1
                )
            )
            if avg_f1 > best_f1:
                patience_counter = 0
                best_model = deepcopy(model)
                best_optimizer = deepcopy(optimizer)
                best_f1 = avg_f1
                best_epoch = epoch
            else:
                patience_counter += 1
            if patience_counter == max_patience:
                break
    state = {'model': best_model.state_dict(), 'optimizer': best_optimizer.state_dict(), 'epoch': best_epoch}
    path = f"../data/model/{model_name}.bin"
    torch.save(state, path)
    return best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--gpu', '-g', type=str, required=True)
    parser.add_argument('--head', type=str, required=True)
    args = parser.parse_args()
    rank = int(args.gpu)
    base = "graph_v3"
    suffix = f"N{str(N_FEATURES)}" if N_FEATURES == 100 else ""
    cache = check_dataset_cache(base, suffix)
    dataset = []
    if not cache:
        dataset = load(f"/home/jrj/postgraduate/vpg/data/{base}{suffix}")
    dataset = HeteroVPGDataset(base, dataset, suffix)
    batch_size = 32
    seed = 100000
    etypes = get_rel_names(["CF", "DFB", "DFE", "CD", "DD", "FC"])
    in_size = N_FEATURES
    out_size = 1
    hid_size = 256
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"GPU Rank:{str(rank)}")
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    model_name = f"{args.model}_{str(N_FEATURES)}_{str(hid_size)}_{args.head}_{formatted_time}"
    print(model_name)
    if model_name.startswith("SAGE_GAT_"):
        model = RSAGE_GAT(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes, n_heads=2).to(
            device)
    elif model_name.startswith("SAGE_") and "GAT" not in model_name:
        model = RSAGE(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes).to(device)
    elif model_name.startswith("GCN_GAT_"):
        model = RGCN_GAT(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes, gcn_layers=5,
                         n_heads=int(args.head)).to(
            device)
    elif model_name.startswith("GCN") and "GAT" not in model_name:
        model = RGCN(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes).to(device)
    elif model_name.startswith("GAT"):
        model = RGAT_linear(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes, n_heads=2).to(
            device)
    else:
        raise Exception("Unexpected model!")
    print(model)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, seed, batch_size)
    best_model = train(train_loader, val_loader, model, device, model_name)
    avg_acc, avg_pre, avg_recall, avg_f1 = evaluate_in_batches(test_loader, best_model, device)
    print("Test Accuracy Acc. {:.4f}, Pre. {:.4f}, Recall. {:.4f}, F1. {:.4f}".format(
        avg_acc, avg_pre, avg_recall, avg_f1))
