import argparse
import os, sys
from copy import deepcopy
from datetime import datetime

from torch.nn import init
from tqdm import tqdm

sys.path.append('/home/jrj/postgraduate/vpg')
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn

from process.vpg_dataset import N_FEATURES, HeteroMetaDataset
from utils.dataManager import check_dataset_cache, load, get_dataloaders

os.environ["DGLBACKEND"] = "pytorch"


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def evaluate(features, labels, model):
    model.eval()
    with torch.no_grad():
        output = model(features)
        pred = torch.sigmoid(output).cpu().numpy()
        pred = np.where(pred >= 0.5, 1, 0)
        labels = labels.data.cpu().numpy()
        acc = accuracy_score(labels, pred)
        precision = precision_score(labels, pred, zero_division=0)
        recall = recall_score(labels, pred, zero_division=0)
        f1 = f1_score(labels, pred, zero_division=0)
        return acc, precision, recall, f1


def evaluate_in_batches(dataloader, model, device):
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    for batch_id, batched_graph in enumerate(dataloader):
        batched_graph = batched_graph.to(device)
        features = batched_graph.nodes["mapping_var"].data["feats"]
        labels = batched_graph.nodes["mapping_var"].data["labels"].unsqueeze(1).float()
        acc, precision, recall, f1 = evaluate(features, labels, model)
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    return total_acc / (batch_id + 1), total_precision / (batch_id + 1), total_recall / (batch_id + 1), total_f1 / (
            batch_id + 1)


def train(train_dataloader, val_dataloader, model, device, max_patience=2):
    loss_fcn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    patience_counter = 0
    best_acc = 0
    best_model = None
    best_optimizer = None
    best_epoch = 0
    for epoch in range(500):
        model.train()
        total_loss = 0
        for batch_id, batched_graph in enumerate(train_dataloader):
            batched_graph = batched_graph.to(device)
            features = batched_graph.nodes["mapping_var"].data["feats"]
            labels = batched_graph.nodes["mapping_var"].data["labels"].unsqueeze(1).float()
            logits = model(features)
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
            # if avg_acc > best_acc:
            #     patience_counter = 0
            #     best_model = deepcopy(model)
            #     best_acc = avg_acc
            #     best_epoch = epoch
            # else:
            #     patience_counter += 1
            # if patience_counter == max_patience:
            #     break
    return model


if __name__ == '__main__':
    rank = 3
    base = "graph_v3"
    suffix = f""
    # cache = check_dataset_cache(base, suffix)
    # dataset = []
    # if not cache:
    #     dataset = load(f"/home/jrj/postgraduate/vpg/data/{base}{suffix}")
    dataset = HeteroMetaDataset(base, [])
    batch_size = 256
    seed = 100000
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
    model_name = f"meta2vec"
    print(model_name)
    model = nn.Sequential(
        FlattenLayer(),
        nn.Linear(in_size, hid_size),
        nn.LeakyReLU(),
        nn.Linear(hid_size, hid_size),
        nn.Tanh(),
        nn.Linear(hid_size, hid_size),
        nn.ReLU(),
        # nn.Linear(hid_size, hid_size),
        # nn.LeakyReLU(),
        # nn.Linear(hid_size, hid_size),
        # nn.LeakyReLU(),
        nn.Linear(hid_size, out_size)
    ).to(device)
    for params in model.parameters():
        init.normal_(params, mean=0, std=0.01)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, seed, batch_size)
    best_model = train(train_loader, val_loader, model, device)
    avg_acc, avg_pre, avg_recall, avg_f1 = evaluate_in_batches(test_loader, best_model, device)
    print("Test Accuracy Acc. {:.4f}, Pre. {:.4f}, Recall. {:.4f}, F1. {:.4f}".format(
        avg_acc, avg_pre, avg_recall, avg_f1))
