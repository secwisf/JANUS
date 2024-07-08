import sys

from tqdm import tqdm

sys.path.append('/home/jrj/postgraduate/vpg')

import os.path
from typing import List

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import split_dataset
from dgl.dataloading import GraphDataLoader

from prepare.graph import HeteroEdgeType
from process.vpg_dataset import HeteroVPGDataset, N_FEATURES, HeteroMetaDataset
from utils.sol_utils import all_path


def merge_dgl():
    g1 = dgl.heterograph({
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
        ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
    })
    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ('user', 'follows', 'topic'): ([], []),
        ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
    })
    g3 = dgl.batch([g1, g2])
    hg1 = dgl.heterograph({
        ('user', 'plays', 'game'): (torch.tensor([0, 1]), torch.tensor([0, 0]))})
    hg2 = dgl.heterograph({
        ('user', 'plays', 'game'): (torch.tensor([0, 0, 0]), torch.tensor([1, 0, 2]))})
    bhg = dgl.batch([hg1, hg2])
    print()


def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]


def load(path, ratio=1):
    files = all_path(path, [".pkl"])
    graph_data_files = sorted(list(filter(lambda x: x.split("/")[-1].startswith("graph_data_"), files)))
    graphs_data = []
    for index in range(len(graph_data_files)):
        graph = pd.read_pickle(graph_data_files[index])
        graph = list(filter(lambda x: len(x['edges']) * len(x['nodes_index']) * len(x['labels']) != 0, graph))
        graphs_data += graph
    if ratio < 1:
        graphs_data = get_ratio(graphs_data, ratio)
    return graphs_data


def check_dataset_cache(base, suffix=""):
    dataset_path = f"/home/jrj/postgraduate/vpg/data/{base}/HeteroVPGDataset{suffix}"
    return os.path.exists(os.path.join(dataset_path, "dgl_graph_list.bin"))


def get_dataloaders(dataset, seed, batch_size):
    train_set, val_set, test_set = split_dataset(
        dataset, frac_list=[0.8, 0.1, 0.1], shuffle=True, random_state=seed
    )
    train_loader = GraphDataLoader(train_set, batch_size=batch_size)
    val_loader = GraphDataLoader(val_set, batch_size=batch_size)
    test_loader = GraphDataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def print_statistics(dataset):
    total_nodes = 0
    total_edges = 0
    total_mapping_vars = 0
    true_labels = 0
    false_labels = 0
    for d in dataset:
        total_nodes += d.num_nodes()
        total_edges += d.num_edges()
        total_mapping_vars += d.num_nodes('mapping_var')
        labels = d.nodes["mapping_var"].data["labels"].unsqueeze(1).numpy()
        true_labels += np.count_nonzero(labels == 1)
        false_labels += np.count_nonzero(labels == 0)

    print(f"""
            avg nodes: {str(total_nodes / len(dataset))}\navg edges: {str(total_edges / len(dataset))}\n
            total_labels: {str(total_mapping_vars)}\n
            true_labels_per:{str(true_labels / total_mapping_vars * 100.0)}%\n
            false_labels_per:{str(false_labels / total_mapping_vars * 100.0)}%        
            """)


def get_rel_names(options: List[str]):
    rel_names = []
    # control flow
    if "CF" in options:
        rel_names += [HeteroEdgeType.DOMINATOR.value, HeteroEdgeType.SUCCESSOR.value]
    # data flow basic
    if "DFB" in options:
        rel_names += [HeteroEdgeType.WT.value, HeteroEdgeType.RF.value]
    # data flow extra
    if "DFE" in options:
        rel_names += [HeteroEdgeType.REFTO.value, HeteroEdgeType.REFED.value]
    # control dependence
    if "CD" in options:
        rel_names += [HeteroEdgeType.LB.value, HeteroEdgeType.LBN.value]
    # data dependence
    if "DD" in options:
        rel_names += [HeteroEdgeType.DDF.value, HeteroEdgeType.DDC.value]
    # fun call
    if "FC" in options:
        rel_names += [HeteroEdgeType.CALL.value, HeteroEdgeType.CALLED.value]
    return rel_names


def delete_nan():
    base = "graph_v3"
    dataset = HeteroMetaDataset(base, [])
    nan_num = 0
    new_graphs = []
    for graph in tqdm(dataset):
        features = graph.nodes['mapping_var'].data['feats']
        contains_nan = torch.isnan(features)
        all_false = torch.all(~contains_nan)
        if all_false.item():
            new_graphs.append(graph)
        else:
            nan_num += 1
    dataset.graphs = new_graphs
    dataset.save()
    print(nan_num)



if __name__ == '__main__':
    delete_nan()
    # dataset = HeteroVPGDataset(base, dataset, suffix=suffix)
    # print_statistics(dataset)
