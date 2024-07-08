import gc
import os
import pickle
import sys

sys.path.append('/home/jrj/postgraduate/vpg')
import numpy as np

from process.vpg_model import RGCN_GAT
from utils.dataManager import get_rel_names

import dgl
import torch
from rapidfuzz import fuzz
from tqdm import tqdm

from prepare.graph import SolidityInfo, VPG, HeteroEdgeType
from process.vpg_dataset import N_FEATURES
from utils.sol_utils import get_solc_version, is_mapping_type, is_address_type, is_number_type, all_path, \
    get_mapping_vars_for_train


def draw_dot():
    def str_node(node_tuple):
        flag = ""
        if node_tuple[-1] == 'function_node':
            flag = "F"
        elif node_tuple[-1] == 'mapping_var':
            flag = "M"
        elif node_tuple[-1] == 'other_state_var':
            flag = "S"
        elif node_tuple[-1] == 'local_var':
            flag = "L"
        return f"{flag}{str(node_tuple[0])}"

    def str_edge(etype):
        style = ""
        if etype in [HeteroEdgeType.DOMINATOR.value, HeteroEdgeType.SUCCESSOR.value]:
            style = f'label="{etype}"'
        elif etype in [HeteroEdgeType.WT.value, HeteroEdgeType.RF.value,
                       HeteroEdgeType.REFTO.value, HeteroEdgeType.REFED.value]:
            style = f'color="orangered", style="dashed", label="{etype}"'
        elif etype in [HeteroEdgeType.LB.value, HeteroEdgeType.LBN.value]:
            style = f'color="teal",label="{etype}"'
        elif etype in [HeteroEdgeType.DDF.value, HeteroEdgeType.DDC.value]:
            style = f'color="limegreen", penwidth="3.0", style="dotted", label="{etype}"'
        elif etype in [HeteroEdgeType.CALL.value, HeteroEdgeType.CALLED.value]:
            style = f'label="{etype}"'
        return style

    def initialize_node(node_tuple):
        content = ""
        if node_tuple[-1] == 'mapping_var':
            content = f'{str_node(node_tuple)}[color="darkviolet"]'
        elif node_tuple[-1] == 'other_state_var':
            content = f'{str_node(node_tuple)}[color="goldenrod"]'
        elif node_tuple[-1] == 'local_var':
            content = f'{str_node(node_tuple)}[color="royalblue"]'
        return content

    file = "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/test_ERC.sol"
    solc_version = get_solc_version(file)
    solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}'
    sol = SolidityInfo(file, solc=solc_version)
    mapping_vars = []
    top_contracts = sol.slither.contracts
    mapping_vars = get_mapping_vars_for_train(top_contracts)
    graph = VPG(sol, mapping_vars)
    node_list = []
    for n in list(graph.nodes_index.values()):
        n_content = initialize_node(n)
        if n_content == "": continue
        node_list.append(n_content)
    node_content = "\n".join(node_list)
    edges_list = []
    for e in graph.edges:
        src = str_node(e[0])
        dst = str_node(e[2])
        edges_list.append(f"{src}->{dst}[{str_edge(e[1])}];")
    edges_content = "\n".join(edges_list)
    dot_content = (f'digraph {{ \n '
                   f'{node_content}\n'
                   f'{edges_content}\n '
                   f'}}')
    with open("test_ERC.dot", "w") as fw:
        fw.write(dot_content)


def process_dgl(graph):
    node_types = ['mapping_var', 'other_state_var', 'local_var', 'function_node']
    edge_types = [
        ('function_node', HeteroEdgeType.DOMINATOR.value, 'function_node'),
        ('function_node', HeteroEdgeType.SUCCESSOR.value, 'function_node'),
        # ('function_node', HeteroEdgeType.FATHER.value, 'function_node'),
        # ('function_node', HeteroEdgeType.SON.value, 'function_node'),
        ('function_node', HeteroEdgeType.CALL.value, 'function_node'),
        ('function_node', HeteroEdgeType.CALLED.value, 'function_node'),
        ('function_node', HeteroEdgeType.REFTO.value, 'local_var'),
        ('local_var', HeteroEdgeType.REFED.value, 'function_node')
    ]
    for i in range(3):
        edge_types.append((node_types[i], HeteroEdgeType.RF.value, 'function_node'))
        edge_types.append(('function_node', HeteroEdgeType.WT.value, node_types[i]))
        edge_types.append((node_types[i], HeteroEdgeType.LB.value, 'function_node'))
        edge_types.append((node_types[i], HeteroEdgeType.LBN.value, 'function_node'))
    for i in range(3):
        for j in range(3):
            if not (i == 2 and j == 2):
                edge_types.append((node_types[i], HeteroEdgeType.DDF.value, node_types[j]))
    for i in range(2):
        for j in range(2):
            edge_types.append((node_types[i], HeteroEdgeType.DDC.value, node_types[j]))
    nums_node_dict = dict()
    graph_data = {key: ([], []) for key in edge_types}
    for node_type in node_types:
        num = len(list(filter(lambda x: x[-1] == node_type, graph['nodes_index'])))
        nums_node_dict[node_type] = num
    for edge in graph['edges']:
        edge_type = (edge[0][-1], edge[1], edge[2][-1])
        graph_data[edge_type][0].append(edge[0][0])
        graph_data[edge_type][1].append(edge[2][0])
    g_graph_data = dict()
    for key in edge_types:
        if not (len(graph_data[key][0]) == 0 and len(graph_data[key][1]) == 0):
            g_graph_data[key] = (torch.tensor(graph_data[key][0]), torch.tensor(graph_data[key][1]))
        else:
            g_graph_data[key] = ([], [])
    g = dgl.heterograph(g_graph_data, num_nodes_dict=nums_node_dict)
    for node_type in node_types:
        if nums_node_dict[node_type] == 0: continue
        g.nodes[node_type].data['feats'] = torch.randn(nums_node_dict[node_type], N_FEATURES)
    g.nodes['mapping_var'].data['labels'] = torch.tensor(
        list(dict(sorted(graph['labels'].items())).values()))
    return g


def identify_key_var():
    eth = all_path("/home/jrj/postgraduate/Symbolic/Backdoor/dataset/eth", [".sol"])
    bsc = all_path("/home/jrj/postgraduate/Symbolic/Backdoor/dataset/bsc", [".sol"])
    total = eth + bsc

    rank = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    etypes = get_rel_names(["CF", "DFB", "DFE", "CD", "DD", "FC"])
    in_size = N_FEATURES
    out_size = 1
    hid_size = 256
    model = RGCN_GAT(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes, n_heads=2).to(
        device)

    checkpoint = torch.load("/home/jrj/postgraduate/vpg/data/model/GCN_GAT_100_256_2024-02-20 15:53:45.bin")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    mapping_var_dict = {}
    for file in tqdm(total):
        try:
            solc_version = get_solc_version(file)
            solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}'
            sol = SolidityInfo(file, solc=solc_version)
            top_contracts = sol.top_contracts
            mapping_vars = get_mapping_vars_for_train(top_contracts)
            if len(mapping_vars) == 0:
                continue
            graph = VPG(sol, mapping_vars)
            graph_data = {
                "nodes_index": list(graph.nodes_index.values()),
                "edges": graph.edges,
                "labels": graph.mapping_vars_name_label
            }
            g = process_dgl(graph_data).to(device)

            with torch.no_grad():
                output = model(g, g.ndata["feats"])
                pred = torch.sigmoid(output).cpu().numpy()
                pred = np.where(pred >= 0.5, 1, 0)
            mapping_result = {}
            for mpv in graph.mapping_nodes_index:
                label = pred[mpv].item()
                var_name = graph.mapping_nodes_index[mpv].canonical_name
                mapping_result[var_name] = label
            mapping_var_dict[file] = mapping_result
        except:
            gc.collect()
            continue
    with open("mapping_identify_result.pkl", "wb") as fw:
        pickle.dump(mapping_var_dict, fw)


def confusionDataset():
    rank = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    etypes = get_rel_names(["CF", "DFB", "DFE", "CD", "DD", "FC"])
    in_size = N_FEATURES
    out_size = 1
    hid_size = 256
    model = RGCN_GAT(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes, n_heads=2,
                     gcn_layers=5).to(
        device)

    checkpoint = torch.load("/home/jrj/postgraduate/vpg/data/model/GCN_GAT_100_256_2024-02-20 15:53:45.bin")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    result_root = "/home/jrj/postgraduate/backdoor_data/result/real_world_v2"
    file_root = "/home/jrj/postgraduate/backdoor_data/dataset/real_world_selected_v2"
    folders = os.listdir(result_root)
    FP, FN = 0, 0
    for folder in tqdm(folders):
        file_path = os.path.join(file_root, folder)
        file = all_path(file_path, [".sol"])[0]
        solc_version = get_solc_version(file)
        solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}'
        sol = SolidityInfo(file, solc=solc_version)
        top_contracts = sol.top_contracts
        mapping_vars = get_mapping_vars_for_train(top_contracts)
        if len(mapping_vars) == 0:
            continue
        graph = VPG(sol, mapping_vars)
        graph_data = {
            "nodes_index": list(graph.nodes_index.values()),
            "edges": graph.edges,
            "labels": graph.mapping_vars_name_label
        }
        g = process_dgl(graph_data).to(device)

        with torch.no_grad():
            output = model(g, g.ndata["feats"])
            pred = torch.sigmoid(output).cpu().numpy()
            pred = np.where(pred >= 0.5, 1, 0)
        mapping_result = {}
        for mpv in graph.mapping_nodes_index:
            label = pred[mpv].item()
            var_name = graph.mapping_nodes_index[mpv].name
            mapping_result[var_name] = label
        has_FN, has_FP, has_TP, has_TN = False, False, False, False
        for var_name in mapping_result:
            has_balance = fuzz.partial_ratio(var_name.lower(), "balance") >= 85
            if has_balance:
                if mapping_result[var_name] == 0:
                    has_FN = True
                else:
                    has_TP = True
            else:
                if mapping_result[var_name] == 1:
                    has_FP = True
                else:
                    has_TN = True
        if has_FN:
            FN += 1
        if has_FP and (not has_TP):
            FP += 1
    print(f"FN:{FN}, FP:{FP}")


if __name__ == '__main__':
    # draw_dot()
    # identify_key_var()
    confusionDataset()
