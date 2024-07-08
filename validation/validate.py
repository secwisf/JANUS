import gc
import json
import os
import pickle
import shutil
import sys
import traceback
from math import ceil
from multiprocessing import Pool, Manager

from fuzzywuzzy import fuzz
###
import numpy as np
import dgl
import torch


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append('/root/vpg')
from func_timeout import FunctionTimedOut
from tqdm import tqdm

from backdoor.iteration import Iteration
from backdoor.state.path_info import PathInfo
from backdoor.state.solidity_info import SolidityInfo
from backdoor.utils.preprocess import PreProcess, SolFile
from scripts.process_dataset import get_solc_version, all_path
from backdoor.utils.preprocess import PreProcess, SolFile, del_comments_and_blank
#####
from vpg.process.vpg_model import RGCN_GAT
from vpg.utils.dataManager import get_rel_names
from vpg.prepare.graph import SolidityInfo, VPG, HeteroEdgeType
from vpg.process.vpg_dataset import N_FEATURES
from vpg.utils.sol_utils import get_solc_version, get_mapping_vars_for_train




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
        g.nodes[node_type].data['feats'] = torch.randn(nums_node_dict[node_type], 100)
    g.nodes['mapping_var'].data['labels'] = torch.tensor(
        list(dict(sorted(graph['labels'].items())).values()))
    return g






def single_real(current_dir,file, folder, balance_list=None):
    if balance_list is None:
        balance_list = []
    dst_root_path = f"{current_dir}/result/example"
    try:
        old_filename = os.path.basename(file)[:-4]
        new_filename = old_filename + '.json'
        dst_folder = os.path.join(dst_root_path, folder)
        new_path = os.path.join(dst_folder, new_filename)
        version = get_solc_version(file)
        solc_path = f"{current_dir}/.solc-select/artifacts/solc-{version}/solc-{version}"
        ite = Iteration(file_path=file, solc=solc_path, balance_list=balance_list)
        has_backdoor = False
        contract_result = []
        try:
            has_backdoor, contract_result = ite.iterative_algorithm()
            print('has_backdoor? '+str(has_backdoor))
        except:
            gc.collect()
        if not os.path.exists(dst_folder):
            os.makedirs(os.path.join(dst_root_path, folder))
        if (not has_backdoor) or (contract_result is None): return
        with open(new_path, 'w') as fw:
            json.dump(contract_result, fw)
        print('The result is saved in '+new_path)
    except:
        gc.collect()
        return
    return



if __name__ == '__main__':
    args = sys.argv  

    financial_vars = []
    solname = "example"
  
    for arg in args[1:]:
        if '--fvars=' in arg:
            financial_vars = arg.split('=')[-1].strip().split(',')
        elif '--sol=' in arg:
            solname = arg.split('=')[-1].strip().split('.')[0]

    current_path = os.path.abspath(os.path.dirname(__file__))
    current_dir = current_path.rpartition('/')[0]
   

    path = f"{current_dir}/contracts/{solname}.sol"
    version = get_solc_version(path)
    solc = f'{current_dir}/.solc-select/artifacts/solc-{version}/solc-{version}'

    
    if financial_vars == []:
        device = torch.device("cpu")
        etypes = get_rel_names(["CF", "DFB", "DFE", "CD", "DD", "FC"])
        in_size = 100#N_FEATURES
        out_size = 1
        hid_size = 256
        model = RGCN_GAT(in_feats=in_size, hid_feats=hid_size, out_feats=out_size, rel_names=etypes, gcn_layers=5, n_heads=2).to(
            device)

        checkpoint = torch.load(f"{current_dir}/vpg/GCN_GAT.bin",map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        model.eval()
        try:
            sol = SolidityInfo(path, solc=solc)
            top_contracts = sol.top_contracts
            mapping_vars = get_mapping_vars_for_train(top_contracts)
            
            if len(mapping_vars) != 0:
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
             
                for mpv in graph.mapping_nodes_index:
                    label = pred[mpv].item()
                    var_name = graph.mapping_nodes_index[mpv].canonical_name
                    if label == 1:
                        if '.' in var_name:
                            var_name = var_name.split('.')[-1]
                        financial_vars.append(var_name)
        except:
            print("Exception")
            gc.collect()


    print("target variables:" +str(financial_vars))

    del_comments_and_blank(path)
    sol_file = SolFile()
    sol_file.set_path(path, solc)
    pre_processor = PreProcess(sol_file, 2)
    pre_processor(path=f"{current_dir}/contracts/", name_suffix="_inline")
    single_real(current_dir,f"{current_dir}/contracts/{solname}_inline.sol", "", financial_vars)