import sys

sys.path.append("/home/jrj/postgraduate/vpg")
from process.vpg_dataset import HeteroMetaDataset

from thefuzz import fuzz

from utils.dataManager import load
from math import ceil
from multiprocessing import Pool, Manager
import pickle

from prepare.graph import SolidityInfo, VPG
from utils.sol_utils import get_solc_version, is_mapping_type, all_path, is_number_type, is_address_type


def generate_mapping_dict(f_index: int, f_files, f_mapping_vars_dict, f_error_list, f_success_list, f_lock):
    print(f"{'*' * 10}Generating index {str(f_index)} {'*' * 10}")
    tmp_mapping_vars_dict = {}
    tmp_success_files = []
    tmp_error_files = []
    for file in f_files:
        try:
            solc_version = get_solc_version(file)
            solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}'
            sol = SolidityInfo(file, solc=solc_version)
            tmp_mapping_vars_dict[file] = []
            top_contracts = sol.top_contracts
            for tc in top_contracts:
                if not tc.is_possible_token: continue
                state_vars = tc.state_variables
                balance_func_sv = set()
                for func in tc.all_functions_called:
                    if fuzz.partial_ratio(func.name.lower(), "balanceof") >= 85:
                        balance_func_sv = balance_func_sv.union(set(func.state_variables_read))
                for sv in state_vars:
                    if not (is_mapping_type(sv.type) and is_address_type(sv.type.type_from)): continue
                    if is_number_type(sv.type.type_to):
                        if len(balance_func_sv) != 0 and sv in balance_func_sv:
                            tmp_mapping_vars_dict[file].append([sv.canonical_name, 1])
                        else:
                            if fuzz.partial_ratio(sv.name.lower(), "balance") >= 85:
                                tmp_mapping_vars_dict[file].append([sv.canonical_name, 1])
                            else:
                                tmp_mapping_vars_dict[file].append([sv.canonical_name, 0])
                    else:
                        tmp_mapping_vars_dict[file].append([sv.canonical_name, 0])
            tmp_success_files.append(file)
        except:
            tmp_error_files.append(file)
            continue
    f_lock.acquire()
    f_mapping_vars_dict.update(tmp_mapping_vars_dict)
    f_error_list += tmp_error_files
    f_success_list += tmp_success_files
    f_lock.release()


def generate_VPG(base_path, f_index: int, f_files, f_mapping_vars_dict, f_error_vpg_list, f_lock):
    print(f"{'*' * 10}Generating index {str(f_index)} {'*' * 10}")
    graph_data = []
    for file in f_files:
        mapping_vars_name = f_mapping_vars_dict[file]
        mapping_vars_node = []
        mapping_vars_label = {}
        if len(f_mapping_vars_dict[file]) == 0: continue
        try:
            solc_version = get_solc_version(file)
            solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}'
            sol = SolidityInfo(file, solc=solc_version)
            top_contracts = sol.top_contracts
            for tc in top_contracts:
                mapping_vars_node += list(filter(lambda x: is_mapping_type(x.type), tc.state_variables))
            graph = VPG(sol, mapping_vars_name)
            for name, label in mapping_vars_name:
                for node in mapping_vars_node:
                    if name == node.canonical_name and node in graph.nodes_index:
                        mapping_vars_label[graph.nodes_index[node][0]] = label
                        break
            data = {
                "file": file,
                "nodes_index": list(graph.nodes_index.values()),
                "edges": graph.edges,
                "labels": mapping_vars_label
            }
            graph_data.append(data)
        except Exception as e:
            f_lock.acquire()
            f_error_vpg_list.append(file)
            f_lock.release()
    with open(f"../data/{base_path}/graph_data_{str(f_index)}.pkl", "wb") as fw:
        pickle.dump(graph_data, fw)


def generate_VPG_task(base_path):
    with open(f"../data/graph_v2/mapping_vars_dict.pkl", "rb") as fr:
        mapping_vars_dict = pickle.load(fr)
    pool = Pool(15)
    manager = Manager()
    files = manager.list(list(mapping_vars_dict.keys()))
    manager_dict = manager.dict()
    manager_dict.update(mapping_vars_dict)
    error_vpg_list = manager.list()
    lock = manager.Lock()
    batch_size = 1000
    jobs = ceil(len(files) / batch_size)
    for index in range(jobs):
        pool.apply_async(func=generate_VPG, args=(
            base_path, index, files[index * batch_size:(index + 1) * batch_size], manager_dict, error_vpg_list, lock,))
    pool.close()
    pool.join()
    with open(f"../data/{base_path}/error_vpg_list.pkl", "wb") as fw:
        pickle.dump(error_vpg_list, fw)


def generate_mapping_dict_task(base_path):
    files = all_path(
        "/home/jrj/postgraduate/Ethereum_smart_contract_datast/Ethereum_smart_contract_datast/contract_dataset_ethereum/",
        [".sol"])
    pool = Pool(10)
    manager = Manager()
    files = manager.list(files)
    manager_dict = manager.dict()
    error_list = manager.list()
    success_list = manager.list()
    lock = manager.Lock()
    batch_size = 1000
    jobs = ceil(len(files) / batch_size)
    for index in range(jobs):
        pool.apply_async(func=generate_mapping_dict, args=(
            index, files[index * batch_size:(index + 1) * batch_size], manager_dict, error_list, success_list, lock,))
    pool.close()
    pool.join()
    with open(f"../data/{base_path}/success_files.pkl", "wb") as fw:
        pickle.dump(list(success_list), fw)
    with open(f"../data/{base_path}/error_files.pkl", "wb") as fw:
        pickle.dump(list(error_list), fw)
    with open(f"../data/{base_path}/mapping_vars_dict.pkl", "wb") as fw:
        pickle.dump(dict(manager_dict), fw)


def generate_meta_dataset():
    jobs = 4
    base = "graph_v3"
    result = HeteroMetaDataset(base, [], force_reload=True)
    for i in range(jobs):
        p_data = HeteroMetaDataset(base, [], suffix=str(i))
        result = result + p_data
    result.save()


def generate_partial_meta_dataset(base, index, dataset):
    HeteroMetaDataset(base, dataset, force_reload=True, suffix=str(index))


def generate_meta_dataset_task():
    base = "graph_v3"
    pool = Pool(4)
    manager = Manager()
    dataset = manager.list(load(f"/home/jrj/postgraduate/vpg/data/{base}"))
    batch_size = 5000
    jobs = ceil(len(dataset) / batch_size)
    for index in range(jobs):
        pool.apply_async(func=generate_partial_meta_dataset, args=(
            base, index, dataset[index * batch_size:(index + 1) * batch_size],))
    pool.close()
    pool.join()


if __name__ == '__main__':
    generate_meta_dataset()
    # generate_meta_dataset_task()
    # generate_VPG_task("graph_v3")
    # generate_mapping_dict_task()
