import pickle

import dgl
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

from prepare.graph import SolidityInfo, VPG
from utils.sol_utils import get_solc_version, is_mapping_type


def create_task():
    file = '/home/jrj/postgraduate/Ethereum_smart_contract_datast/Ethereum_smart_contract_datast/contract_dataset_ethereum/contract5/4585.sol'
    solc_version = get_solc_version(file)
    solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}/solc-{solc_version}'
    sol = SolidityInfo(file, solc=solc_version)
    base_path = "graph_v2"
    with open(f"data/{base_path}/mapping_vars_dict.pkl", "rb") as fr:
        mapping_vars_dict = pickle.load(fr)
    graph = VPG(sol, mapping_vars_dict[file])
    print()



def make_plot():
    file = '/home/jrj/postgraduate/Ethereum_smart_contract_datast/Ethereum_smart_contract_datast/contract_dataset_ethereum/contract5/4585.sol'
    solc_version = get_solc_version(file)
    solc_version = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}/solc-{solc_version}'
    sol = SolidityInfo(file, solc=solc_version)
    base_path = "graph_v2"
    with open(f"data/{base_path}/mapping_vars_dict.pkl", "rb") as fr:
        mapping_vars_dict = pickle.load(fr)
    graph = VPG(sol, mapping_vars_dict[file])




if __name__ == '__main__':
    # create_task()
    make_plot()