import os, sys

from slither.slither import Slither
from slither.core.solidity_types import MappingType
from tqdm import tqdm
sys.path.append('/home/jrj/postgraduate/Symbolic/Backdoor')
from backdoor.utils.node_utils import is_number_type
from process_dataset import all_path, get_solc_version



all_sols = all_path("/home/jrj/postgraduate/Symbolic/Backdoor/dataset/real_world", [".sol"])
err_num = 0
sol_list = []
for sol in tqdm(all_sols):
    if sol.endswith("_inline.sol"): continue
    try:
        solc_version = get_solc_version(sol)
        solc_path = f"/home/jrj/.solc-select/artifacts/solc-{solc_version}/solc-{solc_version}"
        slither_sol = Slither(sol, solc=solc_path)
        for contract in slither_sol.contracts_derived:
            count = 0
            if contract.is_possible_token:
                for state_var in contract.state_variables:
                    if isinstance(state_var.type, MappingType) and is_number_type(state_var.type.type_to):
                        count += 1
            if count > 1:
                sol_list.append(sol)
                break
    except:
        err_num += 1
        pass