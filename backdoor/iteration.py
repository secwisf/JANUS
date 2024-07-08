import copy
from typing import List, Dict

from func_timeout import func_set_timeout
from fuzzywuzzy import fuzz
from slither.core.cfg.node import Node
from slither.core.declarations import Contract

from backdoor.state.path_info import PathInfo
from backdoor.state.solidity_info import SolidityInfo
from backdoor.summaryutils.extra_type import SelfDestructSummary
from backdoor.summaryutils.path_summary import PathSummary
from backdoor.symbolic.svm import SVM
from backdoor.utils.node_utils import is_mapping_type, is_number_type, is_bool_type


class Iteration:
    def __init__(self, file_path: str, solc: str, balance_list: List[str]):
        self.file_path = file_path
        self.solc = solc
        self.balance_list = balance_list
        self.svm = SVM()
        self.summary_factory = PathSummary()

    @func_set_timeout(20 * 60)
    def iterative_algorithm(self):
        sol_info = SolidityInfo(self.file_path, self.solc)
        path_info = PathInfo(sol_info)
        self.svm.set(path_info)
        self.summary_factory.reset()
        contract_result = dict()
        has_backdoor = False
        for contract in sol_info.slither.contracts_derived:
            delta_summaries_record = {}
            round_num = 0

            init_symbolic_state_vars, global_constraints = self.svm.init_contract_state_vars(
                contract, round_num)

            possible_owner_funcs = path_info.get_possible_owner_funcs(
                self.svm.address_vars, contract)

            delta_summaries_record[round_num] = []
            for func in possible_owner_funcs:
                paths = path_info.basic_function_paths_nodes[contract.name][func]
                paths_num = len(paths)
                for path_index in range(paths_num):
                    path = paths[path_index]
                    v_summary = self.exec_with_role("owner", round_num, path, init_symbolic_state_vars,
                                                    init_symbolic_state_vars,
                                                    global_constraints, contract)
                    n_summary = self.exec_with_role("user", round_num, path, init_symbolic_state_vars,
                                                    init_symbolic_state_vars,
                                                    global_constraints, contract)
                    delta_summary = self.summary_factory.compare_summaries(contract, dict(), n_summary, v_summary)
                    if len(delta_summary) != 0:
                        delta_summaries_record[round_num].append([[func], delta_summary])

            while len(delta_summaries_record[round_num]) != 0:
                print(f"Round {round_num+1}...")
                round_num += 1
                delta_summaries_record[round_num] = []
                for last_delta_summary_pair in delta_summaries_record[round_num - 1]:
                    last_delta_summary = last_delta_summary_pair[1]
                    candidate_funcs = list(set(path_info.generate_candidate_funcs(contract, last_delta_summary)).union(set(possible_owner_funcs)))
                    
                    v_init_symbolic_state_vars = self.svm.update_initiate_state(round_num, contract, last_delta_summary,
                                                                                init_symbolic_state_vars)
                    for func in candidate_funcs:
                        paths = path_info.basic_function_paths_nodes[contract.name][func]
                        paths_num = len(paths)
                        for path_index in range(len(paths)):
                            if func == 'LockableToken.transfer(address,uint256)' and path_index == paths_num-1 :
                                print()
                            path = paths[path_index]
                            if func in possible_owner_funcs:
                                v_summary = self.exec_with_role("owner", round_num, path, v_init_symbolic_state_vars,
                                                                init_symbolic_state_vars, global_constraints, contract)
                            else:
                                v_summary = self.exec_with_role("user", round_num, path, v_init_symbolic_state_vars,
                                                                init_symbolic_state_vars, global_constraints, contract)

                            n_summary = self.exec_with_role("user", round_num, path, init_symbolic_state_vars,
                                                            init_symbolic_state_vars, global_constraints, contract)
                
                            delta_summary = self.summary_factory.compare_summaries(contract, last_delta_summary,
                                                                                   n_summary,
                                                                                   v_summary)
                            if len(delta_summary) != 0:
                                func_list = last_delta_summary_pair[0] + [func]
                                delta_summaries_record[round_num].append([func_list, delta_summary])
                                print("Differences found.")
                           
                exploit_list = self.analyse_backdoor_type(
                    delta_summaries_record, possible_owner_funcs, contract)
                if len(exploit_list) != 0:
                    contract_result[contract.name] = exploit_list
                    has_backdoor = True
                if has_backdoor:
                    break
        return has_backdoor, contract_result

    def exec_with_role(self, role: str, round_count: int, path: List[Node], curr_symbolic_state_vars,
                       init_symbolic_state_vars,
                       global_constraints, contract):
        symbolic_local_vars, local_constraints = self.svm.init_local_vars(path, round_count, curr_symbolic_state_vars)
        symbolic_state_vars = copy.deepcopy(curr_symbolic_state_vars)
        constraints, slither_ir_vars, extra_status = self.svm.sys_exec(symbolic_state_vars, symbolic_local_vars, path,
                                                                       role,
                                                                       round_count)
        summary = self.summary_factory.generate_summary(extra_status, path, init_symbolic_state_vars,
                                                        symbolic_state_vars, slither_ir_vars,
                                                        global_constraints + constraints + local_constraints, contract)
        return summary

    def analyse_backdoor_type(self, delta_summaries_record, possible_owner_funcs, contract: Contract):
        exploit_list = []
        backdoor_state_vars_name_set = set()
        round_num = len(delta_summaries_record)
        for round_index in range(round_num - 1, -1, -1):
            delta_summaries = delta_summaries_record[round_index]
            for delta_summary_pair in delta_summaries:
                delta_summary = delta_summary_pair[1]
                state_vars_name_set = sorted(set(delta_summary.keys()))

                related_state_vars = [contract.get_state_variable_from_name(state_var_name) for state_var_name in
                                      state_vars_name_set if state_var_name not in ['contract']]
                has_balances = False
                selfdestruct = False
                has_mapping_bool = False
                has_bool = False

                for state_var in related_state_vars:
                    if is_mapping_type(state_var.type):
                        name = state_var.canonical_name.split('.')[-1]
                        if name in self.balance_list:
                            has_balances = True
                        type_to = state_var.type.type_to
                        while is_mapping_type(type_to):
                            type_to = type_to.type_to

                        if is_bool_type(type_to):
                            has_mapping_bool = True
                    elif is_bool_type(state_var.type):
                        has_bool = True
                if 'contract' in state_vars_name_set:
                    if delta_summary['contract']['value'] == SelfDestructSummary:
                        selfdestruct = True
                if (not has_balances) and (not selfdestruct):
                    continue
                str_delta_summary = {}
                for key in state_vars_name_set:
                    str_delta_summary[key] = {}
                    str_delta_summary[key]['sign'] = delta_summary[key]['sign']
                    str_delta_summary[key]['value'] = {}
                    for value_key in delta_summary[key]['value']:
                        if isinstance(delta_summary[key]['value'][value_key], Dict):
                            str_delta_summary[key]['value'][value_key] = {}
                            for subkey in delta_summary[key]['value'][value_key]:
                                subkey_val = delta_summary[key]['value'][value_key][subkey]
                                if (subkey_val is not None) and (not isinstance(subkey_val, (bool, List))):
                                    str_delta_summary[key]['value'][value_key][subkey] = str(subkey_val)
                                else:
                                    str_delta_summary[key]['value'][value_key][subkey] = subkey_val
                        else:
                            if (delta_summary[key]['value'][value_key] is not None) and (
                                    not isinstance(delta_summary[key]['value'][value_key], (bool, List))):
                                str_delta_summary[key]['value'][value_key] = str(delta_summary[key]['value'][value_key])
                            else:
                                str_delta_summary[key]['value'][value_key] = delta_summary[key]['value'][value_key]
                exploit_list.append(
                    {'summary': str_delta_summary, 'funcs': delta_summary_pair[0]})

        return exploit_list
