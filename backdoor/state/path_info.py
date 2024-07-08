from typing import List, Dict

from slither.core.cfg.node import Node, NodeType
from slither.core.declarations import FunctionContract, Contract, Function

from backdoor.state.solidity_info import SolidityInfo


class PathInfo:
    def __init__(self, sol_info: SolidityInfo):
        self.sol_info = sol_info
        # {contract:{func:[[nodes]]}}
        self.basic_function_paths_nodes = {}
        self.init_paths()

    def init_paths(self):
        contracts_derived = self.sol_info.get_contracts_derived()
        self.basic_function_paths_nodes = {contract.name: {} for contract in contracts_derived}
        for contract in contracts_derived:
            # public_funcs doesn't contain 'constructor' function
            public_funcs = SolidityInfo.get_contracts_public_funcs(contract)
            self.basic_function_paths_nodes[contract.name] = {func.canonical_name: [] for func in public_funcs}
            self.__generate_basic_paths(public_funcs)

    def __generate_basic_paths(self, funcs: List[FunctionContract]):
        for func in funcs:
            # loop_nodes = [node for node in func.nodes if node.type in [NodeType.STARTLOOP, NodeType.IFLOOP]]
            # if len(loop_nodes) != 0: continue
            try:
                node_entry = func.nodes[0]
                self.__traverse_nodes(node_entry, [])
            except:
                pass

    # preprocessed is needed to ensure there is no loop
    def __traverse_nodes(self, node: Node, current_path: List):
        if len(node.sons) == 0:
            current_path.append(node)
            self.basic_function_paths_nodes[node.function.contract.name][node.function.canonical_name].append(
                current_path)
            return
        if (node.expression is None) and (node.type != NodeType.STARTLOOP):
            self.__traverse_nodes(node.sons[0], current_path + [node])
        else:
            if node.type == NodeType.IF:
                self.__traverse_nodes(node.son_true, current_path + [node])
                self.__traverse_nodes(node.son_false, current_path + [node])
            elif node.type == NodeType.STARTLOOP:
                frontier_list = list(node.dominance_frontier)
                end_loop_node = None
                for frontier in frontier_list:
                    if frontier.type == NodeType.ENDIF:
                        end_loop_node = frontier
                        break
                if end_loop_node is not None:
                    self.__traverse_nodes(end_loop_node.sons[0], current_path)
            else:
                self.__traverse_nodes(node.sons[0], current_path + [node])

    def get_possible_owner_funcs(self, address_vars:List, contract: Contract) -> List:
        public_funcs = self.sol_info.get_contracts_public_funcs(contract)
        owner_funcs = []
        for function in public_funcs:
            # if function.canonical_name == 'D100Token.decreaseAllowance(address,uint256)':
            #     print()
            all_functions = function.all_internal_calls() + [function] + function.modifiers
            all_nodes = [f.nodes for f in all_functions if isinstance(f, Function)]
            all_nodes = [item for sublist in all_nodes for item in sublist]
            
            judge = False
            for n in all_nodes:
                n_state_variables_read_set = set([sv.name for sv in n.state_variables_read])
                n_state_variables_written_set = set([sv.name for sv in n.state_variables_written])
                union_set = n_state_variables_read_set.union(n_state_variables_written_set)
                for address_var in address_vars:
                    if address_var.name in union_set:
                        judge = True
                        owner_funcs.append(function.canonical_name)
                        break
                if judge:
                    break
        return owner_funcs

    def generate_candidate_funcs(self, contract: Contract, delta_summary: Dict) -> List:
        # TODO类似get_possible_owner_funcs进行修改
        public_funcs = self.sol_info.get_contracts_public_funcs(contract)
        delta_state_vars = set([str(key) for key in delta_summary.keys()])
        candidate_funcs = []
        for function in public_funcs:
            state_variables_read_set = set([sv.name for sv in function.state_variables_read])
            state_variables_written_set = set([sv.name for sv in function.state_variables_written])
            intersection = state_variables_read_set.union(state_variables_written_set).intersection(delta_state_vars)
            if len(intersection) != 0:
                candidate_funcs.append(function.canonical_name)
        return candidate_funcs

    def generate_candidate_func_paths(self, contract: Contract, func: str, delta_summary: Dict):
        paths = self.basic_function_paths_nodes[contract.name][func]
        candidate_paths = []
        delta_state_vars = set([str(key) for key in delta_summary.keys()])
        for path in paths:
            state_variables_read_set = set(self.get_path_state_variables_read_str(path))
            state_variables_written_set = set(self.get_path_state_variables_witten_str(path))
            intersection = state_variables_read_set.union(state_variables_written_set).intersection(delta_state_vars)
            if len(intersection) != 0:
                candidate_paths.append(path)
        return candidate_paths

    @staticmethod
    def get_path_state_variables_witten_str(path: List[Node]):
        state_variables_written = set()
        state_variables_written_str = set()
        for node in path:
            state_variables_written = state_variables_written.union(set(node.state_variables_written))
        for state_var in state_variables_written:
            state_variables_written_str.add(state_var.name)
        return state_variables_written_str

    @staticmethod
    def get_path_state_variables_read_str(path: List[Node]):
        state_variables_read = set()
        state_variables_read_str = set()
        for node in path:
            state_variables_read = state_variables_read.union(set(node.state_variables_read))
        for state_var in state_variables_read:
            state_variables_read_str.add(state_var.name)
        return state_variables_read_str

    @staticmethod
    def get_path_state_variables_witten_origin(path: List[Node]):
        state_variables_written = set()
        for node in path:
            state_variables_written = state_variables_written.union(set(node.state_variables_written))
        return state_variables_written

    @staticmethod
    def get_path_state_variables_read_origin(path: List[Node]):
        state_variables_read = set()
        for node in path:
            state_variables_read = state_variables_read.union(set(node.state_variables_read))
        return state_variables_read
