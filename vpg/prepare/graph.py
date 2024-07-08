from enum import Enum, auto
from typing import List, Optional

from slither.analyses.data_dependency.data_dependency import is_dependent
from slither.core.cfg.node import NodeType
from slither.core.declarations import Contract, FunctionContract
from slither.core.expressions import Identifier
from slither.core.variables.variable import Variable
from slither.slither import Slither


class SolidityInfo:
    def __init__(self, file: str, solc: str):
        try:
            self.slither = Slither(file, solc=solc)
        except:
            raise Exception("Slither can not analyse the given file.")

    @property
    def top_contracts_old(self) -> List[Optional[Contract]]:
        return [contract for contract in self.slither.contracts_derived if
                contract.is_possible_token and len(contract.functions) > 0]

    @property
    def top_contracts(self) -> List[Optional[Contract]]:
        return self.slither.contracts_derived


class HeteroEdgeType(Enum):
    DOMINATOR = "DOMINATOR"
    SUCCESSOR = "SUCCESSOR"
    FATHER = "FATHER"
    SON = "SON"
    # ReadFrom
    RF = "RF"
    # WriteTo
    WT = "WT"
    # LedBy
    LB = "LB"
    # LedByNeg
    LBN = "LBN"
    # DataDependentInFunc
    DDF = "DDF"
    # DataDependentInContract
    DDC = "DDC"
    # FunctionCall
    CALL = "CALL"
    # FunctionCalled
    CALLED = "CALLED"
    # REFTo
    REFTO = "REFTO"
    # REFFrom
    REFED = "REFED"


class HomoEdgeType(Enum):
    DOMINATOR = auto()
    SUCCESSOR = auto()
    # FATHER = auto()
    # SON = auto()
    # ReadFrom
    RF = auto()
    # WriteTo
    WT = auto()
    # LedBy
    LB = auto()
    # DataDependentInFunc
    DDF = auto()
    # DataDependentInContract
    DDC = auto()
    # FunctionCall
    CALL = auto()
    # FunctionCalled
    CALLED = auto()
    # REFTo
    REFTO = auto()
    # REFFrom
    REFED = auto()


class VPG:
    def __init__(self, sol: SolidityInfo, mapping_vars_name: List):
        self.sol = sol
        self.mapping_vars_name_label = {name: label for name, label in mapping_vars_name}
        self.nodes = dict()
        # [src, edge_type, dst]
        self.edges = list()
        self.nodes_index = dict()
        self.mapping_nodes_index = dict()
        self.build_hetero_graph()

    def build_hetero_graph(self):
        self.get_hetero_nodes()
        self.get_edges()

    def build_homo_graph(self):
        self.get_homo_nodes()
        self.get_edges()

    def get_homo_nodes(self):
        mapping_vars = set()
        other_state_vars = set()
        related_local_vars = set()
        function_set = set()
        for contract in self.sol.top_contracts:
            mapping_state_vars = set(list(filter(
                lambda x: x.canonical_name in self.mapping_vars_name_label, contract.state_variables
            )))
            mapping_vars = mapping_vars.union(mapping_state_vars)

        curr_other_state_vars = -1
        while curr_other_state_vars != len(other_state_vars):
            curr_other_state_vars = len(other_state_vars)
            for contract in self.sol.top_contracts:
                state_vars = set(contract.state_variables)
                for func in contract.all_functions_called:
                    state_vars_used = set(func.state_variables_read).union(
                        set(func.state_variables_written)).intersection(state_vars)
                    flag = False
                    for sv in state_vars_used:
                        if sv in mapping_vars or sv in other_state_vars:
                            flag = True
                            break
                    if flag:
                        related_local_vars = related_local_vars.union(set(func.variables).difference(state_vars_used))
                        other_state_vars = other_state_vars.union(state_vars_used.difference(mapping_vars))
                        function_set.add(func)
                        #   other funcs that call this func
                        for reach_from_func in func.reachable_from_functions:
                            func_state_vars_used = set(reach_from_func.state_variables_read).union(
                                set(reach_from_func.state_variables_written))
                            related_local_vars = related_local_vars.union(
                                set(reach_from_func.variables).difference(func_state_vars_used))
                            other_state_vars = other_state_vars.union(func_state_vars_used.difference(mapping_vars))
                            function_set.add(reach_from_func)
                        #   other funcs that this func calls
                        for call in func.calls_as_expressions:
                            if not (isinstance(call.called, Identifier) and isinstance(call.called.value,
                                                                                       FunctionContract)): continue
                            call_func = call.called.value
                            func_state_vars_used = set(call_func.state_variables_read).union(
                                set(call_func.state_variables_written))
                            related_local_vars = related_local_vars.union(
                                set(call_func.variables).difference(func_state_vars_used))
                            other_state_vars = other_state_vars.union(func_state_vars_used.difference(mapping_vars))
                            function_set.add(call_func)
                        # TODO libcall
        self.nodes['mapping_vars'] = mapping_vars
        self.nodes['other_state_vars'] = other_state_vars
        self.nodes['related_local_vars'] = related_local_vars
        self.nodes['function_set'] = function_set

        index = 0
        for node in self.nodes['mapping_vars']:
            self.nodes_index[node] = [index, 'mapping_var']
            index += 1
        for node in self.nodes['other_state_vars']:
            self.nodes_index[node] = [index, 'other_state_var']
            index += 1
        for node in self.nodes['related_local_vars']:
            self.nodes_index[node] = [index, 'local_var']
            index += 1
        for func in self.nodes['function_set']:
            for node in func.nodes:
                self.nodes_index[node] = [index, 'function_node']
                index += 1

    def get_hetero_nodes(self):
        # 1. get all mapping variables
        # 2. get related functions using mapping variables
        # 3. get related variables according to these functions
        # 4. nodes include "mapping variables, related state variables, all nodes of related functions"
        mapping_vars = set()
        other_state_vars = set()
        related_local_vars = set()
        function_set = set()
        for contract in self.sol.top_contracts:
            mapping_state_vars = set(list(filter(
                lambda x: x.canonical_name in self.mapping_vars_name_label, contract.state_variables
            )))
            mapping_vars = mapping_vars.union(mapping_state_vars)

        curr_other_state_vars = -1
        while curr_other_state_vars != len(other_state_vars):
            curr_other_state_vars = len(other_state_vars)
            for contract in self.sol.top_contracts:
                state_vars = set(contract.state_variables)
                for func in contract.all_functions_called:
                    state_vars_used = set(func.state_variables_read).union(
                        set(func.state_variables_written)).intersection(state_vars)
                    flag = False
                    for sv in state_vars_used:
                        if sv in mapping_vars or sv in other_state_vars:
                            flag = True
                            break
                    if flag:
                        related_local_vars = related_local_vars.union(set(func.variables).difference(state_vars_used))
                        other_state_vars = other_state_vars.union(state_vars_used.difference(mapping_vars))
                        function_set.add(func)
                        #   other funcs that call this func
                        for reach_from_func in func.reachable_from_functions:
                            func_state_vars_used = set(reach_from_func.state_variables_read).union(
                                set(reach_from_func.state_variables_written))
                            related_local_vars = related_local_vars.union(
                                set(reach_from_func.variables).difference(func_state_vars_used))
                            other_state_vars = other_state_vars.union(func_state_vars_used.difference(mapping_vars))
                            function_set.add(reach_from_func)
                        #   other funcs that this func calls
                        for call in func.calls_as_expressions:
                            if not (isinstance(call.called, Identifier) and isinstance(call.called.value,
                                                                                       FunctionContract)): continue
                            call_func = call.called.value
                            func_state_vars_used = set(call_func.state_variables_read).union(
                                set(call_func.state_variables_written))
                            related_local_vars = related_local_vars.union(
                                set(call_func.variables).difference(func_state_vars_used))
                            other_state_vars = other_state_vars.union(func_state_vars_used.difference(mapping_vars))
                            function_set.add(call_func)
                        # TODO libcall
        self.nodes['mapping_vars'] = mapping_vars
        self.nodes['other_state_vars'] = other_state_vars
        self.nodes['related_local_vars'] = related_local_vars
        self.nodes['function_set'] = function_set

        index = 0
        for node in self.nodes['mapping_vars']:
            self.nodes_index[node] = [index, 'mapping_var']
            self.mapping_nodes_index[index] = node
            index += 1
        index = 0
        for node in self.nodes['other_state_vars']:
            self.nodes_index[node] = [index, 'other_state_var']
            index += 1
        index = 0
        for node in self.nodes['related_local_vars']:
            self.nodes_index[node] = [index, 'local_var']
            index += 1
        index = 0
        for func in self.nodes['function_set']:
            for node in func.nodes:
                self.nodes_index[node] = [index, 'function_node']
                index += 1

    def get_edges(self):
        for func in self.nodes['function_set']:
            if_node_dict = dict()
            for index in range(len(func.nodes)):
                node = func.nodes[index]
                # DOMINATOR and SUCCESSOR
                for dominator in node.dominators:
                    if dominator == node: continue
                    self.edges.append(
                        [self.nodes_index[dominator], HeteroEdgeType.DOMINATOR.value, self.nodes_index[node]])
                for successor in node.dominator_successors:
                    self.edges.append(
                        [self.nodes_index[successor], HeteroEdgeType.SUCCESSOR.value, self.nodes_index[node]])
                # FATHER and SON
                # for father in node.fathers:
                #     self.edges.append([self.nodes_index[father], HeteroEdgeType.FATHER.value, self.nodes_index[node]])
                # for son in node.sons:
                #     self.edges.append([self.nodes_index[son], HeteroEdgeType.SON.value, self.nodes_index[node]])
                # CALL and CALLED
                if len(node.calls_as_expression) != 0:
                    for call in node.calls_as_expression:
                        if not (isinstance(call.called, Identifier) and isinstance(call.called.value,
                                                                                   FunctionContract) and call.called.value in
                                self.nodes['function_set']): continue
                        call_func = call.called.value
                        self.edges.append(
                            [self.nodes_index[node], HeteroEdgeType.CALL.value,
                             self.nodes_index[call_func.entry_point]])
                        self.edges.append(
                            [self.nodes_index[call_func.entry_point], HeteroEdgeType.CALLED.value,
                             self.nodes_index[node]])
                        arguments = call.arguments
                        params = call_func.parameters
                        if len(arguments) != len(params): continue
                        for i in range(len(arguments)):
                            para = params[i]
                            if para not in self.nodes_index: continue
                            self.edges.append(
                                [self.nodes_index[node], HeteroEdgeType.REFTO.value,
                                 self.nodes_index[para]])
                            self.edges.append([self.nodes_index[para], HeteroEdgeType.REFED.value,
                                               self.nodes_index[node]])

                # ReadFrom and WriteTo
                # if len(node.variables_written) != 0:
                vars_written = [vw for vw in node.variables_written if isinstance(vw, Variable)]
                vars_read = [vr for vr in node.variables_read if isinstance(vr, Variable)]
                for vw in vars_written:
                    if vw not in self.nodes_index or node not in self.nodes_index: continue
                    self.edges.append([self.nodes_index[node], HeteroEdgeType.WT.value, self.nodes_index[vw]])
                for vr in vars_read:
                    if vr not in self.nodes_index or node not in self.nodes_index: continue
                    self.edges.append([self.nodes_index[vr], HeteroEdgeType.RF.value, self.nodes_index[node]])
                    # v2
                    # self.edges.append([self.nodes_index[vw], HeteroEdgeType.RF.value, self.nodes_index[vr]])
                    # self.edges.append([self.nodes_index[vr], HeteroEdgeType.WT.value, self.nodes_index[vw]])

                if node.type == NodeType.IF:
                    son_true = node.son_true
                    son_false = node.son_false
                    if_node_dict[node] = {"true": [], "false": []}
                    if_node_dict[node]["true"] = func.nodes[son_true.node_id:son_false.node_id]
                    son_true_vars = set()
                    # condition_vars = set(set(node.variables_read).union(node.variables_written))
                    for node_true in if_node_dict[node]["true"]:
                        son_true_vars = son_true_vars.union(
                            set(node_true.variables_read).union(set(node_true.variables_written)))
                    # for cv in condition_vars:
                    #     if cv not in self.nodes_index: continue
                    # LedBy
                    for stv in son_true_vars:
                        if stv not in self.nodes_index or node not in self.nodes_index: continue
                        self.edges.append([self.nodes_index[stv], HeteroEdgeType.LB.value, self.nodes_index[node]])

                    # LedByNeg
                    if_num = 1
                    tmp_index = son_false.node_id
                    while (if_num != 0) and (tmp_index < len(func.nodes)):
                        tmp_node = func.nodes[tmp_index]
                        if tmp_node.type == NodeType.IF:
                            if_num += 1
                        if tmp_node.type == NodeType.ENDIF:
                            if_num -= 1
                        if_node_dict[node]["false"].append(tmp_node)
                        tmp_index += 1
                    son_false_vars = set()
                    for node_false in if_node_dict[node]["false"]:
                        son_false_vars = son_false_vars.union(
                            set(node_false.variables_read).union(set(node_false.variables_written)))
                    for sfv in son_false_vars:
                        if sfv not in self.nodes_index or node not in self.nodes_index: continue
                        self.edges.append([self.nodes_index[sfv], HeteroEdgeType.LBN.value, self.nodes_index[node]])

            # DataDependentInFunc
            sv_list = set(func.state_variables_read).union(set(func.state_variables_written))
            v_list = set(func.variables).union(sv_list)
            sv_list = list(sv_list)
            v_list = list(v_list)
            for sv in sv_list:
                if sv not in self.nodes_index: continue
                for v in v_list:
                    if v not in self.nodes_index or sv == v: continue
                    if is_dependent(sv, v, func):
                        self.edges.append([self.nodes_index[sv], HeteroEdgeType.DDF.value, self.nodes_index[v]])
                    if is_dependent(v, sv, func):
                        self.edges.append([self.nodes_index[v], HeteroEdgeType.DDF.value, self.nodes_index[sv]])
        # DataDependentInContract
        sv_list = list(self.nodes['mapping_vars'].union(self.nodes['other_state_vars']))
        num = len(sv_list)
        for c in self.sol.top_contracts:
            for i in range(num):
                for j in range(i + 1, num):
                    if is_dependent(sv_list[i], sv_list[j], c):
                        self.edges.append(
                            [self.nodes_index[sv_list[i]], HeteroEdgeType.DDC.value, self.nodes_index[sv_list[j]]])
                    if is_dependent(sv_list[j], sv_list[i], c):
                        self.edges.append(
                            [self.nodes_index[sv_list[j]], HeteroEdgeType.DDC.value, self.nodes_index[sv_list[i]]])
