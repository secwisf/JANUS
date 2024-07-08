import hashlib
from typing import List, Dict

from slither.core.cfg.node import Node
from slither.core.declarations import Contract
from z3 import Then, ParThen, OrElse, Solver, With

from backdoor.state.path_info import PathInfo
from backdoor.summaryutils.extra_type import SelfDestructSummary
from backdoor.summaryutils.state_variable.address_type import AddressTypeSummary
from backdoor.summaryutils.state_variable.bool_type import BoolTypeSummary
from backdoor.summaryutils.state_variable.mapping_type import MappingTypeSummary
from backdoor.summaryutils.state_variable.number_type import NumberTypeSummary
from backdoor.symbolic.svm import ExtraStatus
from backdoor.utils.node_utils import is_number_type, is_address_type, is_bool_type, is_mapping_type


# my_solver = Solver()


class PathSummary:
    """
    generate function path summaries
    """

    def __init__(self):
        t2 = Then('simplify', 'solve-eqs', 'smt', 'bit-blast')
        _t = Then('tseitin-cnf-core', 'split-clause')
        t1 = ParThen(_t, t2)
        self.my_solver = OrElse(t1, t2).solver()
        # Solver.set("timeout", 100)
        self.delta_summaries_set = set()
        # self.my_solver = Then(With('simplify', mul2concat=True),
        #                       'solve-eqs',
        #                       'bit-blast',
        #                       'aig',
        #                       'sat').solver()
        # self.my_solver = Solver()
        self.my_solver.set("timeout", 180)

    def reset(self):
        self.my_solver.reset()
        self.delta_summaries_set = set()

    def generate_summary(self, exec_status: int, node_list: List[Node], origin_symbolic_state_vars: Dict,
                         new_symbolic_state_vars: Dict, slither_ir_vars: Dict, constraints: List,
                         contract: Contract) -> Dict:
        summary = {}
        state_vars_written = PathInfo.get_path_state_variables_witten_origin(node_list)
        summary['contract'] = {}
        if exec_status == ExtraStatus.revert:
            for state_var in state_vars_written:
                summary[state_var.name] = {}
        else:
            self.my_solver.append(constraints)
            sat = str(self.my_solver.check())
            self.my_solver.reset()
            if sat in ['sat', 'unknown']:
                for state_var in state_vars_written:
                    try:
                        if (state_var.name not in origin_symbolic_state_vars) or (
                                state_var.name not in new_symbolic_state_vars):
                            summary[state_var.name] = {}
                            continue
                        origin_symbolic_var = origin_symbolic_state_vars[state_var.name]
                        new_symbolic_var = new_symbolic_state_vars[state_var.name]
                        if is_number_type(state_var.type):
                            summary[state_var.name] = NumberTypeSummary.make_summary(contract, new_symbolic_var,
                                                                                     new_symbolic_state_vars,
                                                                                     origin_symbolic_var, constraints,
                                                                                     self.my_solver)
                        elif is_address_type(state_var.type):
                            summary[state_var.name] = AddressTypeSummary.make_summary(new_symbolic_var,
                                                                                      origin_symbolic_var,
                                                                                      constraints, self.my_solver)
                        elif is_bool_type(state_var.type):
                            summary[state_var.name] = BoolTypeSummary.make_summary(new_symbolic_var, constraints,
                                                                                   self.my_solver)
                        elif is_mapping_type(state_var.type):
                            summary[state_var.name] = MappingTypeSummary.make_summary(contract, state_var,
                                                                                      slither_ir_vars,
                                                                                      origin_symbolic_var,
                                                                                      new_symbolic_state_vars,
                                                                                      constraints, self.my_solver)
                    except:
                        summary[state_var.name] = {}
                if exec_status == ExtraStatus.selfdestruct:
                    summary['contract'] = SelfDestructSummary
            else:
                for state_var in state_vars_written:
                    summary[state_var.name] = {}
        return summary

    def compare_summaries(self, contract: Contract, base_delta_summary: Dict, summary_normal: Dict,
                          summary_vulnerable: Dict):
        cache_delta_summary = {}

        for state_var_name in summary_normal:
            if state_var_name == 'contract':
                delta_summary = self.compute_elementary_type_delta_summary(summary_normal[state_var_name],
                                                                           summary_vulnerable[state_var_name])
                if (delta_summary['sign'] == 1) and (len(delta_summary['value']) == 0):
                    continue
            else:
                state_var = contract.get_state_variable_from_name(state_var_name)
                # delta是指vulnerable-normal，默认特权summary会多于normal
                if is_number_type(state_var.type) or is_bool_type(state_var.type) or is_address_type(state_var.type):
                    delta_summary = self.compute_elementary_type_delta_summary(summary_normal[state_var_name],
                                                                               summary_vulnerable[state_var_name])
                    if (delta_summary['sign'] == 1) and (len(delta_summary['value']) == 0):
                        continue
                elif is_mapping_type(state_var.type):
                    delta_summary = self.compute_mapping_type_delta_summary(summary_normal[state_var_name],
                                                                            summary_vulnerable[state_var_name])
                    if (delta_summary['sign'] == 1) and len(delta_summary['value']) == 0:
                        continue
                else:
                    continue
            cache_delta_summary[state_var_name] = delta_summary

        cache_delta_summary.update(
            (k, base_delta_summary[k]) for k in base_delta_summary if k not in cache_delta_summary)

        # 对字典排序，防止hash错误
        cache_delta_summary = recursive_sort(cache_delta_summary)

        if self.check_if_visited(cache_delta_summary, self.delta_summaries_set):
            cache_delta_summary = {}

        return cache_delta_summary

    @staticmethod
    def check_if_visited(delta_summary, delta_summaries_record) -> bool:
        if len(delta_summary) == 0:
            return True
        hash_tool = hashlib.sha3_512()
        hash_tool.update(str(delta_summary).encode('utf-8'))
        hash_val = hash_tool.hexdigest()
        if hash_val in delta_summaries_record:
            return True
        else:
            delta_summaries_record.add(hash_val)
            return False

    @staticmethod
    def compute_elementary_type_delta_summary(summary_normal, summary_vulnerable):
        delta_summary = {'sign': 1, 'value': {}}
        if (summary_normal is None and summary_vulnerable is None) or (
                len(summary_normal) == 0 and len(summary_vulnerable) == 0):
            return delta_summary
        if summary_normal is None or len(summary_normal) == 0:
            delta_summary['value'] = summary_vulnerable
        elif summary_vulnerable is None or len(summary_vulnerable) == 0:
            delta_summary['sign'] = 0
            delta_summary['value'] = summary_normal
        else:
            same = True
            for key in summary_normal:
                if str(summary_normal[key]) != str(summary_vulnerable[key]):
                    same = False
                    break
            if not same:
                delta_summary['value'] = summary_vulnerable
        return delta_summary

    @staticmethod
    def compute_mapping_type_delta_summary(summary_normal, summary_vulnerable):
        delta_summary = {'sign': 1, 'value': {}}
        if (summary_normal is None and summary_vulnerable is None) or (
                len(summary_normal) == 0 and len(summary_vulnerable) == 0):
            return delta_summary
        if summary_normal is None or len(summary_normal) == 0:
            delta_summary['value'] = {}
            for key in summary_vulnerable:
                delta_summary['value'][str(key)] = summary_vulnerable[key]
        elif summary_vulnerable is None or len(summary_vulnerable) == 0:
            delta_summary['sign'] = 0
            for key in summary_normal:
                delta_summary['value'][str(key)] = summary_normal[key]
        else:
            normal_address_set = set(summary_normal.keys())
            vulnerable_address_set = set(summary_vulnerable.keys())
            len_point_to = len(list(normal_address_set)[0])
            if len_point_to in [1, 2]:
                if normal_address_set != vulnerable_address_set:
                    for key in summary_vulnerable:
                        delta_summary['value'][str(key)] = summary_vulnerable[key]
                else:
                    same = True
                    for address in summary_normal:
                        if summary_normal[address] != summary_vulnerable[address]:
                            same = False
                            break
                    if not same:
                        for key in summary_vulnerable:
                            delta_summary['value'][str(key)] = summary_vulnerable[key]
        return delta_summary


def recursive_sort(d):
    if isinstance(d, dict):
        sorted_items = dict(sorted(d.items(), key=lambda x: str(x[0])))
        for k in sorted_items:
            sorted_items[k] = recursive_sort(sorted_items[k])
        return sorted_items
    else:
        return d
