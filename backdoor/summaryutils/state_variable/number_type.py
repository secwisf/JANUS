from typing import List

from slither.core.declarations import Contract
from z3 import UGT, Solver, ULT, BitVecNumRef

from backdoor.utils.node_utils import is_number_type


class NumberTypeSummary:
    @staticmethod
    def make_summary(contract: Contract, new_symbolic_var, new_symbolic_state_vars, origin_symbolic_var,
                     constraints: List, solver: Solver):
        # summary = {'related_to_state_vars': []}
        summary = {'can_increase': False, 'can_decrease': False, 'related_to_state_vars': []}
        NumberTypeSummary.analyse_can_increase(summary, constraints, new_symbolic_var, origin_symbolic_var, solver)
        NumberTypeSummary.analyse_can_decrease(summary, constraints, new_symbolic_var, origin_symbolic_var, solver)
        NumberTypeSummary.analyse_related_to_state_vars(summary, contract, new_symbolic_var, new_symbolic_state_vars)
        return summary

    @staticmethod
    def analyse_can_increase(summary, constraints, new_symbolic_var, origin_symbolic_var, solver):
        curr_constraints = constraints + [UGT(new_symbolic_var, origin_symbolic_var)]
        solver.append(curr_constraints)
        sat = str(solver.check())
        if sat in ['sat', 'unknown']:
            summary['can_increase'] = True
        solver.reset()

    @staticmethod
    def analyse_can_decrease(summary, constraints, new_symbolic_var, origin_symbolic_var, solver):
        curr_constraints = constraints + [ULT(new_symbolic_var, origin_symbolic_var)]
        solver.append(curr_constraints)
        sat = str(solver.check())
        if sat in ['sat', 'unknown']:
            summary['can_decrease'] = True
        solver.reset()

    @staticmethod
    def analyse_related_to_state_vars(summary, contract: Contract, new_symbolic_var, new_symbolic_state_vars):
        summary_set = set()
        if not isinstance(new_symbolic_var, BitVecNumRef):
            for ssv in new_symbolic_state_vars:
                if (str(new_symbolic_state_vars[ssv]) in str(new_symbolic_var)) and (not isinstance(
                        new_symbolic_state_vars[ssv], BitVecNumRef)):
                    origin_ssv = contract.get_state_variable_from_name(ssv)
                    if is_number_type(origin_ssv.type):
                        summary_set.add(ssv)
            summary_list = list(summary_set)
            summary_list = sorted(summary_list, key=lambda x: str(x))
            summary['related_to_state_vars'] = summary_list
