from typing import List

from z3 import Solver, Not, BitVecNumRef


class AddressTypeSummary:
    @staticmethod
    def make_summary(new_symbolic_var, origin_symbolic_var, constraints: List, solver: Solver):
        summary = {'can_be_specific': None, 'can_be_different': False}
        AddressTypeSummary.analyse_can_be_specific(summary, new_symbolic_var)
        AddressTypeSummary.analyse_can_be_different(summary, new_symbolic_var, origin_symbolic_var, constraints, solver)
        return summary

    @staticmethod
    def analyse_can_be_specific(summary, new_symbolic_var):
        if isinstance(new_symbolic_var, BitVecNumRef):
            summary['can_be_specific'] = new_symbolic_var
        else:
            if str(new_symbolic_var) in ['user0', 'user1', 'user2', 'this']:
                summary['can_be_specific'] = new_symbolic_var

    @staticmethod
    def analyse_can_be_different(summary, new_symbolic_var, origin_symbolic_var, constraints, solver):
        curr_constraints = constraints + [Not(new_symbolic_var == origin_symbolic_var)]
        solver.append(curr_constraints)
        sat = str(solver.check())
        if sat in ['sat', 'unknown']:
            summary['can_be_different'] = True
        solver.reset()
