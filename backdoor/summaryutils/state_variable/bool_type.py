from typing import List

from z3 import Solver


class BoolTypeSummary():
    @staticmethod
    def make_summary(new_symbolic_var, constraints: List, solver: Solver):
        summary = {'can_be_true': False, 'can_be_false': False}
        BoolTypeSummary.analyse_can_be_true(summary, new_symbolic_var, constraints, solver)
        BoolTypeSummary.analyse_can_be_false(summary, new_symbolic_var, constraints, solver)
        return summary

    @staticmethod
    def analyse_can_be_true(summary, new_symbolic_var, constraints, solver):
        curr_constraints = constraints + [(new_symbolic_var == True)]
        solver.append(curr_constraints)
        sat = str(solver.check())
        if sat in ['sat', 'unknown']:
            summary['can_be_true'] = True
        solver.reset()

    @staticmethod
    def analyse_can_be_false(summary, new_symbolic_var, constraints, solver):
        curr_constraints = constraints + [(new_symbolic_var == False)]
        solver.append(curr_constraints)
        sat = str(solver.check())
        if sat in ['sat', 'unknown']:
            summary['can_be_false'] = True
        solver.reset()
