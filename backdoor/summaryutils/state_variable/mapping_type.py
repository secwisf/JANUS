import re
from typing import List

from slither.core.declarations import Contract
from slither.core.solidity_types import MappingType
from slither.core.variables.state_variable import StateVariable
from slither.slithir.variables import ReferenceVariable
from z3 import Solver, BitVecNumRef, Select

from backdoor.summaryutils.state_variable.bool_type import BoolTypeSummary
from backdoor.summaryutils.state_variable.number_type import NumberTypeSummary
from backdoor.utils.exceptions import TooManyUsersError
from backdoor.utils.node_utils import is_number_type, is_bool_type, is_address_type


class MappingTypeSummary:
    @staticmethod
    def make_summary(contract: Contract, state_var: StateVariable, slither_ir_vars, origin_symbolic_var,
                     new_symbolic_state_vars, constraints: List, solver: Solver):
        summary = {}
        new_symbolic_var = new_symbolic_state_vars[state_var.name]
        address_list = MappingTypeSummary.deal_with_address(contract, state_var, slither_ir_vars,
                                                            new_symbolic_state_vars)
        if is_number_type(state_var.type.type_to):
            MappingTypeSummary.deal_with_map_number(contract, summary, address_list, new_symbolic_var,
                                                    origin_symbolic_var, constraints, solver, new_symbolic_state_vars)
        elif is_bool_type(state_var.type.type_to):
            MappingTypeSummary.deal_with_map_bool(summary, address_list, new_symbolic_var, constraints, solver)
        elif isinstance(state_var.type.type_to, MappingType):
            if is_number_type(state_var.type.type_to.type_to):
                MappingTypeSummary.deal_with_map_map_number(contract, summary, address_list, new_symbolic_var,
                                                            origin_symbolic_var, constraints, solver,
                                                            new_symbolic_state_vars)
            elif is_bool_type(state_var.type.type_to.type_to):
                MappingTypeSummary.deal_with_map_map_bool(summary, address_list, new_symbolic_var, constraints, solver)
        return summary

    @staticmethod
    def deal_with_address(contract, mapping_state_var, slither_ir_vars, symbolic_state_vars):
        address_set = set()
        len_point_to = len(re.findall(r'\bmapping\b', str(mapping_state_var.type)))
        for key in slither_ir_vars:
            if isinstance(key, ReferenceVariable) and key.points_to_origin.name == mapping_state_var.name and len(
                    slither_ir_vars[key]) == len_point_to:
                symbolic_ref_var_list = slither_ir_vars[key]
                if len(symbolic_ref_var_list) == 1:
                    address = MappingTypeSummary.get_str_address(contract, symbolic_ref_var_list[0],
                                                                 symbolic_state_vars,
                                                                 mapping_state_var)
                    if address is not None:
                        address_set.add((address, symbolic_ref_var_list[0]))
                elif len(symbolic_ref_var_list) == 2:
                    address_first = MappingTypeSummary.get_str_address(contract, symbolic_ref_var_list[0],
                                                                       symbolic_state_vars, mapping_state_var)
                    address_second = MappingTypeSummary.get_str_address(contract, symbolic_ref_var_list[1],
                                                                        symbolic_state_vars, mapping_state_var)
                    if (address_first is not None) and (address_second is not None):
                        address_set.add(
                            ((address_first, symbolic_ref_var_list[0]), (address_second, symbolic_ref_var_list[1])))
                else:
                    raise TooManyUsersError(key.contract.name, key.function.name)

        return list(address_set)

    @staticmethod
    def get_str_address(contract, symbolic_ref_var, symbolic_state_vars, mapping_state_var):
        if str(symbolic_ref_var) in ['user0', 'user1', 'user2', 'this']:
            return str(symbolic_ref_var)
        elif isinstance(symbolic_ref_var, BitVecNumRef):
            return str(symbolic_ref_var)
        else:
            for state_var_name in symbolic_state_vars:
                if symbolic_state_vars[state_var_name] is None: continue
                state_var = contract.get_state_variable_from_name(state_var_name)
                if is_address_type(state_var.type) and (
                        str(symbolic_ref_var) == str(symbolic_state_vars[state_var_name])):
                    return state_var_name
        return None

    @staticmethod
    def deal_with_map_number(contract, summary, address_set, new_symbolic_mapping_var, origin_symbolic_mapping_var,
                             constraints, solver, new_symbolic_state_vars):
        for address_pair in address_set:
            new_symbolic_number_var = Select(new_symbolic_mapping_var, address_pair[1])
            origin_symbolic_number_var = Select(origin_symbolic_mapping_var, address_pair[1])
            address_number_summary = NumberTypeSummary.make_summary(contract, new_symbolic_number_var,
                                                                    new_symbolic_state_vars, origin_symbolic_number_var,
                                                                    constraints, solver)
            summary[address_pair[0]] = address_number_summary

    @staticmethod
    def deal_with_map_bool(summary, address_set, new_symbolic_mapping_var, constraints, solver):
        for address_pair in address_set:
            new_symbolic_bool_var = Select(new_symbolic_mapping_var, address_pair[1])
            address_bool_summary = BoolTypeSummary.make_summary(new_symbolic_bool_var, constraints, solver)
            summary[address_pair[0]] = address_bool_summary

    @staticmethod
    def deal_with_map_map_number(contract, summary, address_set, new_symbolic_mapping_var, origin_symbolic_mapping_var,
                                 constraints, solver, new_symbolic_state_vars):
        for address_pair in address_set:
            new_symbolic_number_var = Select(Select(new_symbolic_mapping_var, address_pair[0][1]),
                                             address_pair[1][1])
            origin_symbolic_number_var = Select(Select(origin_symbolic_mapping_var, address_pair[0][1]),
                                                address_pair[1][1])
            address_number_summary = NumberTypeSummary.make_summary(contract, new_symbolic_number_var,
                                                                    new_symbolic_state_vars, origin_symbolic_number_var,
                                                                    constraints, solver)
            summary[(address_pair[0][0], address_pair[1][0])] = address_number_summary

    @staticmethod
    def deal_with_map_map_bool(summary, address_set, new_symbolic_mapping_var, constraints, solver):
        for address_pair in address_set:
            new_symbolic_bool_var = Select(Select(new_symbolic_mapping_var, address_pair[0][1]), address_pair[1][1])
            address_bool_summary = BoolTypeSummary.make_summary(new_symbolic_bool_var, constraints, solver)
            summary[(address_pair[0][0], address_pair[1][0])] = address_bool_summary
