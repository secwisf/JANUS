import copy
from enum import Enum
from typing import Union, Dict, List

from fuzzywuzzy import fuzz
from slither.core.cfg.node import Node, NodeType
from slither.core.declarations import SolidityVariableComposed, Contract, SolidityVariable
from slither.core.declarations.function import FunctionType
from slither.core.solidity_types import MappingType, Type, ElementaryType
from slither.core.variables.local_variable import LocalVariable
from slither.core.variables.state_variable import StateVariable
from slither.core.variables.variable import Variable
from slither.slithir.operations import Unary, Index, Assignment, Binary, BinaryType, TypeConversion, UnaryType, Delete, \
    Condition, LibraryCall, SolidityCall, EventCall
from slither.slithir.variables import ReferenceVariable, TemporaryVariable, Constant
from z3 import BitVec, Bool, Array, BitVecSort, BoolSort, ArraySort, Store, Select, Not, UDiv, URem, LShR, And, Or, ULT, \
    UGT, ULE, UGE, BitVecVal, Extract, StringVal, BoolVal, String, BitVecNumRef, Const, ForAll, Consts

from backdoor.state.path_info import PathInfo
from backdoor.state.solidity_info import SolidityInfo
from backdoor.utils.exceptions import TooManyUsersError, TooComplexMapping
from backdoor.utils.node_utils import is_number_type, is_bool_type, is_address_type, get_local_var_name, \
    is_mapping_type, is_signed_number, is_string_type, is_unsigned_number


class ExtraStatus(Enum):
    success = 0
    revert = 1
    selfdestruct = 2


# Symbolic Virtual Machine
class SVM:
    owner: Union[BitVec, BitVecVal]
    path_info: PathInfo
    sol_info: SolidityInfo
    address_vars: List[StateVariable]

    def __init__(self):
        # {contract:{var:symbol}}
        self.accounts = {
            'user0': BitVec('user0', 160),
            'user1': BitVec('user1', 160),
            'user2': BitVec('user2', 160),
            # 'user3': BitVec('user3', 160),
            'this': BitVec('this', 160)
        }

    def set(self, path_info: PathInfo):
        self.path_info = path_info
        self.sol_info = path_info.sol_info

    @staticmethod
    def get_owner_var(contract: Contract, address_vars: List):
        try:

            owner_name_score_dict = {state_var.name: 0 for state_var in address_vars}
            ico_name_score_dict = {state_var.name: 0 for state_var in address_vars}
            for var in address_vars:
                score = fuzz.ratio("owner", var.name.lower())
                owner_name_score_dict[var.name] = score

                score = fuzz.ratio("ico", var.name.lower())
                ico_name_score_dict[var.name] = score

            max_owner_name = max(owner_name_score_dict, key=owner_name_score_dict.get)
            max_ico_name = max(ico_name_score_dict, key=ico_name_score_dict.get)

            if owner_name_score_dict[max_owner_name] >= ico_name_score_dict[max_ico_name]:
                return contract.get_state_variable_from_name(max_owner_name)
            else:
                return contract.get_state_variable_from_name(max_ico_name)
        except:
            owner = Variable()
            owner.name = 'owner'
            return owner

    def init_contract_state_vars(self, contract, round_count: int):
        symbolic_state_vars = {}
        global_constraints = []
        state_vars = contract.state_variables
        self.address_vars = [
            state_var for state_var in state_vars if is_address_type(state_var.type)]
        self.owner = self.get_owner_var(contract, self.address_vars)
        for address_var in self.address_vars:
            symbolic_state_var, constraints = self.make_symbolic_var(address_var, round_count, symbolic_state_vars)
            if symbolic_state_var is not None:
                global_constraints += constraints
                symbolic_state_vars[address_var.name] = symbolic_state_var

        other_state_vars = [sv for sv in state_vars if sv not in self.address_vars]
        for state_var in other_state_vars:
            symbolic_state_var, constraints = self.make_symbolic_var(state_var, round_count, symbolic_state_vars)
            if symbolic_state_var is not None:
                global_constraints += constraints
                symbolic_state_vars[state_var.name] = symbolic_state_var

        construct_vars_funcs = [func for func in contract.all_functions_called if
                                func.function_type in [FunctionType.CONSTRUCTOR_VARIABLES,
                                                       FunctionType.CONSTRUCTOR_CONSTANT_VARIABLES]]
        constructors = SolidityInfo.get_contracts_constructor(contract)

        for construct_vars_func in construct_vars_funcs:
            nodes_list = construct_vars_func.nodes
            self.sys_exec(symbolic_state_vars, dict(), nodes_list, "owner", round_count)

        for constructor in constructors:
            nodes_list = constructor.nodes
            symbolic_local_vars, constraints = self.init_local_vars(nodes_list, round_count, symbolic_state_vars)
            global_constraints += constraints
            self.sys_exec(symbolic_state_vars, symbolic_local_vars, nodes_list, "owner", round_count)

        account_keys = list(self.accounts.keys())
        len_account_keys = len(account_keys)
        for userx_index in range(len_account_keys):
            for usery_index in range(userx_index + 1, len_account_keys):
                global_constraints.append(
                    Not(self.accounts[account_keys[userx_index]] == self.accounts[account_keys[usery_index]]))

        for address_state_var in self.address_vars:
            for userx_index in range(len(self.accounts) - 1):
                global_constraints.append(
                    Not(self.accounts[account_keys[userx_index]] == symbolic_state_vars[address_state_var.name]))
        return symbolic_state_vars, global_constraints

    def init_local_vars(self, nodes_list: List[Node], round_count: int, symbolic_state_vars):
        symbolic_local_vars = {}
        accompany_constraints = []
        local_vars_set = set()
        for node in nodes_list:
            local_vars_set = local_vars_set.union(set(node.local_variables_read)).union(
                set(node.local_variables_written))
        for local_var in local_vars_set:
            local_var_name = get_local_var_name(local_var, round_count)
            symbolic_local_var, constraints = self.make_symbolic_var(local_var, round_count, symbolic_state_vars,
                                                                     name=local_var_name)
            if symbolic_local_var is not None:
                accompany_constraints += constraints
                symbolic_local_vars[local_var_name] = symbolic_local_var
        return symbolic_local_vars, accompany_constraints

    @staticmethod
    def make_symbolic_constant_vars(var_type: Type, value=None):
        symbolic_var = None
        if is_address_type(var_type):
            value = value if value is not None else 0
            symbolic_var = BitVecVal(value, 160)
        elif is_number_type(var_type):
            value = value if value is not None else 0
            symbolic_var = BitVecVal(value, var_type.size)
        elif is_bool_type(var_type):
            value = value if value is not None else False
            symbolic_var = BoolVal(value)
        elif is_string_type(var_type):
            value = value if value is not None else ""
            symbolic_var = StringVal(value)
        return symbolic_var

    def make_symbolic_var(self, var, round_count: int, symbolic_state_vars, name: str = None):
        symbolic_var = None
        accompany_constraints = []
        if isinstance(var, StateVariable):
            name = name if name is not None else var.name
            if isinstance(var.type, ElementaryType):
                if var.type.name == 'address':
                    symbolic_var = BitVec(name, 160)
                elif is_number_type(var.type):
                    size = var.type.size
                    symbolic_var = BitVec(name, size)
                    if is_unsigned_number(var.type):
                        accompany_constraints.append(UGE(symbolic_var, 0))
                else:
                    symbolic_var = self.make_symbolic_constant_vars(
                        var_type=var.type)
            elif isinstance(var.type, MappingType):
                if is_number_type(var.type.type_to):
                    size = var.type.type_to.size
                    symbolic_var = Array(
                        name, BitVecSort(160), BitVecSort(size))
                    # if is_unsigned_number(var.type.type_to):
                    #     x = Const('c_x', BitVecSort(160))
                    #     accompany_constraints.append(ForAll(x, UGE(Select(symbolic_var, x), 0)))
                    # for account in self.accounts:
                    #     symbolic_var = Store(
                    #         symbolic_var, self.accounts[account], BitVecVal(0, size))
                elif is_bool_type(var.type.type_to):
                    symbolic_var = Array(name, BitVecSort(160), BoolSort())
                    for account in self.accounts:
                        symbolic_var = Store(
                            symbolic_var, self.accounts[account], BoolVal(False))
                    for address in self.address_vars:
                        symbolic_var = Store(
                            symbolic_var, symbolic_state_vars[address.name], BoolVal(False))

                elif isinstance(var.type.type_to, MappingType):
                    if is_number_type(var.type.type_to.type_to):
                        size = var.type.type_to.type_to.size
                        symbolic_var = Array(name, BitVecSort(160), ArraySort(
                            BitVecSort(160), BitVecSort(size)))
                        # if is_unsigned_number(var.type.type_to.type_to):
                        #     x, y = Consts('c_x c_y', BitVecSort(160))
                        #     accompany_constraints.append(ForAll([x,y], UGE(Select(Select(symbolic_var, x), y),0)))
                        # for account1 in self.accounts:
                        #     for account2 in self.accounts:
                        #         symbolic_var = Store(symbolic_var, self.accounts[account1],
                        #                              Store(symbolic_var[self.accounts[account1]],
                        #                                    self.accounts[account2],
                        #                                    BitVecVal(0, size)))
                    elif is_bool_type(var.type.type_to.type_to):
                        symbolic_var = Array(name, BitVecSort(
                            160), ArraySort(BitVecSort(160), BoolSort()))
                        for account1 in self.accounts:
                            for account2 in self.accounts:
                                symbolic_var = Store(symbolic_var, self.accounts[account1],
                                                     Store(symbolic_var[self.accounts[account1]],
                                                           self.accounts[account2],
                                                           BoolVal(False)))
                        for address in self.address_vars:
                            for account in self.accounts:
                                symbolic_var = Store(symbolic_var, symbolic_state_vars[address.name],
                                                     Store(symbolic_var[symbolic_state_vars[address.name]],
                                                           self.accounts[account],
                                                           BoolVal(False)))
                                symbolic_var = Store(symbolic_var, self.accounts[account],
                                                     Store(symbolic_var[self.accounts[account]],
                                                           symbolic_state_vars[address.name],
                                                           BoolVal(False)))

        else:
            name = name if name is not None else get_local_var_name(var, round_count)
            if is_number_type(var.type):
                size = var.type.size
                symbolic_var = BitVec(name, size)
                if is_unsigned_number(var.type):
                    accompany_constraints.append(UGE(symbolic_var, 0))
            elif is_bool_type(var.type):
                symbolic_var = Bool(name)
            elif is_address_type(var.type):
                symbolic_var = BitVec(name, 160)
            elif is_string_type(var.type):
                symbolic_var = String(name)
            elif isinstance(var.type, MappingType):
                if is_number_type(var.type.type_to):
                    size = var.type.type_to.size
                    symbolic_var = Array(
                        name, BitVecSort(160), BitVecSort(size))
                    # if is_unsigned_number(var.type.type_to):
                    #     x = Const('c_x', BitVecSort(160))
                    #     accompany_constraints.append(ForAll(x, UGE(Select(symbolic_var, x), 0)))
                    # for account in self.accounts:
                    #     symbolic_var = Store(
                    #         symbolic_var, self.accounts[account], BitVecVal(0, size))
                elif is_bool_type(var.type.type_to):
                    symbolic_var = Array(name, BitVecSort(160), BoolSort())
                    for account in self.accounts:
                        symbolic_var = Store(
                            symbolic_var, self.accounts[account], BoolVal(False))
                    for address in self.address_vars:
                        symbolic_var = Store(
                            symbolic_var, symbolic_state_vars[address.name], BoolVal(False))
                elif isinstance(var.type.type_to, MappingType):
                    if is_number_type(var.type.type_to.type_to):
                        size = var.type.type_to.type_to.size
                        symbolic_var = Array(name, BitVecSort(160), ArraySort(
                            BitVecSort(160), BitVecSort(size)))
                        # if is_unsigned_number(var.type.type_to.type_to):
                        #     x, y = Consts('c_x c_y', BitVecSort(160))
                        #     accompany_constraints.append(ForAll([x,y], UGE(Select(Select(symbolic_var, x), y),0)))
                        # for account1 in self.accounts:
                        #     for account2 in self.accounts:
                        #         symbolic_var = Store(symbolic_var, self.accounts[account1],
                        #                              Store(symbolic_var[self.accounts[account1]],
                        #                                    self.accounts[account2],
                        #                                    BitVecVal(0, size)))
                    elif is_bool_type(var.type.type_to.type_to):
                        symbolic_var = Array(name, BitVecSort(
                            160), ArraySort(BitVecSort(160), BoolSort()))
                        for account1 in self.accounts:
                            for account2 in self.accounts:
                                symbolic_var = Store(symbolic_var, self.accounts[account1],
                                                     Store(symbolic_var[self.accounts[account1]],
                                                           self.accounts[account2],
                                                           BoolVal(False)))
                        for address in self.address_vars:
                            for account in self.accounts:
                                symbolic_var = Store(symbolic_var, symbolic_state_vars[address.name],
                                                     Store(symbolic_var[symbolic_state_vars[address.name]],
                                                           self.accounts[account],
                                                           BoolVal(False)))
                                symbolic_var = Store(symbolic_var, self.accounts[account],
                                                     Store(symbolic_var[self.accounts[account]],
                                                           symbolic_state_vars[address.name],
                                                           BoolVal(False)))
        return symbolic_var, accompany_constraints

    def sys_exec(self, state_vars: Dict, local_vars: Dict, nodes_list: List[Node], caller: str, round_count: int):
        constraints = []
        # 存储出现的局部地址变量，便于判断上下文，只存储局部变量
        address_context = {}
        # 存储ref var和tmp var的符号值
        slither_ir_vars = {}
        extra_status = ExtraStatus.success

        for index in range(len(nodes_list)):
            node = nodes_list[index]
            irs = node.irs
            for ir in irs:
                try:
                    symbolic_rvalue = None
                    # 在slither_ir_vars中保存ref变量的索引和tmp变量的符号值
                    if isinstance(ir, Index):
                        lvalue = ir.lvalue
                        if lvalue.name not in slither_ir_vars:
                            slither_ir_vars[lvalue] = []
                        var_left = ir.variable_left
                        var_right = ir.variable_right
                        if is_mapping_type(var_left.type):
                            # 如果是状态变量/局部变量只需要保存var_right索引
                            if isinstance(var_left, (StateVariable, LocalVariable)):
                                symbolic_address = self.__get_address_symbolic_var(var_right, address_context,
                                                                                   state_vars, local_vars,
                                                                                   slither_ir_vars, caller, round_count)
                                slither_ir_vars[lvalue].append(symbolic_address)
                            # 如果是ref变量需要带上其所有索引
                            elif isinstance(var_left, ReferenceVariable):
                                slither_ir_vars[lvalue] = slither_ir_vars[var_left] + slither_ir_vars[lvalue]
                                symbolic_address = self.__get_address_symbolic_var(var_right, address_context,
                                                                                   state_vars, local_vars,
                                                                                   slither_ir_vars, caller, round_count)
                                slither_ir_vars[lvalue].append(symbolic_address)
                    elif isinstance(ir, Assignment):
                        lvalue = ir.lvalue
                        rvalue = ir.rvalue
                        symbolic_rvalue = self.__get_symbolic_rvalue(rvalue, state_vars, local_vars, slither_ir_vars,
                                                                     caller, round_count)
                        if symbolic_rvalue is not None:
                            self.__set_symbolic_lvalue(lvalue, symbolic_rvalue, state_vars, local_vars, slither_ir_vars,
                                                       round_count)
                    elif isinstance(ir, Binary):
                        lvalue = ir.lvalue
                        # rvalue中的var_left和var_right
                        symbolic_var_left = self.__get_symbolic_rvalue(ir.variable_left, state_vars, local_vars,
                                                                       slither_ir_vars, caller, round_count)
                        symbolic_var_right = self.__get_symbolic_rvalue(ir.variable_right, state_vars, local_vars,
                                                                        slither_ir_vars, caller, round_count)
                        if ir.type == BinaryType.POWER:
                            symbolic_rvalue = symbolic_var_left * symbolic_var_left * symbolic_var_left
                        elif ir.type == BinaryType.MULTIPLICATION:
                            symbolic_rvalue = symbolic_var_left * symbolic_var_right
                        elif ir.type == BinaryType.DIVISION:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = symbolic_var_left / symbolic_var_right
                            else:
                                symbolic_rvalue = UDiv(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.MODULO:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = symbolic_var_left % symbolic_var_right
                            else:
                                symbolic_rvalue = URem(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.ADDITION:
                            symbolic_rvalue = symbolic_var_left + symbolic_var_right
                            constraints += [UGE(symbolic_rvalue,symbolic_var_left), UGE(symbolic_rvalue,symbolic_var_right)]
                        elif ir.type == BinaryType.SUBTRACTION:
                            symbolic_rvalue = symbolic_var_left - symbolic_var_right
                            constraints += [UGE(symbolic_var_left, symbolic_rvalue),
                                            UGE(symbolic_var_left, symbolic_var_right)]
                        elif ir.type == BinaryType.LEFT_SHIFT:
                            symbolic_rvalue = symbolic_var_left << symbolic_var_right
                        elif ir.type == BinaryType.RIGHT_SHIFT:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = symbolic_var_left >> symbolic_var_right
                            else:
                                symbolic_rvalue = LShR(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.AND:
                            symbolic_rvalue = symbolic_var_left & symbolic_var_right
                        elif ir.type == BinaryType.CARET:
                            symbolic_rvalue = symbolic_var_left ^ symbolic_var_right
                        elif ir.type == BinaryType.OR:
                            symbolic_rvalue = symbolic_var_left | symbolic_var_right
                        elif ir.type == BinaryType.LESS:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = (
                                        symbolic_var_left < symbolic_var_right)
                            else:
                                symbolic_rvalue = ULT(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.GREATER:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = (
                                        symbolic_var_left > symbolic_var_right)
                            else:
                                symbolic_rvalue = UGT(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.LESS_EQUAL:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = (
                                        symbolic_var_left <= symbolic_var_right)
                            else:
                                symbolic_rvalue = ULE(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.GREATER_EQUAL:
                            if is_signed_number(ir.variable_left.type) or is_signed_number(ir.variable_right.type):
                                symbolic_rvalue = (
                                        symbolic_var_left >= symbolic_var_right)
                            else:
                                symbolic_rvalue = UGE(
                                    symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.NOT_EQUAL:
                            symbolic_rvalue = Not(
                                symbolic_var_left == symbolic_var_right)
                        elif ir.type == BinaryType.EQUAL:
                            symbolic_rvalue = (
                                    symbolic_var_left == symbolic_var_right)
                        elif ir.type == BinaryType.ANDAND:
                            symbolic_rvalue = And(
                                symbolic_var_left, symbolic_var_right)
                        elif ir.type == BinaryType.OROR:
                            symbolic_rvalue = Or(
                                symbolic_var_left, symbolic_var_right)

                        if symbolic_rvalue is not None:
                            self.__set_symbolic_lvalue(lvalue, symbolic_rvalue, state_vars, local_vars, slither_ir_vars,
                                                       round_count)
                    elif isinstance(ir, TypeConversion):
                        var = ir.variable
                        if isinstance(var, Constant):
                            symbolic_rvalue = self.make_symbolic_constant_vars(
                                var_type=ir.type, value=var.value)
                        elif isinstance(var, StateVariable):
                            if is_number_type(ir.type):
                                size = ir.type.size
                                if is_number_type(var.type):
                                    symbolic_rvalue = Extract(
                                        size - 1, 0, state_vars[var.name])
                        elif isinstance(var, LocalVariable):
                            if is_number_type(ir.type):
                                size = ir.type.size
                                if is_number_type(var.type):
                                    symbolic_rvalue = Extract(
                                        size - 1, 0, local_vars[get_local_var_name(var, round_count)])
                        elif isinstance(var, TemporaryVariable):
                            if is_number_type(ir.type):
                                size = ir.type.size
                                if is_number_type(var.type):
                                    symbolic_rvalue = Extract(
                                        size - 1, 0, slither_ir_vars[var])
                        elif isinstance(var, SolidityVariableComposed) or isinstance(var, SolidityVariable):
                            symbolic_rvalue = self.__get_address_symbolic_var(var, address_context, state_vars,
                                                                              local_vars, slither_ir_vars, caller,
                                                                              round_count)
                        if symbolic_rvalue is not None:
                            slither_ir_vars[ir.lvalue] = symbolic_rvalue
                    elif isinstance(ir, Unary):
                        lvalue = ir.lvalue
                        symbolic_var = self.__get_symbolic_rvalue(ir.rvalue, state_vars, local_vars, slither_ir_vars,
                                                                  caller, round_count)
                        if ir.type.name == "BANG":
                            symbolic_rvalue = Not(symbolic_var)
                        elif ir.type.name == "TILD":
                            symbolic_rvalue = ~symbolic_var
                        if symbolic_rvalue is not None:
                            self.__set_symbolic_lvalue(lvalue, symbolic_rvalue, state_vars, local_vars, slither_ir_vars,
                                                       round_count)
                    elif isinstance(ir, Delete):
                        var = ir.variable
                        if isinstance(var, StateVariable):
                            if is_number_type(var.type) or is_address_type(var.type):
                                symbolic_rvalue = BitVecVal(0, var.type.size)
                            elif is_bool_type(var.type):
                                symbolic_rvalue = BoolVal(False)
                            else:
                                symbolic_rvalue = self.make_symbolic_var(var, round_count, state_vars)
                            if symbolic_rvalue is not None:
                                self.__set_symbolic_lvalue(var, symbolic_rvalue, state_vars, local_vars,
                                                           slither_ir_vars, round_count)
                        elif isinstance(var, LocalVariable):
                            symbolic_rvalue = self.make_symbolic_var(var, round_count, state_vars)
                            if symbolic_rvalue is not None:
                                self.__set_symbolic_lvalue(var, symbolic_rvalue, state_vars, local_vars,
                                                           slither_ir_vars, round_count)
                        elif isinstance(var, ReferenceVariable):
                            point_to_type = var.points_to_origin.type.type_to
                            while is_mapping_type(point_to_type):
                                point_to_type = point_to_type.type_to
                            if is_number_type(point_to_type):
                                symbolic_rvalue = BitVecVal(
                                    0, point_to_type.size)
                            elif is_bool_type(point_to_type):
                                symbolic_rvalue = BoolVal(False)
                            if symbolic_rvalue is not None:
                                self.__set_symbolic_lvalue(var, symbolic_rvalue, state_vars, local_vars,
                                                           slither_ir_vars, round_count)
                    elif isinstance(ir, Condition):
                        if node.type != NodeType.IF:
                            continue
                        symbolic_rvalue = self.__get_symbolic_rvalue(ir.value, state_vars, local_vars, slither_ir_vars,
                                                                     caller, round_count)
                        if symbolic_rvalue is not None:
                            if nodes_list[index + 1] == node.son_true:
                                constraints.append(symbolic_rvalue)
                            elif nodes_list[index + 1] == node.son_false:
                                constraints.append(Not(symbolic_rvalue))
                    elif isinstance(ir, LibraryCall):
                        # TODO abs和pow需要进一步处理
                        func_name = ir.function.name
                        if len(ir.arguments) != 2:
                            continue
                        lvalue = ir.lvalue
                        symbolic_var_left = self.__get_symbolic_rvalue(ir.arguments[0], state_vars, local_vars,
                                                                       slither_ir_vars, caller, round_count)
                        symbolic_var_right = self.__get_symbolic_rvalue(ir.arguments[1], state_vars, local_vars,
                                                                        slither_ir_vars, caller, round_count)
                        if fuzz.partial_ratio(func_name, 'add') >= 90:
                            symbolic_rvalue = symbolic_var_left + symbolic_var_right
                            constraints += [UGE(symbolic_rvalue, symbolic_var_left),
                                            UGE(symbolic_rvalue, symbolic_var_right)]
                        elif fuzz.partial_ratio(func_name, 'sub') >= 90:
                            symbolic_rvalue = symbolic_var_left - symbolic_var_right
                            constraints.append(ULT(symbolic_rvalue, symbolic_var_left))
                        elif fuzz.partial_ratio(func_name, 'mul') >= 90:
                            symbolic_rvalue = symbolic_var_left * symbolic_var_right
                            if is_signed_number(ir.arguments[0].type) or is_signed_number(ir.arguments[1].type):
                                constraints.append(symbolic_rvalue / symbolic_var_left == symbolic_var_right)
                            else:
                                constraints.append(UDiv(symbolic_rvalue, symbolic_var_left) == symbolic_var_right)
                        elif fuzz.partial_ratio(func_name, 'div') >= 90:
                            constraints.append(Not(symbolic_var_right == 0))
                            if is_signed_number(ir.arguments[0].type) or is_signed_number(ir.arguments[1].type):
                                symbolic_rvalue = symbolic_var_left / symbolic_var_right
                            else:
                                symbolic_rvalue = UDiv(symbolic_var_left, symbolic_var_right)
                        elif fuzz.partial_ratio(func_name, 'mod') >= 90:
                            constraints.append(Not(symbolic_var_right == 0))
                            if is_signed_number(ir.arguments[0].type) or is_signed_number(ir.arguments[1].type):
                                symbolic_rvalue = symbolic_var_left % symbolic_var_right
                            else:
                                symbolic_rvalue = URem(symbolic_var_left, symbolic_var_right)
                        elif fuzz.partial_ratio(func_name, 'pow') >= 90 or fuzz.partial_ratio(func_name, 'exp') >= 90:
                            symbolic_rvalue = symbolic_var_left * symbolic_var_left * symbolic_var_left
                        elif fuzz.partial_ratio(func_name, 'abs') >= 90:
                            pass

                        if symbolic_rvalue is not None:
                            self.__set_symbolic_lvalue(lvalue, symbolic_rvalue, state_vars, local_vars, slither_ir_vars,
                                                       round_count)
                    elif isinstance(ir, SolidityCall):
                        name = ir.function.name
                        if name.startswith('require'):
                            constraints.append(
                                self.__get_symbolic_rvalue(ir.arguments[0], state_vars, local_vars, slither_ir_vars,
                                                           caller, round_count))
                        elif name.startswith('assert'):
                            constraints.append(
                                self.__get_symbolic_rvalue(ir.arguments[0], state_vars, local_vars, slither_ir_vars,
                                                           caller, round_count))
                        elif name.startswith('revert'):
                            extra_status = ExtraStatus.revert
                            return constraints, slither_ir_vars, extra_status
                        elif name.startswith(('selfdestruct', 'suicide')):
                            extra_status = ExtraStatus.selfdestruct
                    elif isinstance(ir, EventCall):
                        if ir.name.startswith('AutoGenerated_mapping'):
                            state_vars[ir.arguments[0].name] = state_vars[ir.arguments[1].name]
                except Exception as e:
                    # print(e)
                    continue

        # TODO internal_call中可能会把不同的地址局部变量映射到同一个account，这样地址变量的约束可能会产生冲突导致无意义的UNSAT
        # for address in address_context:
        #     constraints.append(address_context[address] == local_vars[address])
        return constraints, slither_ir_vars, extra_status

    def __get_address_symbolic_var(self, var, address_context: Dict, state_vars: Dict, local_vars: Dict,
                                   slither_ir_vars: Dict, caller: str, round_count):
        if isinstance(var, SolidityVariableComposed):
            if var.name == 'msg.sender':
                if caller == 'user':
                    address_context[var.name] = self.accounts['user0']
                    return self.accounts['user0']
                elif caller == 'owner':
                    address_context[var.name] = state_vars[self.owner.name]
                    return state_vars[self.owner.name]
        if isinstance(var, SolidityVariable):
            if var.name == 'this':
                return self.accounts['this']
        if isinstance(var, StateVariable):
            return state_vars[var.name]
        if isinstance(var, LocalVariable):
            local_var_name = get_local_var_name(var, round_count)
            # 常数地址
            if isinstance(local_vars[local_var_name], BitVecNumRef):
                return local_vars[local_var_name]
            if (str(local_vars[local_var_name]) in self.accounts) or (
                    str(local_vars[local_var_name]) == str(state_vars[self.owner.name])):
                return local_vars[local_var_name]
            for state_var in state_vars:
                if str(local_vars[local_var_name]) == str(state_vars[state_var]):
                    return local_vars[local_var_name]
        if isinstance(var, TemporaryVariable):
            if isinstance(slither_ir_vars[var], BitVecNumRef):
                return slither_ir_vars[var]
        if var.name in address_context:
            return address_context[var.name]
        else:
            current_address_num = len(address_context)
            if current_address_num >= len(self.accounts) - 1:
                raise TooManyUsersError(
                    var.function.name, var.function.contract.name)
            user_name = f'user{str(current_address_num)}'
            address_context[var.name] = self.accounts[user_name]
            return address_context[var.name]

    def __set_symbolic_lvalue(self, lvalue, symbolic_rvalue, state_vars: Dict, local_vars: Dict, slither_ir_vars: Dict,
                              round_count):
        if isinstance(lvalue, StateVariable):
            state_vars[lvalue.name] = symbolic_rvalue
        elif isinstance(lvalue, LocalVariable):
            local_vars[get_local_var_name(lvalue, round_count)] = symbolic_rvalue
        elif isinstance(lvalue, TemporaryVariable):
            slither_ir_vars[lvalue] = symbolic_rvalue
        elif isinstance(lvalue, ReferenceVariable):
            self.store_ref_symbolic_var(lvalue, symbolic_rvalue, state_vars, local_vars, slither_ir_vars, round_count)

    def __get_symbolic_rvalue(self, rvalue, state_vars: Dict, local_vars: Dict, slither_ir_vars: Dict, caller,
                              round_count):
        if isinstance(rvalue, TemporaryVariable):
            return slither_ir_vars[rvalue]
        elif isinstance(rvalue, ReferenceVariable):
            if isinstance(rvalue.points_to_origin, StateVariable):
                return self.select_ref_symbolic_var(rvalue, state_vars[
                    rvalue.points_to_origin.name], slither_ir_vars)
            elif isinstance(rvalue.points_to_origin, LocalVariable):
                return self.select_ref_symbolic_var(rvalue, local_vars[
                    get_local_var_name(rvalue.points_to_origin, round_count)], slither_ir_vars)
        elif isinstance(rvalue, LocalVariable):
            return local_vars[get_local_var_name(rvalue, round_count)]
        elif isinstance(rvalue, StateVariable):
            return state_vars[rvalue.name]
        elif isinstance(rvalue, Constant):
            return self.make_symbolic_constant_vars(var_type=rvalue.type, value=rvalue.value)
        elif isinstance(rvalue, SolidityVariableComposed):
            if rvalue.name == 'msg.sender':
                if caller == 'user':
                    return self.accounts['user0']
                elif caller == 'owner':
                    return state_vars[self.owner.name]
            elif rvalue.name == 'msg.value':
                return BitVec(f"{caller}_value_{str(round_count)}", 256)
        elif isinstance(rvalue, SolidityVariable):
            if rvalue.name == 'this':
                return self.accounts['this']
            elif rvalue.name == 'now':
                return BitVec(f'now_{str(round_count)}', 256)

    @staticmethod
    def select_ref_symbolic_var(ref_var: ReferenceVariable, ref_origin: Array, slither_ir_vars: Dict):
        ref_len = len(slither_ir_vars[ref_var])
        if ref_len == 1:
            ref_to = slither_ir_vars[ref_var][0]
            return Select(ref_origin, ref_to)
        elif ref_len == 2:
            ref_to_first = slither_ir_vars[ref_var][0]
            ref_to_second = slither_ir_vars[ref_var][1]
            return Select(Select(ref_origin, ref_to_first), ref_to_second)
        else:
            raise TooComplexMapping(
                ref_var.function.contract.name, ref_var.function.name)

    @staticmethod
    def store_ref_symbolic_var(lvalue: ReferenceVariable, symbolic_rvalue, state_vars: Dict, local_vars: Dict,
                               slither_ir_vars: Dict, round_count):
        ref_len = len(slither_ir_vars[lvalue])
        if ref_len == 1:
            ref_to = slither_ir_vars[lvalue][0]
            if isinstance(lvalue.points_to_origin, StateVariable):
                state_vars[lvalue.points_to_origin.name] = Store(state_vars[lvalue.points_to_origin.name], ref_to,
                                                                 symbolic_rvalue)
            elif isinstance(lvalue.points_to_origin, LocalVariable):
                local_vars[get_local_var_name(lvalue.points_to_origin, round_count)] = Store(
                    local_vars[get_local_var_name(
                        lvalue.points_to_origin, round_count)], ref_to, symbolic_rvalue)
        elif ref_len == 2:
            ref_to_first = slither_ir_vars[lvalue][0]
            ref_to_second = slither_ir_vars[lvalue][1]
            if isinstance(lvalue.points_to_origin, StateVariable):
                symbolic_lvalue = state_vars[lvalue.points_to_origin.name]
                state_vars[lvalue.points_to_origin.name] = Store(symbolic_lvalue, ref_to_first,
                                                                 Store(symbolic_lvalue[ref_to_first], ref_to_second,
                                                                       symbolic_rvalue))
            elif isinstance(lvalue.points_to_origin, LocalVariable):
                symbolic_lvalue = local_vars[get_local_var_name(
                    lvalue.points_to_origin, round_count)]
                local_vars[get_local_var_name(lvalue.points_to_origin, round_count)] = Store(symbolic_lvalue,
                                                                                             ref_to_first, Store(
                        symbolic_lvalue[ref_to_first], ref_to_second, symbolic_rvalue))
        else:
            raise TooManyUsersError(lvalue.contract.name, lvalue.function.name)

    def update_initiate_state(self, round_num, contract: Contract, delta_summary, init_symbolic_state_vars):
        v_initiate_state = copy.deepcopy(init_symbolic_state_vars)
        for state_var_name in delta_summary:
            if delta_summary[state_var_name]['sign'] == 0 or state_var_name == 'contract':
                continue
            state_var = contract.get_state_variable_from_name(state_var_name)
            if is_address_type(state_var.type):
                if delta_summary[state_var_name]['value']['can_be_specific'] is not None:
                    v_initiate_state[state_var_name] = delta_summary[state_var_name]['value']['can_be_specific']
                elif delta_summary[state_var_name]['value']['can_be_different']:
                    in_accounts = False
                    for account in self.accounts:
                        if str(v_initiate_state[state_var_name]) == str(self.accounts[account]):
                            in_accounts = True
                            break
                    if not in_accounts:
                        v_initiate_state[state_var_name] = BitVec(f"{state_var_name}_{str(round_num)}", 160)
            elif is_bool_type(state_var.type):
                v_initiate_state[state_var_name] = self.__update_bool_var(delta_summary[state_var_name]['value'],
                                                                          init_symbolic_state_vars[state_var_name])
            elif is_mapping_type(state_var.type):
                mapping_symbolic_var = init_symbolic_state_vars[state_var_name]
                if is_bool_type(state_var.type.type_to):
                    for address in delta_summary[state_var_name]['value']:
                        symbolic_address = self.get_symbolic_address(address, init_symbolic_state_vars)
                        if symbolic_address is None: continue
                        symbolic_address_value = self.__update_bool_var(delta_summary[state_var_name]['value'][address],
                                                                        Select(mapping_symbolic_var, symbolic_address))
                        if symbolic_address_value is None:
                            continue
                        new_symbolic_var = Store(v_initiate_state[state_var_name], symbolic_address,
                                                 symbolic_address_value)
                        v_initiate_state[state_var_name] = new_symbolic_var
                elif isinstance(state_var.type.type_to, MappingType):
                    if is_bool_type(state_var.type.type_to.type_to):
                        for address_pair in delta_summary[state_var_name]['value']:
                            symbolic_address_first = self.get_symbolic_address(address_pair[0],
                                                                               init_symbolic_state_vars)
                            symbolic_address_second = self.get_symbolic_address(address_pair[1],
                                                                                init_symbolic_state_vars)
                            if symbolic_address_first is None or symbolic_address_second is None:
                                continue
                            symbolic_address_value = self.__update_bool_var(delta_summary[state_var_name]['value'][
                                                                                address_pair], Select(
                                Select(mapping_symbolic_var, symbolic_address_first), symbolic_address_second))
                            new_symbolic_var = Store(v_initiate_state[state_var_name], symbolic_address_first,
                                                     Store(v_initiate_state[state_var_name][symbolic_address_first],
                                                           symbolic_address_second, symbolic_address_value))
                            v_initiate_state[state_var_name] = new_symbolic_var
        return v_initiate_state

    @staticmethod
    def __update_bool_var(summary, origin_symbolic_var):
        if summary['can_be_true'] and summary['can_be_false']:
            # return BoolVal(True)
            return Not(origin_symbolic_var)
        elif summary['can_be_true'] and not summary['can_be_false']:
            return BoolVal(True)
        elif not summary['can_be_true'] and summary['can_be_false']:
            return BoolVal(False)
        else:
            return origin_symbolic_var

    def get_symbolic_address(self, address: str, init_symbolic_state_vars):
        if address in self.accounts:
            symbolic_address = self.accounts[address]
        elif address in init_symbolic_state_vars:
            symbolic_address = init_symbolic_state_vars[address]
        else:
            if address.isnumeric():
                symbolic_address = BitVecVal(int(address), 160)
            else:
                symbolic_address = None
        return symbolic_address
