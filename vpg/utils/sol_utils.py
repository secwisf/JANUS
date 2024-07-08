import os
import re
from typing import List

from rapidfuzz import fuzz
from slither.core.cfg.node import Node
from slither.core.solidity_types import ElementaryType, Type, MappingType, ArrayType
from slither.core.variables.variable import Variable


def get_mapping_vars_for_train(top_contracts):
    mapping_vars = []
    for tc in top_contracts:
        # if not tc.is_possible_token: continue
        state_vars = tc.state_variables
        balance_func_sv = set()
        for func in tc.all_functions_called:
            if fuzz.partial_ratio(func.name.lower(), "balanceof") >= 85:
                balance_func_sv = balance_func_sv.union(set(func.state_variables_read))
        for sv in state_vars:
            if not (is_mapping_type(sv.type) and is_address_type(sv.type.type_from)): continue
            if is_number_type(sv.type.type_to):
                if len(balance_func_sv) != 0 and sv in balance_func_sv:
                    mapping_vars.append([sv.canonical_name, 1])
                else:
                    if fuzz.partial_ratio(sv.name.lower(), "balance") >= 85:
                        mapping_vars.append([sv.canonical_name, 1])
                    else:
                        mapping_vars.append([sv.canonical_name, 0])
            else:
                mapping_vars.append([sv.canonical_name, 0])
    return mapping_vars


def get_mapping_vars_for_test(top_contracts):
    mapping_vars = []
    for tc in top_contracts:
        # if not tc.is_possible_token: continue
        state_vars = tc.state_variables
        balance_func_sv = set()
        for func in tc.all_functions_called:
            if fuzz.partial_ratio(func.name.lower(), "balanceof") >= 85:
                balance_func_sv = balance_func_sv.union(set(func.state_variables_read))
        for sv in state_vars:
            if not (is_mapping_type(sv.type) and is_address_type(sv.type.type_from)): continue
            if is_number_type(sv.type.type_to):
                if len(balance_func_sv) != 0 and sv in balance_func_sv:
                    mapping_vars.append([sv.canonical_name, 1])
                else:
                    if fuzz.partial_ratio(sv.name.lower(), "balance") >= 85:
                        mapping_vars.append([sv.canonical_name, 1])
                    else:
                        mapping_vars.append([sv.canonical_name, 0])
            else:
                mapping_vars.append([sv.canonical_name, 0])
    return mapping_vars


def get_local_var_name(var, round_num: int) -> str:
    return f'{var.function.name}_{var.name}_{str(round_num)}'


def is_address_type(var_type: Type) -> bool:
    if isinstance(var_type, ElementaryType):
        if var_type.name == 'address':
            return True
    return False


def is_number_type(var_type: Type) -> bool:
    if isinstance(var_type, ElementaryType):
        if var_type.name.startswith('uint') or var_type.name.startswith('int'):
            return True
    return False


def is_bytes_type(var_type: Type) -> bool:
    if isinstance(var_type, ElementaryType):
        if var_type.name == "bytes":
            return True
    return False


def is_signed_number(var_type: Type) -> bool:
    if is_number_type(var_type):
        if var_type.name.startswith('int'):
            return True
    return False


def is_bool_type(var_type: Type) -> bool:
    if isinstance(var_type, ElementaryType):
        if var_type.name == 'bool':
            return True
    return False


def is_mapping_type(var_type: Type) -> bool:
    return isinstance(var_type, MappingType)


def is_array_type(var_type: Type) -> bool:
    return isinstance(var_type, ArrayType)


def is_mapping_mapping_type(var_type: Type) -> bool:
    return isinstance(var_type, MappingType) and is_mapping_type(var_type.type_to)


def is_string_type(var_type: Type) -> bool:
    if isinstance(var_type, ElementaryType):
        if var_type.name == 'string':
            return True
    return False


def get_path_hash(path: List[Node]):
    content = ""
    for node in path:
        content += str(node)
    return hash(content)


def is_reading_in_conditional_node(nodes_list: List[Node], variable: Variable):
    variables_reads = [n.variables_read for n in nodes_list if n.contains_if()]
    variables_read = [item for sublist in variables_reads for item in sublist]
    return variable in variables_read


def get_related_variables(var: Variable, path_node_list: List[Node]):
    result = []
    for node in path_node_list:
        if var in node.variables_written:
            result += node.variables_read
    return result


def all_path(dirname, filter_list):
    result = []  # 所有的文件
    # filter = [".wav"]
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            ext = os.path.splitext(apath)[1]
            if ext in filter_list:
                result.append(apath)
    return result


def get_higher_version(ver1: str, ver2: str) -> str:
    ver1_list = [int(s) for s in ver1.split('.')]
    ver2_list = [int(s) for s in ver2.split('.')]
    for i in range(1, 3):
        if ver1_list[i] > ver2_list[i]:
            return ver1
    return ver2


def get_solc_version(path) -> str:
    max_version = '0.4.0'
    with open(path, 'r') as fr:
        content_list = fr.readlines()
    for line in content_list:
        if 'pragma solidity' not in line:
            continue
        curr_ver = re.findall(r'(\d+\.\d+\.\d+)', line)[0]
        max_version = get_higher_version(max_version, curr_ver)
    return max_version
