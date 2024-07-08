from typing import List

from slither.core.cfg.node import Node
from slither.core.solidity_types import ElementaryType, Type, MappingType, ArrayType, UserDefinedType
from slither.core.variables.variable import Variable


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


def is_bytes32_type(var_type: Type) -> bool:
    if isinstance(var_type, ElementaryType):
        if var_type.name == "bytes32":
            return True
    return False


def is_signed_number(var_type: Type) -> bool:
    if is_number_type(var_type):
        if var_type.name.startswith('int'):
            return True
    return False


def is_unsigned_number(var_type: Type) -> bool:
    if is_number_type(var_type):
        if var_type.name.startswith('uint'):
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


def is_user_defined_type(var_type: Type) -> bool:
    return isinstance(var_type, UserDefinedType)


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
