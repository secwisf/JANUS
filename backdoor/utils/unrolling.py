import copy
import os
import re
import traceback
from solidity_parser import parse_file

DEFAULT_LEN = 3
MAX_LEN = 20

class SolParser:
    def __init__(self):
        self.file = ''
        self.origin_sol = []
        self.new_sol = []

    def set_path(self, file):
        self.file = file
        with open(file, 'r') as fr:
            self.origin_sol = fr.read().split("\n")


def get_line_number(loc: dict) -> list:
    lines = [loc['start']['line'], loc['end']['line']]
    return lines


def update_new_contracts_with_lines(lines: list, sol: SolParser):
    start = lines[0]
    end = lines[1]
    for i in range(start - 1, end):
        sol.new_sol.append(sol.origin_sol[i])


def check_del_array(lines: list, sol: SolParser, del_state_list: list, del_local_list: list) -> bool:
    content_list = sol.origin_sol[lines[0] - 1:lines[1]]
    del_array_list = del_state_list + del_local_list
    for line in content_list:
        for da in del_array_list:
            pattern = r'{}\s*\[.*\]'.format(da)
            r = re.search(pattern, line)
            if r:
                return True
    return False


def parse_var_type(node):
    node_type = ""
    try:
        if 'typeName' in node:
            if node['typeName']['type'] == 'ElementaryTypeName':
                node_type = node['typeName']['name']
            elif node['typeName']['type'] == 'ArrayTypeName':
                node_type = node['typeName']['baseTypeName']['name'] + "[]"
            elif node['typeName']['type'] == 'UserDefinedTypeName':
                node_type = node['typeName']['namePath']
            elif node['typeName']['type'] == 'Mapping':
                key_type = parse_var_type(node['typeName']['keyType'])
                value_type = parse_var_type(node['typeName']['valueType'])
                node_type = 'mapping(%s=>%s)' % (key_type, value_type)
        else:
            if node['type'] == 'ElementaryTypeName':
                node_type = node['name']
            elif node['type'] == 'ArrayTypeName':
                node_type = node['baseTypeName']['name'] + "[]"
            elif node['type'] == 'UserDefinedTypeName':
                node_type = node['namePath']
            elif node['type'] == 'Mapping':
                key_type = parse_var_type(node['keyType'])
                value_type = parse_var_type(node['valueType'])
                node_type = 'mapping(%s=>%s)' % (key_type, value_type)
    except:
        pass
    return node_type


# 只用来处理数组变量的长度，定长返回数字，不定长地址数组返回-1，
def cal_var_length(node) -> int:
    if node['type'] == 'Parameter':
        var = node
    else:
        var = node['variables'][0]
    if 'initialValue' in node:
        initial = node['initialValue']
    else:
        initial = None
    if var is None:
        return -1
    # uint[3] amounts
    if 'typeName' in var and 'length' in var['typeName']:
        length = var['typeName']['length']
        if length is not None and length['type'] == 'NumberLiteral':
            return length['number']
    # uint[] amounts_new = new uint[](3);
    if initial is not None:
        if 'arguments' in initial and initial['arguments'] is not None and len(initial['arguments']) != 0:
            arg = initial['arguments'][0]
            if arg['type'] is not None and arg['type'] == 'NumberLiteral':
                return arg['number']
    return -1


def deal_with_vars(node):
    var_list = []
    # if node['type'] == 'StructDefinition':
    #     tmp = "struct " + node['name']
    #     var_list.append(tmp)
    if node['type'] == 'StateVariableDeclaration':
        if node['variables'] is not None:
            for var in node['variables']:
                if var['type'] == 'VariableDeclaration':
                    var_name = var['name']
                    var_typeName = parse_var_type(var)
                    if "[]" in var_typeName:
                        length = cal_var_length(node)
                        var_list.append(("%s %s" % (var_typeName, var_name), node, length))
                    else:
                        var_list.append(("%s %s" % (var_typeName, var_name), node))
    return var_list


def parse_local_declarations(node, declarations: list):
    if node is None: return
    if 'type' not in node: return
    if node['type'] == 'IfStatement':
        parse_local_declarations(node['condition'], declarations)
        parse_local_declarations(node['TrueBody'], declarations)
        parse_local_declarations(node['FalseBody'], declarations)
    elif node['type'] == 'BinaryOperation':
        parse_local_declarations(node['left'], declarations)
        parse_local_declarations(node['right'], declarations)
    elif node['type'] == 'UnaryOperation':
        parse_local_declarations(node['subExpression'], declarations)
    elif node['type'] == 'Block':
        if node['statements'] is not None:
            for s in node['statements']:
                parse_local_declarations(s, declarations)
    elif node['type'] == 'WhileStatement':
        parse_local_declarations(node['condition'], declarations)
        parse_local_declarations(node['body'], declarations)
    elif node['type'] == 'ForStatement':
        parse_local_declarations(node['conditionExpression'], declarations)
        parse_local_declarations(node['loopExpression'], declarations)
        parse_local_declarations(node['initExpression'], declarations)
        parse_local_declarations(node['body'], declarations)
    elif node['type'] == 'VariableDeclarationStatement':
        if node['variables'] is not None:
            for v in node['variables']:
                if v['type'] == 'VariableDeclaration':
                    v_type = parse_var_type(v)
                    if "[]" in v_type:
                        length = cal_var_length(node)
                        declarations.append(("%s %s" % (v_type, v['name']), length))
    elif node['type'] == 'ExpressionStatement':
        parse_local_declarations(node['expression'], declarations)
    elif node['type'] == 'TupleExpression':
        for c in node['components']:
            parse_local_declarations(c, declarations)


def deal_num(s):
    ss = str(s.group())
    num = re.findall(r'\[\s*(.*)\s*\]', ss)[0]
    ss = re.sub(r'\[\s*(.*)\s*\]', num, ss)
    ss = ss.strip('\r\n')
    return ss


def transform(type: str, sol: SolParser, lines: list, **kwargs):
    new_content = []
    if type == 'array':
        info_dict = kwargs['info']
        state_array_length_dict = kwargs['state']
        keys = list(state_array_length_dict.keys())
        for line in range(lines[0] - 1, lines[1]):
            var_name = info_dict['name']
            arr_len = info_dict['len']
            origin_line_content = sol.origin_sol[line]
            origin_part = origin_line_content.split("=")[0]
            origin_part = origin_part.split(";")[0].strip('\r\n')
            if arr_len <= MAX_LEN:
                for count in range(arr_len):
                    new_content_line = "{};".format(re.sub('\[\s*[0-9]*\s*\]', '', origin_part))
                    new_content_line = new_content_line.replace(var_name, "{}{}".format(var_name, str(count)))
                    for key in keys:
                        new_content_line = re.sub(r'{}.length'.format(key), str(state_array_length_dict[key]), new_content_line)
                    new_content.append(new_content_line)
    elif type == 'for':
        terminal = kwargs['terminal']
        index_var = kwargs['index']
        # for(){}只考虑大括号里body的内容
        content = "\n".join(sol.origin_sol[lines[0]-1:lines[1]])
        for_content_list = re.findall(r'for\s*\([^\(]*\)\s*\{([^\{]*)\}\s*$', content, re.S)
        content = for_content_list[0].strip('\r\n')
        state_array_length_dict = kwargs['state']
        local_array_length_dict = kwargs['local']
        keys = list(state_array_length_dict.keys()) + list(local_array_length_dict.keys())
        for count in range(terminal):
            new_content_line = re.sub(r'\[\s*{}\s*\]'.format(index_var), str(count), content, re.S)
            for key in keys:
                if key in state_array_length_dict:
                    if int(state_array_length_dict[key]) <= MAX_LEN:
                        pattern = r'{}\s*\[\s*([0-9][0-9]*)\s*\]'.format(key)
                        new_content_line = re.sub(pattern, deal_num, new_content_line)
                    new_content_line = re.sub(r'{}.length'.format(key), str(state_array_length_dict[key]),
                                              new_content_line)
                else:
                    if int(local_array_length_dict[key]) <= MAX_LEN:
                        pattern = r'{}\s*\[\s*[0-9][0-9]*\s*\]'.format(key)
                        new_content_line = re.sub(pattern, deal_num, new_content_line)
                    new_content_line = re.sub(r'{}.length'.format(key), str(local_array_length_dict[key]),
                                              new_content_line)
            new_content.append(new_content_line)

    elif type == 'func':
        state_array_length_dict = kwargs['state']
        local_array_length_dict = kwargs['local']
        keys = list(state_array_length_dict.keys()) + list(local_array_length_dict.keys())
        for line in range(lines[0] - 1, lines[1]):
            new_content_line = sol.origin_sol[line]
            for key in keys:
                if key in state_array_length_dict:
                    if int(state_array_length_dict[key]) <= MAX_LEN:
                        pattern = r'{}\s*\[\s*[0-9][0-9]*\s*\]'.format(key)
                        new_content_line = re.sub(pattern, deal_num, new_content_line)
                    new_content_line = re.sub(r'{}.length'.format(key), str(state_array_length_dict[key]), new_content_line)
                else:
                    if int(local_array_length_dict[key]) <= MAX_LEN:
                        pattern = r'{}\s*\[\s*[0-9][0-9]*\s*\]'.format(key)
                        new_content_line = re.sub(pattern, deal_num, new_content_line)
                    new_content_line = re.sub(r'{}.length'.format(key), str(local_array_length_dict[key]),
                                              new_content_line)
            new_content.append(new_content_line)
    elif type == 'other':
        state_array_length_dict = kwargs['state']
        keys = list(state_array_length_dict.keys())
        for line in range(lines[0] - 1, lines[1]):
            new_content_line = sol.origin_sol[line]
            for key in keys:
                if int(state_array_length_dict[key]) <= MAX_LEN:
                    pattern = r'{}\s*\[\s*[0-9][0-9]*\s*\]'.format(key)
                    new_content_line = re.sub(pattern, deal_num, new_content_line)
                    new_content_line = re.sub(r'{}.length'.format(key), str(state_array_length_dict[key]), new_content_line)
            new_content.append(new_content_line)
    return new_content


def process_state_var(node, state_array_lenth_dict, del_array_list: list):
    vars = deal_with_vars(node)
    for var in vars:
        type, name = var[0].split(" ")
        if "[]" in type:
            length = int(var[-1])
            arr_type = type[:-2]
            if length == -1 and arr_type != 'address':
                del_array_list.append(name)
                continue
            elif length == -1 and arr_type == 'address':
                length = DEFAULT_LEN
            # length为定值
            state_array_lenth_dict.setdefault(name, length)
            return arr_type, name
    return "", ""


def process_local_var(declarations: list, local_array_lenth_dict: dict, del_local_list: list):
    for t in declarations:
        type, name = t[0].split(" ")
        length = int(t[-1])
        arr_type = type[:-2]
        if length == -1 and arr_type != 'address':
            del_local_list.append(name)
            continue
        elif length == -1 and arr_type == 'address':
            length = DEFAULT_LEN

        local_array_lenth_dict.setdefault(name, length)


def process_contract(node, sol: SolParser):
    state_array_length_dict = {}
    del_state_list = []
    # 生成的内容，返回
    new_content_list = []
    # lines = [node['loc']['start']['line'], node['loc']['start']['line']]
    # update_new_contracts_with_lines(lines, sol)
    # if '{' not in sol.origin_sol[node['loc']['start']['line']]:
    #     sol.new_sol.append('{')
    if 'subNodes' in node:
        for sub in node['subNodes']:
            cache_sol = copy.copy(new_content_list)
            try:
                # 处理全局数组，按节点单个处理
                if sub['type'] == 'StateVariableDeclaration':
                    arr_type, name = process_state_var(sub, state_array_length_dict, del_state_list)
                    if name == "":
                        lines = get_line_number(sub['loc'])
                        new_content_list += sol.origin_sol[lines[0]-1:lines[1]]
                        continue
                    length = state_array_length_dict[name]
                    info_dict = {'name': name, 'type': arr_type, 'len': length}
                    new_content_list += transform('array', sol, get_line_number(sub['loc']), info=info_dict, state=state_array_length_dict)
                elif sub['type'] == 'FunctionDefinition':
                    body = sub['body']
                    if body is None or len(body) == 0:
                        # 没有转换的意义，直接复制
                        lines = get_line_number(sub['loc'])
                        new_content_list += sol.origin_sol[lines[0]-1:lines[1]]
                        continue
                    local_array_length_dict = {}
                    # [(type+name, length)]，只保存数组局部变量
                    declarations = []
                    del_local_list = []

                    # 处理函数，得到函数头
                    params = sub['parameters']
                    other_loop = False
                    lines = get_line_number(sub['loc'])
                    func_content = "\n".join(sol.origin_sol[lines[0]-1:lines[1]])
                    func_content_list = re.findall(r'([function|constructor][^\{]*)\{.*\}\s*$', func_content, re.S)

                    new_content = func_content_list[0].strip('\r\n')
                    for param in params['parameters']:
                        var_typeName = parse_var_type(param)
                        if "[]" in var_typeName:
                            name = param['name']
                            length = int(cal_var_length(param))
                            arr_type = var_typeName[:-2]
                            if length == -1 and arr_type != 'address':
                                other_loop = True
                                break
                            elif length == -1 and arr_type == 'address':
                                if other_loop:
                                    break
                                else:
                                    length = DEFAULT_LEN
                            local_array_length_dict.setdefault(name, length)
                            sub_content = ""
                            if length <= MAX_LEN:
                                for count in range(length):
                                    sub_content += "{} {}{},".format(arr_type, name, str(count))
                                sub_content = sub_content[:-1]
                                pattern = r'{type}\[([0-9][0-9]*)*\]\s*{name}'.format(type=arr_type, name=name)
                                new_content = re.sub(pattern, sub_content, new_content, re.S)
                    if other_loop:
                        #  不处理当前sub节点（函数）
                        continue
                    else:
                        # 处理函数头
                        new_content_list.append(new_content)
                        if '{' not in new_content:
                            new_content_list.append('{')

                    # 处理局部数组，扫描函数节点批量处理
                    statements = body['statements']
                    if statements is not None:
                        for statement in statements:
                            parse_local_declarations(statement, declarations)
                        process_local_var(declarations, local_array_length_dict, del_local_list)

                    # 使用了长度不固定的非地址数组，删除该函数
                    r = check_del_array(get_line_number(body['loc']), sol, del_state_list, del_local_list)
                    if r:
                        new_content_list = cache_sol
                        continue

                    # 逐条语句分析，同时判断是否存在其他类型的循环，存在则删除该函数
                    if statements is not None:
                        new_content = []
                        other_loop = False
                        for statement in statements:
                            # 存在while类型的不处理，直接删除
                            if statement['type'] == 'WhileStatement':
                                other_loop = True
                                break
                            elif statement['type'] == 'ForStatement':
                                index_var = statement['initExpression']['variables'][0]['name']
                                condition = statement['conditionExpression']
                                if condition['type'] == 'BinaryOperation' and condition['right'][
                                    'type'] == 'MemberAccess' and condition['right']['memberName'] == 'length' and \
                                        condition['right']['expression']['type'] == 'Identifier':
                                    terminal = condition['right']['expression']['name']
                                    # 循环长度未知，删除函数
                                    if terminal not in state_array_length_dict and terminal not in local_array_length_dict:
                                        other_loop = True
                                        break
                                    else:
                                        if terminal in state_array_length_dict:
                                            terminal_num = state_array_length_dict[terminal]
                                            if terminal_num > MAX_LEN:
                                                other_loop = True
                                                break
                                            tmp_content = transform('for', sol, get_line_number(statement['loc']),
                                                                    terminal=terminal_num,
                                                                    state=state_array_length_dict,
                                                                    local=local_array_length_dict, index=index_var)
                                        elif terminal in local_array_length_dict:
                                            terminal_num = local_array_length_dict[terminal]
                                            if terminal_num > MAX_LEN:
                                                other_loop = True
                                                break
                                            tmp_content = transform('for', sol, get_line_number(statement['loc']),
                                                                    terminal=terminal_num,
                                                                    state=state_array_length_dict,
                                                                    local=local_array_length_dict, index=index_var)
                                        new_content += tmp_content
                                elif condition['type'] == 'BinaryOperation' and condition['right'][
                                    'type'] == 'NumberLiteral':
                                    terminal = int(condition['right']['number'])
                                    if terminal > MAX_LEN:
                                        other_loop = True
                                        break
                                    tmp_content = transform('for', sol, get_line_number(statement['loc']),
                                                            terminal=terminal,
                                                            state=state_array_length_dict,
                                                            local=local_array_length_dict, index=index_var)
                                    new_content += tmp_content
                                else:
                                   other_loop = True
                                   break
                            elif statement['type'] == 'VariableDeclarationStatement':
                                var = statement['variables'][0]
                                var_name = var['name']
                                var_type = parse_var_type(var)
                                if '[]' in var_type:
                                    info_dict = {'name':var_name, 'len':local_array_length_dict[var_name]}
                                    tmp_content = transform('array', sol, get_line_number(var['loc']), info=info_dict, state=local_array_length_dict)
                                else:
                                    tmp_content = transform('func', sol, get_line_number(statement['loc']),
                                                            state=state_array_length_dict,
                                                            local=local_array_length_dict)
                                new_content += tmp_content
                            else:
                                tmp_content = transform('func', sol, get_line_number(statement['loc']),
                                                        state=state_array_length_dict, local=local_array_length_dict)
                                new_content += tmp_content
                        if other_loop:
                            new_content_list = new_content_list[:-2]
                        else:
                            new_content_list += new_content
                            # 函数结束的大括号
                            new_content_list.append('}')
                else:
                    new_content_list += transform('other', sol, get_line_number(sub['loc']), state=state_array_length_dict)
            except Exception:
                print(traceback.format_exc())
                new_content_list = cache_sol
    return new_content_list


def deal_with_statements(sol: SolParser):
    soure_unit = parse_file(sol.file, loc=True)['children']
    for index in range(len(soure_unit)):
        node = soure_unit[index]
        if node is None:
            continue
        elif 'type' not in node:
            continue
        elif node['type'] == 'ContractDefinition':
            if 'kind' in node:
                if node['kind'] == 'contract':
                    # contract的两个大括号手工处理
                    lines = get_line_number(node['loc'])
                    contract_content = "\n".join(sol.origin_sol[lines[0]-1:lines[1]])
                    contract_content_list = re.findall(r'(contract[^\{]*)\s*\{.*\}\s*$',contract_content,re.S)
                    contract_name = contract_content_list[0].strip('\r\n') + ' {'
                    sol.new_sol.append(contract_name)
                    new_content_list = process_contract(node, sol)
                    # 合约结束的大括号
                    sol.new_sol += new_content_list
                    sol.new_sol.append('}')
                    continue
        lines = get_line_number(node['loc'])
        update_new_contracts_with_lines(lines, sol)


def write_new_file(sol: SolParser):
    old_filename = os.path.basename(sol.file)[:-4]
    new_filename = old_filename + '.sol'
    path = os.path.dirname(sol.file)
    new_path = os.path.join(path, new_filename)
    with open(new_path, 'w') as fw:
        fw.write("\n".join(sol.new_sol))
    return new_path


if __name__ == '__main__':
    loop1_file = '/home/jrj/postgraduate/Symbolic/Backdoor/dataset/verified/DestroyToken/38.sol'
    sol = SolParser()
    sol.set_path(loop1_file)
    deal_with_statements(sol)
    write_new_file(sol)

