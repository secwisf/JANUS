import gc
import os
import logging
import traceback


from fuzzywuzzy import fuzz
from solidity_parser import parser
from collections import deque
from tqdm import tqdm

deal_props = ['block', 'msg', 'tx']
deal_funcs = ['transfer', 'call', 'send']
not_op_list = ['<=', '>=', '==', '!=', ">", "<"]
balance_list = ['balance*', 'ownedTokensCount', 'tokenAmountOf', 'ownershipTokenCount', '_ownedTokensCount', 'accounts',
                'ownerToNFTokenCount', 'tokens']
balance_class_list = ['balance', 'tokencount', 'accounts']
uint256_except = ["totalsupply","currentsupply"]

def all_path(dirname, filter):
    result = []  # 所有的文件
    # filter = [".wav"]
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            ext = os.path.splitext(apath)[1]
            if ext in filter:
                result.append(apath)
    return result


def parse_sol(file):
    soure_unit = parser.parse_file(file, loc=True)
    children = soure_unit['children']
    # contracts_inheritance_dict = {}
    # all_contracts_list = []
    # base_contracts_list = []
    info_dict = {}
    for node in children:
        if node is None: continue
        if 'type' not in node: continue
        if node['type'] == 'ContractDefinition':
            if 'kind' in node:
                if node['kind'] == 'contract':
                    # if node['kind'] == 'contract' or node['kind'] == 'interface':
                    result = deal_with_contracts(node, file)
                    info_dict.setdefault(node['name'], result)
                    # contracts_inheritance_dict.setdefault(node['name'], result['baseContracts'])
                    # all_contracts_list.append(node['name'])
                    # base_contracts_list += result['baseContracts']
    return info_dict


def deal_with_contracts(node, file):
    result = {}
    result.setdefault('contractName', node['name'])
    # 处理baseContracts
    # baseContracts = []
    # if 'baseContracts' in node:
    #     for bC in node['baseContracts']:
    #         baseContracts.append(bC['baseName']['namePath'])
    # result.setdefault('baseContracts', baseContracts)

    functions_list = []
    vars_list = []
    # origin_functions_list = []
    # origin_vars_list = []
    # map256_list = []
    # 处理variables和functions
    if 'subNodes' in node:
        for sub in node['subNodes']:
            try:
                if sub['type'] == 'StateVariableDeclaration':
                    vars = deal_with_vars(sub)
                    vars_list += vars
                    # origin_vars_list.append(sub)

                if sub['type'] == 'FunctionDefinition':
                    funcs = deal_with_funcs(sub)
                    functions_list.append(funcs)
                    # origin_functions_list.append(sub)
            except Exception:
                msg = "%s-%s\n" % (file, traceback.format_exc())
                logging.error(msg)
    # for v in vars_list:
    #     type, name = v.split(" ")
    #     if type == 'mapping(address=>uint256)' or type == 'mapping(address=>uint)':
    #         map256_list.append(name)
    result.setdefault('functions', functions_list)
    result.setdefault('variables', vars_list)
    # result.setdefault('map256', map256_list)
    # result.setdefault("origin_functions", origin_functions_list)
    # result.setdefault("origin_variables", origin_vars_list)
    return result


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


def deal_with_funcs(node):
    func_dict = {}
    func_dict.setdefault("funcName", node['name'])
    func_dict.setdefault("stateMutability", node['stateMutability'])
    # return_list = []
    # returnParams = node['returnParameters']
    # if returnParams is not None:
    #     if 'parameters' in returnParams:
    #         for rn in returnParams['parameters']:
    #             if rn['type'] == 'Parameter':
    #                 para_type = parse_var_type(rn)
    #                 para_name = rn['name'] if rn['name'] is not None else ""
    #                 return_list.append("%s %s" % (para_type, para_name))
    para_list = []
    Params = node['parameters']
    if Params is not None:
        if 'parameters' in Params:
            for pn in Params['parameters']:
                para_type = parse_var_type(pn)
                para_name = pn['name'] if pn['name'] is not None else ""
                para_list.append(("%s %s" % (para_type, para_name), pn))
        func_dict.setdefault("parameters", para_list)
        # func_dict.setdefault("returnParameters", return_list)
        func_dict.setdefault('origin', node)
    return func_dict


def find_all_related_contracts(contract_list, info_dict):
    visited = []
    curr_deque = deque(contract_list)
    while curr_deque:
        curr_contract = curr_deque.popleft()
        # 如果已经访问过，跳过
        if curr_contract in visited or curr_contract not in info_dict: continue
        # 如果当前合约不再含有base contract，只将当前合约加入visited即可
        if len(info_dict[curr_contract]['baseContracts']) == 0:
            visited.append(curr_contract)
        # 如果当前合约含有base contract，将当前合约加入visited，并将其base contract加入队列
        else:
            visited.append(curr_contract)
            curr_deque.extend(info_dict[curr_contract]['baseContracts'])
    return list(set(visited))


def deal_with_inheritance(info_dict, all_contracts_list, base_contracts_list):
    top_contracts_list = list(set(all_contracts_list).difference(set(base_contracts_list)))
    result_dict = {}
    for tc in top_contracts_list:
        contract_dict = {}
        contract_dict.setdefault("contractName", tc)
        related_list = find_all_related_contracts(info_dict[tc]['baseContracts'], info_dict)
        funcs_list = []
        vars_list = []
        map256_list = []
        org_funcs_list = []
        org_vars_list = []
        for rc in related_list:
            funcs_list += info_dict[rc]['functions']
            vars_list += info_dict[rc]['variables']
            org_funcs_list += info_dict[rc]['origin_functions']
            org_vars_list += info_dict[rc]['origin_variables']
            map256_list += info_dict[rc]['map256']
        map256_list += info_dict[tc]['map256']
        funcs_list += info_dict[tc]['functions']
        vars_list += info_dict[tc]['variables']
        org_funcs_list += info_dict[tc]['origin_functions']
        org_vars_list += info_dict[tc]['origin_variables']
        contract_dict.setdefault("baseContracts", related_list)
        contract_dict.setdefault("functions", funcs_list)
        contract_dict.setdefault("variables", vars_list)
        contract_dict.setdefault("origin_functions", org_funcs_list)
        contract_dict.setdefault("origin_variables", org_vars_list)
        if len(map256_list) != 0:
            map256_list = sorted(map256_list, key=lambda i: len(i), reverse=False)
        contract_dict.setdefault("map256", map256_list)
        result_dict.setdefault(tc, contract_dict)
    return result_dict


def trans_functions(functions):
    func_content = ""
    num = 0
    for func in functions:
        num += 1
        tmp = "(%s)name:%s\nparameters: %s\nreturn: %s\n" % (
            str(num), func['funcName'], ", ".join(func['parameters']), ", "
                                                                       "".join(func['returnParameters']))
        func_content += tmp
    return func_content[:-1]


def trans_vars(vars):
    var_content = ""
    num = 0
    for var in vars:
        num += 1
        tmp = "(%s)%s\t" % (str(num), var)
        var_content += tmp
    return var_content[:-1]


def transform(result_dict):
    content = ""
    c_num = 1
    for key in result_dict:
        func_content = trans_functions(result_dict[key]['functions'])
        var_content = trans_vars(result_dict[key]['variables'])
        tmp_content = "%s.合约名称：%s\n函数:\n%s\n变量：\n%s\n代币：%s\n以太币：%s\n\n" % (str(c_num), key, func_content, var_content,
                                                                           result_dict[key]['isToken'],
                                                                           result_dict[key][
                                                                               'isEther'])
        content += tmp_content
        c_num += 1
    context = "共包含%s个合约\n%s" % (str(len(result_dict)), content)
    return context


def checkFunName(name):
    if name is not None:
        f = fuzz.partial_ratio("transfer", name)
        if f >= 80: return True
        f = fuzz.partial_ratio("send", name)
        if f >= 80: return True
        f = fuzz.partial_ratio("balance", name)
        if f >= 80: return True

    return False


def parse_call_expression(expressions):
    id_call_dict = {}
    for ex_node in expressions:
        accesses = set()
        if ex_node is None: continue
        while 'expression' in ex_node or 'base' in ex_node:
            if ex_node['type'] == 'MemberAccess':
                accesses.add(ex_node['memberName'])
                ex_node = ex_node['expression']
            elif ex_node['type'] == 'IndexAccess':
                ex_node = ex_node['base']
            elif ex_node['type'] == 'Identifier' and (len(accesses) != 0):
                if ex_node['name'] not in id_call_dict:
                    id_call_dict.setdefault(ex_node['name'], set())
            else:
                ex_node = ex_node['expression']
        if ex_node['type'] == 'MemberAccess':
            accesses.add(ex_node['memberName'])
        elif ex_node['type'] == 'Identifier' and len(accesses) != 0:
            if ex_node['name'] not in id_call_dict:
                id_call_dict.setdefault(ex_node['name'], set())
            id_call_dict[ex_node['name']] = id_call_dict[ex_node['name']].union(accesses)
    return id_call_dict


def parse_call_statement(node, expressions: list, declarations):
    if node is None: return
    if 'type' not in node: return
    if node['type'] == 'IfStatement':
        parse_call_statement(node['condition'], expressions, declarations)
        parse_call_statement(node['TrueBody'], expressions, declarations)
        parse_call_statement(node['FalseBody'], expressions, declarations)
    elif node['type'] == 'BinaryOperation':
        parse_call_statement(node['left'], expressions, declarations)
        parse_call_statement(node['right'], expressions, declarations)
    elif node['type'] == 'UnaryOperation':
        parse_call_statement(node['subExpression'], expressions, declarations)
    elif node['type'] == 'Block':
        if node['statements'] is not None:
            for s in node['statements']:
                parse_call_statement(s, expressions, declarations)
    elif node['type'] == 'WhileStatement':
        parse_call_statement(node['condition'], expressions, declarations)
        parse_call_statement(node['body'], expressions, declarations)
    elif node['type'] == 'ForStatement':
        parse_call_statement(node['conditionExpression'], expressions, declarations)
        parse_call_statement(node['loopExpression'], expressions, declarations)
        parse_call_statement(node['initExpression'], expressions, declarations)
        parse_call_statement(node['body'], expressions, declarations)
    elif node['type'] == 'VariableDeclarationStatement':
        if node['variables'] is not None:
            for v in node['variables']:
                if v['type'] == 'VariableDeclaration':
                    v_type = parse_var_type(v)
                    declarations.append("%s %s" % (v_type, v['name']))
    elif node['type'] == 'ExpressionStatement':
        parse_call_statement(node['expression'], expressions, declarations)
    elif node['type'] == 'TupleExpression':
        for c in node['components']:
            parse_call_statement(c, expressions, declarations)
    else:
        expressions.append(node)


def parse_op_expression(expressions: list):
    op_vars = set()
    for ex_node in expressions:
        if ex_node is None: continue
        while 'expression' in ex_node or 'base' in ex_node:
            if ex_node['type'] == 'MemberAccess':
                ex_node = ex_node['expression']
            elif ex_node['type'] == 'IndexAccess':
                ex_node = ex_node['base']
            elif ex_node['type'] == 'Identifier':
                op_vars.add(ex_node['name'])
            else:
                ex_node = ex_node['expression']
        if ex_node['type'] == 'Identifier':
            op_vars.add(ex_node['name'])
    return op_vars


# expression包括进行了一元和二元操作的变量，以供后续分析
def parse_op_statement(node, expressions: list):
    if node is None: return
    if 'type' not in node: return
    if node['type'] == 'IfStatement':
        parse_op_statement(node['condition'], expressions)
        parse_op_statement(node['TrueBody'], expressions)
        parse_op_statement(node['FalseBody'], expressions)
    elif node['type'] == 'Block':
        if node['statements'] is not None:
            for s in node['statements']:
                parse_op_statement(s, expressions)
    elif node['type'] == 'WhileStatement':
        parse_op_statement(node['condition'], expressions)
        parse_op_statement(node['body'], expressions)
    elif node['type'] == 'ForStatement':
        parse_op_statement(node['conditionExpression'], expressions)
        parse_op_statement(node['loopExpression'], expressions)
        parse_op_statement(node['initExpression'], expressions)
        parse_op_statement(node['body'], expressions)
    elif node['type'] == 'ExpressionStatement':
        parse_op_statement(node['expression'], expressions)
    elif node['type'] == 'TupleExpression':
        for c in node['components']:
            parse_op_statement(c, expressions)
    elif node['type'] == 'BinaryOperation':
        if 'operator' not in node: return
        if node['operator'] in not_op_list: return
        expressions.append(node['right'])
    elif node['type'] == 'UnaryOperation':
        expressions.append(node['subExpression'])


def judge_id_ether(id, declarations, state_vars):
    if id in deal_props: return True
    for declaration in declarations:
        type, name = declaration.split(" ")
        if name != id: continue
        if fuzz.partial_ratio("address", type) >= 90:
            return True
    for sv in state_vars:
        type, name = sv.split(" ")
        if name != id: continue
        if fuzz.partial_ratio("address", type) >= 90:
            return True
    return False


def checkFunBodyForOperation(body):
    if 'statements' not in body:
        return
    statements = body['statements']
    if statements is None or len(statements) == 0:
        return
    expressions = []
    for node in statements:
        if node is None: continue
        if 'type' not in node: continue
        parse_op_statement(node, expressions)
    return parse_op_expression(expressions)


def checkFunBodyForCall(body, state_vars):
    rej = False
    rtj = False
    if 'statements' not in body:
        return rtj, rej
    statements = body['statements']
    if statements is None or len(statements) == 0:
        return rtj, rej
    expressions = []
    declarations = []
    for node in statements:
        if node is None: continue
        if 'type' not in node: continue
        parse_call_statement(node, expressions, declarations)
    id_call_dict = parse_call_expression(expressions)
    for id in id_call_dict:
        idj = judge_id_ether(id, declarations, state_vars)
        accj = False
        for acc in id_call_dict[id]:
            if acc in deal_funcs:
                accj = True
                break
            # for df in deal_funcs:
            #     if fuzz.partial_ratio(df, acc) >= 80:
            #         accj = True
            #         break
        if rtj and rej:
            break
        if idj and accj:
            rej = True
        if (not idj) and accj:
            rtj = True
    return rtj, rej


def checkFuncs(origin_func_list, state_vars):
    rt = False
    re = False
    for i in range(len(origin_func_list)):
        tmp_t, tmp_e = checkFunBodyForCall(origin_func_list[i]['body'], state_vars)
        re = re or tmp_e
        rt = rt or tmp_t
        # rt = rt or checkFunName(origin_func_list[i]['name'])
        # op_exp = checkFunBodyForOperation(origin_func_list[i]['body'])
    return rt, re


def checkVars(var_list):
    rt = False
    for var in var_list:
        f = fuzz.partial_ratio('mapping(address=>uint256) balanceOf', var)
        if f >= 80: rt = True
    return rt


def deal_with_token(result_dict):
    for contract in result_dict:
        origin_func_list = result_dict[contract]['origin_functions']
        var_list = result_dict[contract]['variables']
        rtv = checkVars(var_list)
        rtf, ref = checkFuncs(origin_func_list, var_list)
        result_dict[contract].setdefault('isToken', str(rtv and rtf))
        result_dict[contract].setdefault('isEther', str(ref))
    return result_dict


def checkLoop(statement, result:dict):
    if statement is None:
        return
    elif statement['type'] == 'Block':
        for stm in statement['statements']:
            checkLoop(stm, result)
    elif statement['type'] == 'WhileStatement' or statement['type'] == 'ForStatement':
        result['record'].append("{0}-{1}".format(str(statement['loc']['start']['line']), str(statement['loc']['end']['line'])))
        return
    elif statement['type'] == 'IfStatement':
        checkLoop(statement['TrueBody'], result)
        checkLoop(statement['FalseBody'], result)
    else:
        return



