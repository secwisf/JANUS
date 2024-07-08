import os
import sys
from typing import List

from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import json
from fuzzywuzzy import fuzz
from process_dataset import all_path

files = all_path("/home/jrj/postgraduate/backdoor_data/result/real_world_v2", [".json"])
threshold = 85


def check_params():
    selected = []
    for file in tqdm(files):
        judge = False
        with open(file, 'r') as fr:
            result = json.load(fr)
        for contract in result:
            for summary_dict in result[contract]:
                if judge:
                    break
                summary = summary_dict['summary']
                for state_var in summary:
                    if judge:
                        break
                    if fuzz.partial_ratio(state_var, "balances") >= threshold or fuzz.partial_ratio(
                            state_var, "accounts") >= threshold:
                        value = summary[state_var]['value']
                        for field in value:
                            if not isinstance(value[field], dict): continue
                            if 'related_to_state_vars' in value[field] and len(
                                    value[field]['related_to_state_vars']) > 0:
                                for r_var in value[field]['related_to_state_vars']:
                                    if fuzz.partial_ratio(r_var.lower(),
                                                          "totalsupply") < threshold and fuzz.partial_ratio(
                                        r_var.lower(), "total") < threshold and fuzz.partial_ratio(
                                        r_var.lower(), "supply") < threshold:
                                        judge = True
                                        break
                                if judge:
                                    break
        if judge:
            selected.append(file)
    return selected


def check_arbitrary_transfer(summary: dict) -> bool:
    if len(summary) == 1:
        key = list(summary.keys())[0]
        if fuzz.partial_ratio(key, "balances") >= threshold and summary[key]['sign'] == 1:
            return True
    return False


def check_destroy_token(funcs: List[str]) -> bool:
    for func in funcs:
        f_name = func.split(".")[-1]
        if fuzz.partial_ratio(f_name, "burn") >= threshold or fuzz.partial_ratio(f_name, "kill") >= threshold:
            return True
    return False


def check_disable_transfer(summary: dict) -> bool:
    if len(summary) >= 2:
        for state_var in summary:
            value = summary[state_var]['value']
            if 'can_be_true' in value:
                return True
    return False


def check_freeze_account(summary: dict) -> bool:
    if len(summary) >= 2:
        for state_var in summary:
            if fuzz.partial_ratio(state_var, "balances") >= threshold: continue
            value = summary[state_var]['value']
            for field in value:
                if not isinstance(value[field], dict): continue
                if 'can_be_true' in value[field]:
                    return True
    return False


def check_generate_token(funcs: List[str]) -> bool:
    for func in funcs:
        f_name = func.split(".")[-1]
        if fuzz.partial_ratio(f_name, "mint") >= threshold:
            return True
    return False


def check_whitelist(summary: dict) -> bool:
    balance_flag = False
    bool_flag = False
    if len(summary) >= 2:
        for state_var in summary:
            if fuzz.partial_ratio(state_var, "balances") >= threshold:
                value = summary[state_var]['value']
                if len(value) == 1:
                    balance_flag = True
                continue
            value = summary[state_var]['value']
            for field in value:
                if not isinstance(value[field], dict): continue
                if 'can_be_true' in value[field] and summary[state_var]['sign'] == 1:
                    bool_flag = True
    return (balance_flag and bool_flag)


def classify():
    real_world_v2_statistics = {}
    real_world_v2_statistics["whitelist"] = []
    for file in tqdm(files):
        with open(file, 'r') as fr:
            result = json.load(fr)
        arbitrary_transfer_flag = False
        destroy_token_flag = False
        disable_transfer_flag = False
        freeze_account_flag = False
        generate_token_flag = False
        whitelist_flag = False
        for contract in result:
            for summary_dict in result[contract]:
                summary = summary_dict['summary']
                funcs = summary_dict['funcs']
                # if (not arbitrary_transfer_flag) and check_arbitrary_transfer(summary):
                #     real_world_v2_statistics["arbitrary_transfer"] = real_world_v2_statistics.get("arbitrary_transfer",
                #                                                                                   0) + 1
                #     arbitrary_transfer_flag = True
                # if (not destroy_token_flag) and check_destroy_token(funcs):
                #     real_world_v2_statistics["destroy_token"] = real_world_v2_statistics.get("destroy_token", 0) + 1
                #     destroy_token_flag = True
                # if (not disable_transfer_flag) and check_disable_transfer(summary):
                #     real_world_v2_statistics["disable_transfer"] = real_world_v2_statistics.get("disable_transfer",
                #                                                                                 0) + 1
                #     disable_transfer_flag = True
                # if (not freeze_account_flag) and check_freeze_account(summary):
                #     real_world_v2_statistics["freeze_account"] = real_world_v2_statistics.get("freeze_account", 0) + 1
                #     freeze_account_flag = True
                # if (not generate_token_flag) and check_generate_token(funcs):
                #     real_world_v2_statistics["generate_token"] = real_world_v2_statistics.get("generate_token", 0) + 1
                #     generate_token_flag = True
                if (not whitelist_flag) and check_whitelist(summary):
                    # real_world_v2_statistics["whitelist"] = real_world_v2_statistics.get("whitelist", 0) + 1
                    real_world_v2_statistics["whitelist"] = real_world_v2_statistics.get("whitelist", list()) + [file]
                    whitelist_flag = True
    with open("white_list_selected.txt", "w") as fw:
        fw.write("\n".join(real_world_v2_statistics["whitelist"]))
    # print(real_world_v2_statistics)


def calculate():
    while True:
        n1, n2 = map(float, input("请输入两个数值（小 大）：").split())
        print(f"{(n2 - n1) / n1 * 100}")


if __name__ == '__main__':
    # selected_files = check_params()
    # with open("selected_files_v2.txt", "w") as fw:
    #     fw.write("\n".join(selected_files))
    # print()
    classify()
    # calculate()
