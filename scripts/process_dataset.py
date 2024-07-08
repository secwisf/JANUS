import gc
import glob
import json
import os
import pickle
import re
import shutil
import sys
import traceback
from math import ceil
from multiprocessing import Pool, Manager

import solcx
from fuzzywuzzy import fuzz
from slither import Slither

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from backdoor.state.solidity_info import SolidityInfo
from backdoor.utils.node_utils import is_mapping_type, is_number_type
from backdoor.utils.preprocess import SolFile, PreProcess, del_comments_and_blank

from tqdm import tqdm


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
        elif ver1_list[i] < ver2_list[i]:
            return ver2
    return ver1


def get_solc_version(path) -> str:
    max_version = '0.4.11'
    with open(path, 'r') as fr:
        content_list = fr.readlines()
    for line in content_list:
        if 'pragma solidity' not in line:
            continue
        curr_ver = re.findall(r'(\d+\.\d+\.\d+)', line)[0]
        max_version = get_higher_version(max_version, curr_ver)
    return max_version


def add_to_verified_dataset():
    sol_dict = {
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/Arbix_inline.sol": "GenerateToken",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/DeFi100_inline.sol": "DisableTransfer",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/LEVTOKEN_inline.sol": "GenerateToken",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/METAMOONMARS_inline.sol": "DestroyToken",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/MINIBASKETBALL_inline.sol": "DestroyToken",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/RUNE_inline.sol": "GenerateToken",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/SadeIT_inline.sol": "FreezeAccount",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/SudoRare_inline.sol": "GenerateToken",
        "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/wZNN_inline.sol": "GenerateToken"
    }
    for file in sol_dict:
        fileName = file.split("/")[-1]
        solType = sol_dict[file]
        dst_path = os.path.join("/home/jrj/postgraduate/backdoor_data/dataset/verified/modified", solType, fileName)
        shutil.copy2(file, dst_path)


def process_verified_dataset(sol_type):
    # from tqdm import tqdm
    # from backdoor.utils.preprocess import del_comments_and_blank, SolFile, PreProcess

    # process_verified_dataset
    all_sols = all_path(
        f"/home/jrj/postgraduate/backdoor_data/dataset/verified/patched_origin/{sol_type}", [".sol"])
    dst_path = os.path.join("/home/jrj/postgraduate/backdoor_data/dataset/verified/patched_modified", f"{sol_type}")
    error_list = []

    for sol_path in tqdm(all_sols):
        try:
            version = get_solc_version(sol_path)
            solc_path = f"/home/jrj/.solc-select/artifacts/solc-{version}"
            sol_file = SolFile()
            del_comments_and_blank(sol_path)
            sol_file.set_path(sol_path, solc_path)
            pre_processor = PreProcess(sol_file, 2)
            pre_processor.pre_process(path=dst_path, name_suffix="_inline")

            # sol_parser.set_path(new_sol_path)
            # deal_with_statements(sol_parser)
            # new_sol_path = write_new_file(sol_parser)
            #
            # sol_file.set_path(new_sol_path, solc=solc_version)
        except:
            error_list.append(f"{sol_path}\n{traceback.format_exc()}\n")

    with open(f'process_error_list_{sol_type}.txt', 'w') as fw:
        fw.write('\n'.join(error_list))
    print()


def delete_inline():
    root_path = '/home/jrj/postgraduate/backdoor_data/dataset/verified/patched_bytecodes/'
    dir_name = os.listdir(root_path)
    for folder in tqdm(dir_name):
        folder_path = os.path.join(root_path, folder)
        all_sols = all_path(folder_path, [".sol", '.hex'])
        for sol_file in all_sols:
            old_filename = os.path.basename(sol_file)[:-4]
            if old_filename.endswith('_inline'):
                os.remove(sol_file)


def check_potential_token(file, solc_version):
    result = False
    solc_path = f"/home/jrj/.solc-select/artifacts/solc-{solc_version}"
    try:
        slither_obj = Slither(file, solc=solc_path)
    except:
        return False
    for contract in slither_obj.contracts_derived:
        has_balances = False
        for state_var in contract.state_variables:
            if is_mapping_type(state_var.type):
                if is_number_type(state_var.type.type_to):
                    if fuzz.partial_ratio(state_var.name, "balances") >= 85:
                        has_balances = True
                        break
        if not contract.is_possible_token and not has_balances:
            continue
        else:
            result = True
            break
    return result


def process_real_dataset(folder_list):
    src_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/real_world_v2"
    dst_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/real_world_selected_v2"
    for folder in folder_list:
        src_path = os.path.join(src_root_path, folder)
        sols = all_path(src_path, [".sol"])
        if len(sols) > 1 or len(sols) == 0: continue
        sol_path = sols[0]
        version = get_solc_version(sol_path)
        solc_path = f"/home/jrj/.solc-select/artifacts/solc-{version}"
        try:
            del_comments_and_blank(sol_path)
            sol_file = SolFile()
            sol_file.set_path(sol_path, solc_path)
            pre_processor = PreProcess(sol_file, 2)
            dst_folder_path = os.path.join(dst_root_path, folder)
            if not os.path.exists(dst_folder_path):
                os.mkdir(dst_folder_path)
            pre_processor(path=dst_folder_path, name_suffix="_inline")
        except:
            continue
    for folder in folder_list:
        path = os.path.join(dst_root_path, folder)
        sols = all_path(path, [".sol"])
        if len(sols) == 0:
            os.rmdir(path)


def process_real_dataset_task():
    pool = Pool(15)
    manager = Manager()
    src_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/real_world_v2"
    folders = manager.list(os.listdir(src_root_path))
    result = manager.list()
    lock = manager.Lock()
    batch_size = 1000
    jobs = ceil(len(folders) / batch_size)
    for index in range(jobs):
        pool.apply_async(func=select_not_common_pair,
                         args=(folders[index * batch_size:(index + 1) * batch_size], result, lock,))
    pool.close()
    pool.join()
    with open("not_common_pair_select.txt", "w") as fw:
        fw.write("\n".join(list(result)))


def process_meaningless_dataset():
    with open("/home/jrj/postgraduate/Symbolic/Backdoor/scripts/filter_identify_result.pkl", "rb") as fr:
        all_sols = list(pickle.load(fr).keys())
    dst_path = "/home/jrj/postgraduate/Symbolic/Backdoor/dataset/modified"
    for sol_path in tqdm(all_sols):
        try:
            version = get_solc_version(sol_path)
            solc_path = f"/home/jrj/.solc-select/artifacts/solc-{version}"
            sol_file = SolFile()
            del_comments_and_blank(sol_path)
            sol_file.set_path(sol_path, solc_path)
            pre_processor = PreProcess(sol_file, 2)
            pre_processor.pre_process(path=dst_path, name_suffix="")

            # sol_parser.set_path(new_sol_path)
            # deal_with_statements(sol_parser)
            # new_sol_path = write_new_file(sol_parser)
            #
            # sol_file.set_path(new_sol_path, solc=solc_version)
        except:
            gc.collect()
            continue


def mv_real_dataset(root_path):
    max_version = '0.9.0'
    # max_version = '0.5.18'
    min_version = '0.3.0'
    # min_version = '0.4.9'
    dst_root_path = '/home/jrj/postgraduate/backdoor_data/dataset/real_world_v2'
    folder_list = os.listdir(root_path)
    for folder in tqdm(folder_list):
        path = os.path.join(root_path, folder)
        if os.path.isdir(path):
            all_files = all_path(path, [".sol"])
            if len(all_files) > 1:
                continue
            try:
                file = all_files[0]
                solc_version = get_solc_version(file)
                if get_higher_version(solc_version, max_version) == solc_version:
                    continue
                if get_higher_version(solc_version, min_version) == min_version:
                    continue

                result = check_potential_token(file, solc_version)
                if not result:
                    continue
                dst_path = os.path.join(dst_root_path, folder)
                if os.path.exists(dst_path):
                    continue
                shutil.copytree(path, dst_path)
            except:
                continue
        elif os.path.isfile(path):
            try:
                if path.endswith('json'):
                    with open(path, 'r') as fr:
                        json_dict_list = json.load(fr)
                    for meta_dict in json_dict_list:
                        if meta_dict['Compiler'] != 'Solidity':
                            continue
                        if get_higher_version(meta_dict['Version'], max_version) == meta_dict['Version']:
                            continue
                        if get_higher_version(meta_dict['Version'], min_version) == min_version:
                            continue
                        tmp_folder = f"{meta_dict['Address']}_{meta_dict['Contract Name']}"
                        tmp_path = os.path.join(root_path, tmp_folder)
                        if os.path.isdir(tmp_path):
                            result = check_potential_token(tmp_path, meta_dict)
                            if not result:
                                continue

                            dst_path = os.path.join(dst_root_path, tmp_folder)
                            if os.path.exists(dst_path):
                                continue
                            shutil.copytree(tmp_path, dst_path)
            except:
                continue


def merge_real_dataset():
    v1_paths = ['/home/jrj/postgraduate/backdoor_data/dataset/real_world_v1/0.4.x-0.5.x',
                '/home/jrj/postgraduate/backdoor_data/dataset/real_world_v1/0.6.x']
    v2_path = '/home/jrj/postgraduate/backdoor_data/dataset/real_world_v2'
    v2_folders = set(os.listdir(v2_path))
    for v1_path in v1_paths:
        folder_list = os.listdir(v1_path)
        for folder in tqdm(folder_list):
            if folder in v2_folders: continue
            src_path = os.path.join(v1_path, folder)
            dst_path = os.path.join(v2_path, folder)
            if os.path.exists(dst_path): continue
            shutil.copytree(src_path, dst_path)


def reveal_json(root_path):
    folder_list = os.listdir(root_path)
    for folder in tqdm(folder_list):
        try:
            path = os.path.join(root_path, folder)
            if os.path.isdir(path):
                json_files = all_path(path, ".json")
                if len(json_files) == 1:
                    json_file = json_files[0]
                    json_name = json_file.split("/")[-1]
                    if json_name.startswith(folder):
                        with open(json_file, 'r') as fr:
                            json_dict = json.load(fr)
                        source_code_dict = json.loads(json_dict["SourceCode"][1:-1])
                        for source in source_code_dict['sources']:
                            source_code = source_code_dict['sources'][source]['content']
                            source_code_file = os.path.join(path, source.split("/")[-1])
                            with open(source_code_file, 'w') as fw:
                                fw.write(source_code)
        except:
            continue


def calculate_number():
    # max_version = '0.6.13'
    max_version = '0.5.18'
    # min_version = '0.5.17'
    min_version = '0.4.9'
    dst_root_path = '/home/jrj/postgraduate/Symbolic/Backdoor/dataset/real_world/0.6.x'
    root_path = "/home/jrj/postgraduate/Symbolic/etherscan-contract-crawler/bsc_contracts"
    folder_list = os.listdir(root_path)
    num = 0
    for folder in folder_list:
        path = os.path.join(root_path, folder)
        if os.path.isdir(path):
            try:
                meta_file = os.path.join(path, "meta.json")
                with open(meta_file, 'r') as fr:
                    meta_dict = json.load(fr)
            except:
                continue
            if meta_dict['Compiler'] != 'Solidity':
                continue
            if get_higher_version(meta_dict['Version'], max_version) == meta_dict['Version']:
                continue
            if get_higher_version(meta_dict['Version'], min_version) == min_version:
                continue

            # result = check_potential_token(path, meta_dict)
            # if not result:
            #     continue
            num += 1
        elif os.path.isfile(path):
            if path.endswith('json'):
                with open(path, 'r') as fr:
                    json_dict_list = json.load(fr)
                for meta_dict in json_dict_list:
                    if meta_dict['Compiler'] != 'Solidity':
                        continue
                    if get_higher_version(meta_dict['Version'], max_version) == meta_dict['Version']:
                        continue
                    if get_higher_version(meta_dict['Version'], min_version) == min_version:
                        continue
                    tmp_folder = f"{meta_dict['Address']}_{meta_dict['Contract Name']}"
                    tmp_path = os.path.join(root_path, tmp_folder)
                    if os.path.isdir(tmp_path):
                        # result = check_potential_token(tmp_path, meta_dict)
                        # if not result:
                        #     continue
                        num += 1
    print(num)


def delete_empty():
    src_root_path = "/home/jrj/postgraduate/backdoor_data/result/real_world_v2"
    folder_list = os.listdir(src_root_path)
    for folder in tqdm(folder_list):
        path = os.path.join(src_root_path, folder)
        jsons = all_path(path, [".json"])
        if len(jsons) == 0:
            os.rmdir(path)
        elif len(jsons) == 1:
            size = os.path.getsize(jsons[0])
            if size == 0:
                shutil.rmtree(path)


def generate_bytecode_task():
    folders = os.listdir("/home/jrj/postgraduate/backdoor_data/dataset/confusion")
    src_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/confusion"
    dst_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/confusion_bytecodes"
    for folder in tqdm(folders):
        src_folder = os.path.join(src_root_path, folder)
        dst_folder = os.path.join(dst_root_path, folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        sols = all_path(src_folder, [".sol"])
        for file in sols:
            file_name = file.split("/")[-1][:-4] + ".hex"
            solc_version = get_solc_version(file)
            solc_path = f"/home/jrj/.solc-select/artifacts/solc-{solc_version}"
            contract = SolidityInfo(file, solc=solc_path).get_contracts_derived()[0]
            result = solcx.compile_files(
                [file],
                output_values=["bin-runtime"],
                solc_version=solc_version
            )
            key = f"{file}:{contract.name}"
            bytecode = result[key]['bin-runtime']
            new_path = os.path.join(dst_folder, file_name)
            with open(new_path, "w") as fw:
                fw.write(bytecode)


def generate_vul_hunter_result():
    folders = os.listdir("/home/jrj/postgraduate/backdoor_data/dataset/verified/origin")
    src_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/verified/bytecodes"
    dst_root_path = "/home/jrj/postgraduate/backdoor_data/result/vulHunter"
    PYTHON = "/home/jrj/miniconda3/envs/vulhunter/bin/python3"
    content = " #！/bin/zsh\n"
    for folder in folders:
        sh_content = "#！/bin/zsh\n"
        src_folder = os.path.join(src_root_path, folder)
        dst_folder = os.path.join(dst_root_path, folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        hexs = all_path(src_folder, [".hex"])
        for file in tqdm(hexs):
            file_name = file.split("/")[-1][:-4] + ".txt"
            output_file = os.path.join(dst_folder, file_name)
            cmd = f'{PYTHON} /home/jrj/postgraduate/VulHunter/VulHunter/main/main.py --contract "{file}" --filetype bytecode --model-dir /home/jrj/postgraduate/VulHunter/VulHunter/models --instance-len 10 --ifmap nomap >> {output_file}\n'
            sh_content += cmd
        with open(f"/home/jrj/postgraduate/VulHunter/VulHunter/scripts/{folder}.sh", "w") as fw:
            fw.write(sh_content)
        content += f"nohup time /home/jrj/postgraduate/VulHunter/VulHunter/scripts/{folder}.sh > /home/jrj/postgraduate/VulHunter/VulHunter/scripts/{folder}.log 2>&1 &\n"
        with open("/home/jrj/postgraduate/VulHunter/VulHunter/scripts/run_all.sh", "w") as fw:
            fw.write(content)


def analyse_vul_hunter_result():
    folders = os.listdir("/home/jrj/postgraduate/backdoor_data/dataset/verified/origin")
    src_root_path = "/home/jrj/postgraduate/backdoor_data/result/vulHunter"
    result_dict = {}
    for folder in folders:
        result_dict[folder] = {}
        src_folder = os.path.join(src_root_path, folder)
        txts = all_path(src_folder, [".txt"])
        for file in tqdm(txts):
            with open(file, "r") as fr:
                content = fr.readlines()
            for line in content:
                if not line.startswith("Vulnerability:"):
                    continue
                v_type = line.split(",")[0].split(": ")[-1]
                severity = line.split(",")[1].split(": ")[-1]
                if severity not in ["High"]:
                    continue
                result_dict[folder][v_type] = result_dict[folder].get(v_type, 0) + 1
    with open("../result/verified/vul_result.pkl", "wb") as fw:
        pickle.dump(result_dict, fw)


def generate_tokeer_result():
    folders = os.listdir("/home/jrj/postgraduate/backdoor_data/dataset/confusion_bytecodes")
    PYTHON = "/home/jrj/miniconda3/envs/pied/bin/python3"
    TOKEER = "/home/jrj/postgraduate/Tokeer/tokeer.py"
    sh_content = " #！/bin/zsh\n"
    src_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/confusion_bytecodes"
    dst_root_path = "/home/jrj/postgraduate/backdoor_data/result/confusion/tokeer"

    for folder in folders:
        src_folder = os.path.join(src_root_path, folder)
        dst_folder = os.path.join(dst_root_path, folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)
        hexs = all_path(src_folder, [".hex"])
        for file in tqdm(hexs):
            file_name = file.split("/")[-1][:-4] + ".json"
            output_file = os.path.join(dst_folder, file_name)
            working_dir = os.path.join("/home/jrj/postgraduate/Tokeer/.temp/confusion", folder)
            cmd = f'{PYTHON} {TOKEER}  -C "/home/jrj/postgraduate/Tokeer/logic/Tokeer.dl" -r "{output_file}" -w "{working_dir}" "{file}"\n'
            sh_content += cmd
    with open("/home/jrj/postgraduate/Tokeer/tokeer_confusion.sh", "w") as fw:
        fw.write(sh_content)


def analyze_tokeer_result():
    folders = os.listdir("/home/jrj/postgraduate/backdoor_data/dataset/verified/origin")
    src_root_path = "/home/jrj/postgraduate/backdoor_data/result/tokeer/origin"
    result = ["ModifyBalance", "BlackList", "TimeLimit", "AlienDepend"]
    result = set([name.lower() for name in result])
    result_dict = {}
    for folder in folders:
        result_dict[folder] = {'total': 0, 'pos': 0, 'avg_time': 0.0}
        src_folder = os.path.join(src_root_path, folder)
        jsons = all_path(src_folder, [".json"])
        files_num = len(jsons)
        all_time = 0.0
        for file in jsons:
            with open(file, "r") as fr:
                json_file = json.load(fr)[0]
            if len(json_file) != 4:
                files_num -= 1
                continue
            res = set([fact.lower() for fact in json_file[1]])
            if len(res.intersection(result)) != 0:
                result_dict[folder]['pos'] += 1
            all_time += json_file[3]['disassemble_time'] + json_file[3]['decomp_time'] + json_file[3][
                'inline_time'] + json_file[3]['client_time']
        result_dict[folder]['total'] = files_num
        result_dict[folder]['avg_time'] = all_time / files_num
    print(result_dict)


def analyze_pied_piper_result():
    outputs = glob.glob("/home/jrj/postgraduate/PiedPiperBackdoor/tools/analyser/batch_confusion*.out")
    total_num = 0
    total_time = 0.0
    POS = 0
    backdoor_types = ["ArbitraryTransfer", "DestroyToken", "DisableTransfer", "FrozeAccount", "GenerateToken"]
    for file in tqdm(outputs):
        with open(file, "r") as fr:
            content_list = fr.readlines()
        for res in content_list:
            res_list = res.split(",")
            if len(res_list) < 3: continue
            total_time += float(res_list[1]) + float(res_list[2])
            total_num += 1
            for backdoor in backdoor_types:
                if backdoor in res:
                    print(backdoor)
                    POS += 1
                    break
    print(f"total:{str(total_num)}, avg_time:{str(total_time/total_num)}, positive:{str(POS)}")       
            
    

def calculate_proportion():
    root_path = "/home/jrj/postgraduate/backdoor_data/dataset/real_world_v2"
    folders = os.listdir(root_path)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(root_path, folder))]
    eth_num = 0
    bsc_num = 0
    for folder in tqdm(folders):
        bsc_path = os.path.join("/home/jrj/postgraduate/backdoor_data/bsc_contracts", folder)
        if os.path.exists(bsc_path) or os.path.exists(f"{bsc_path}.json"):
            bsc_num += 1
    print(33176 - bsc_num)
    print(bsc_num)


def select_not_common_pair(folder_list, result, lock):
    tmp_result = []
    src_root_path = "/home/jrj/postgraduate/backdoor_data/dataset/real_world_v2"
    for folder in tqdm(folder_list):
        src_path = os.path.join(src_root_path, folder)
        file = all_path(src_path, [".sol"])[0]
        flag = False
        try:
            solc_version = get_solc_version(file)
            solc_path = f"/home/jrj/.solc-select/artifacts/solc-{solc_version}"
            sol_slither = Slither(file, solc=solc_path)
            for contract in sol_slither.contracts_derived:
                if not flag:
                    for sv in contract.state_variables:
                        if is_mapping_type(sv.type) and ((fuzz.partial_ratio(sv.name.lower(), "balances") >= 85) or (fuzz.partial_ratio(sv.name.lower(), "towned") >= 85) or (fuzz.partial_ratio(sv.name.lower(), "rowned") >= 85)):
                            flag = True
                            break
        except:
            flag = False
            gc.collect()
        if not flag:
            tmp_result.append(file)
    lock.acquire()
    result += tmp_result
    lock.release()
    

def mv_confusion_dataset():
    with open("/home/jrj/postgraduate/vpg/utils/confusion_TP.txt", "r") as fw:
        TP_list = fw.readlines()
    with open("/home/jrj/postgraduate/vpg/utils/confusion_FN.txt", "r") as fw:
        FN_list = fw.readlines()
    all_list = TP_list + FN_list
    dst_root = "/home/jrj/postgraduate/backdoor_data/dataset/confusion"
    for file in tqdm(all_list):
        src_folder =  os.path.dirname(file)
        folder = file.split("/")[-2]
        dst_folder = os.path.join(dst_root, folder)
        shutil.copytree(src_folder, dst_folder)
              
    


if __name__ == '__main__':
    # all_files = all_path('/home/jrj/postgraduate/Symbolic/Backdoor/dataset/verified/modified', ['.sol'])
    # files_dict = {}
    # for file in tqdm(all_files):
    #     version = get_solc_version(file)
    #     files_dict[file] = version
    # with open('/home/jrj/postgraduate/Symbolic/Backdoor/dataset/verified/modified/files_dict.pkl', 'wb') as fw:
    #     pickle.dump(files_dict, fw)
    # sol_type = "FreezeAccount"
    # delete_inline()
    # process_verified_dataset(sol_type)
    # merge_real_dataset()
    # mv_real_dataset()
    # root_path = '/home/jrj/postgraduate/backdoor_data/bsc_contracts'
    # root_path = '/home/jrj/postgraduate/backdoor_data/contracts/'
    # print(root_path)
    # reveal_json(root_path)
    # mv_real_dataset(root_path)
    # calculate_number()
    # process_real_dataset_task()
    # delete_empty()
    # add_to_verified_dataset()
    # generate_bytecode_task()
    # generate_vul_hunter_result()
    # analyse_vul_hunter_result()
    # delete_empty()
    # process_meaningless_dataset()
    # generate_tokeer_result()
    # analyze_tokeer_result()
    # calculate_proportion()
    # process_real_dataset_task()
    # mv_confusion_dataset()
    analyze_pied_piper_result()
