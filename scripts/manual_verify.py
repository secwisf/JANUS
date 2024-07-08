import os
import sys
import traceback

from tqdm import tqdm
sys.path.append("/home/jrj/postgraduate/Symbolic/Backdoor")
from backdoor.utils.preprocess import SolFile, del_comments_and_blank, PreProcess

from scripts.process_dataset import all_path, get_solc_version


version = "0.6.x"
# origin_folders = os.listdir("/home/jrj/postgraduate/Symbolic/Backdoor/dataset/real_world/0.4.x-0.5.x")
# select_folders = os.listdir("/home/jrj/postgraduate/Symbolic/Backdoor/dataset/real_world_selected_v1/0.4.x-0.5.x")
origin_folders = os.listdir(f"/home/jrj/postgraduate/Symbolic/Backdoor/dataset/real_world/{version}")
select_folders = os.listdir(f"/dataset/real_world_selected_v1/{version}")
difference_folders = list(set(origin_folders).difference(select_folders))

base = f"/home/jrj/postgraduate/Symbolic/Backdoor/dataset/real_world/{version}"
for folder in tqdm(difference_folders):
    folder_path = os.path.join(base, folder)
    folder_list = folder.split("_")        
    file_name = "_".join(folder_list[1:]) + ".sol"
    sol_path = os.path.join(folder_path, file_name)
    sol_file = SolFile()
    pre_processor = PreProcess(2)

    try:
        del_comments_and_blank(sol_path)
        solc_version = get_solc_version(sol_path)
        solc_path = f'/home/jrj/.solc-select/artifacts/solc-{solc_version}/solc-{solc_version}'
        sol_file.set_path(sol_path, solc=solc_path)
        pre_processor.set_sol_file(sol_file)
        new_sol_path = pre_processor.pre_process()

        # dst_folder_path = os.path.join(dst_root_path, folder)
        # if os.path.exists(dst_folder_path):
        #     continue
        # os.mkdir(dst_folder_path)
        # shutil.copy(new_sol_path, os.path.join(dst_folder_path,
        #             f"{meta_dict['Contract Name']}_inline.sol"))
        # shutil.copy(meta_file, os.path.join(dst_folder_path, "meta.json"))
    except:
        msg = traceback.format_exc()
        with open(os.path.join(f'/home/jrj/postgraduate/Symbolic/Backdoor/dataset/manual_recheck/{version}',
                               f"{folder}.log"), 'w') as fw:
            fw.write(msg)
