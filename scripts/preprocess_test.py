import argparse
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from backdoor.utils.preprocess import PreProcess, SolFile, del_comments_and_blank
from scripts.process_dataset import get_solc_version

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file', '-f', type=str, required=True)
    # args = parser.parse_args()
    path = f"/home/jrj/postgraduate/Symbolic/Backdoor/contracts/XENFT.sol"
    version = get_solc_version(path)
    solc = f'/home/jrj/.solc-select/artifacts/solc-{version}'
    del_comments_and_blank(path)
    sol_file = SolFile()
    sol_file.set_path(path, solc)
    pre_processor = PreProcess(sol_file, 2)
    pre_processor(path="/home/jrj/postgraduate/Symbolic/Backdoor/contracts/", name_suffix="_inline")
# new_path = "/home/jrj/postgraduate/Symbolic/Backdoor/contracts/LinkToken_inline.sol"
# with open(new_path, 'w') as fw:
#     for line in sol_file.new_sol:
#         if isinstance(line, list):
#             for line_in in line:
#                 fw.write(f"{line_in}\n")
#         elif isinstance(line, str):
#             fw.write(f"{line}\n")
# print()
