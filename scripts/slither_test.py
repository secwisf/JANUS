from slither.printers.call.call_graph import PrinterCallGraph
from slither.slither import Slither

# from scripts.process_dataset import get_solc_version

# sol_path = '/home/jrj/postgraduate/Symbolic/Backdoor/dataset/modified/Withdraw.sol'
# solc_version = get_solc_version(sol_path)
# solc_path = f"/home/jrj/.solc-select/artifacts/solc-{solc_version}"
# slither = Slither(sol_path, solc=solc_path)
# slither.register_printer(PrinterCallGraph)
# result = slither.run_printers()
# print()


from fuzzywuzzy import fuzz
print(fuzz.partial_ratio("launchTotal".lower(), "balances"))