from typing import List
from slither.core.declarations import Contract, FunctionContract, Function, SolidityFunction
from slither.slither import Slither


class SolidityInfo:
    def __init__(self, file: str, solc: str):
        try:
            self.slither = Slither(file, solc=solc)
        except Exception as e:
            raise Exception("Slither can not analyse the given file.")

    def get_contracts(self) -> List:
        return [contract for contract in self.slither.contracts if
                contract.kind in ['contract', 'library', 'interface']]

    def get_contracts_derived(self) -> List:
        return [contract for contract in self.slither.contracts_derived if contract.kind == 'contract']

    @staticmethod
    def get_contracts_all_funcs(contract: Contract) -> List:
        return contract.all_functions_called

    @staticmethod
    def get_contracts_public_funcs(contract: Contract) -> List[FunctionContract]:
        return [func for func in contract.functions_entry_points if not func.is_constructor]

    @staticmethod
    def get_contracts_constructor(contract: Contract):
        return contract.constructors
