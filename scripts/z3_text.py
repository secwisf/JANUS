from z3 import *

balances_num = Array('balances_bool', BitVecSort(160), IntSort())
addr1 = BitVecVal(0, 160)
addr2 = BitVecVal(1, 160)
addr3 = BitVecVal(2, 160)
addr4 = BitVecVal(3, 160)
balances_num = Store(balances_num, addr1, 10)
balances_num = Store(balances_num, addr2, 20)
balances_num = Store(balances_num, addr3, 30)
balances_num = Store(balances_num, addr4, 40)
s=Solver()
s.add(Select(balances_num, addr1) == 10)
print(s.check())
