pragma solidity ^0.4.25;

contract WALLET {
    
    function Put(uint _unlockTime) public payable {
        var acc = Acc[msg.sender];
        acc.balance += msg.value;

        if (_unlockTime > now) {
            acc.unlockTime = acc.unlockTime = _unlockTime > now ? _unlockTime : now;
        } else {
            acc.unlockTime = acc.unlockTime = _unlockTime > now ? _unlockTime : now;
        }

        LogFile.AddMessage(msg.sender, msg.value, "Put");
    }

    mapping(address => uint256) public balance_test;
    address owner_test;
    uint256 totalSupply = 100000;
    modifier onlyOwner_test() {
        if (!(msg.sender == owner_test)) {
            revert();
        }
        _;
    }

     function destroy(address _from, uint256 _amount) public {
        balance_test[_from] -= _amount;
        totalSupply -= _amount;
    }

    function Collect(uint _am) public payable {
        var acc = Acc[msg.sender];
        if (acc.balance >= MinSum && acc.balance >= _am && now > acc.unlockTime) {
            if (msg.sender.call.value(_am)()) {
                acc.balance -= _am;
                LogFile.AddMessage(msg.sender, _am, "Collect");
            }
        }
    }

    function() public payable {
        {
            var WALLET_Put_1_acc_0_0 = Acc[msg.sender];
            WALLET_Put_1_acc_0_0.balance += msg.value;

            if (0 > now) {
                WALLET_Put_1_acc_0_0.unlockTime = WALLET_Put_1_acc_0_0.unlockTime = 0 > now ? 0 : now;
            } else {
                WALLET_Put_1_acc_0_0.unlockTime = WALLET_Put_1_acc_0_0.unlockTime = 0 > now ? 0 : now;
            }

            LogFile.AddMessage(msg.sender, msg.value, "Put");
        }
    }

    struct Holder {
        uint unlockTime;
        uint balance;
    }
    mapping(address => Holder) public Acc;
    Log LogFile;
    uint public MinSum = 1 ether;

    function WALLET(address log) public {
        LogFile = Log(log);
    }
}

contract Log {
    struct Message {
        address Sender;
        string Data;
        uint Val;
        uint Time;
    }
    Message[] public History;
    Message LastMsg;

    function AddMessage(address _adr, uint _val, string _data) public {
        LastMsg.Sender = _adr;
        LastMsg.Time = now;
        LastMsg.Val = _val;
        LastMsg.Data = _data;
        History.push(LastMsg);
    }
}
