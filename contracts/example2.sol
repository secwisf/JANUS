pragma solidity ^0.4.26;

contract owned {
    address public owner;

    function owned() public {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        if (!(msg.sender == owner)) {
            revert();
        }
        _;
    }

    function transferOwnership(address originNewOwner) public onlyOwner {
        if (!(msg.sender == owner)) {
            revert();
        }
        owner = originNewOwner;
    }
}

interface tokenRecipient {

    function receiveApproval(address _from, uint256 _value, address _token, bytes _extraData) external;
}

contract TokenERC20 {
    string public name;
    string public symbol;
    uint8 public decimals = 18;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
    event Burn(address indexed from, uint256 value);

    function TokenERC20(uint256 initialSupply, string tokenName, string tokenSymbol) public {
        totalSupply = initialSupply * 10 ** uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
        name = tokenName;
        symbol = tokenSymbol;
    }

    function _transfer(address _from, address _to, uint _value) internal {
        if (!(_to != address(0x0))) {
            revert();
        }
        if (!(balanceOf[_from] >= _value)) {
            revert();
        }
        if (!(balanceOf[_to] + _value > balanceOf[_to])) {
            revert();
        }
        uint previousBalances = balanceOf[_from] + balanceOf[_to];
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(_from, _to, _value);
        if (!(balanceOf[_from] + balanceOf[_to] == previousBalances)) {
            revert();
        }
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        {
            if (!(_to != address(0x0))) {
                revert();
            }
            if (!(balanceOf[msg.sender] >= _value)) {
                revert();
            }
            if (!(balanceOf[_to] + _value > balanceOf[_to])) {
                revert();
            }
            uint TokenERC20__transfer_3_previousBalances_0_0 = balanceOf[msg.sender] + balanceOf[_to];
            balanceOf[msg.sender] -= _value;
            balanceOf[_to] += _value;
            emit Transfer(msg.sender, _to, _value);
            if (!(balanceOf[msg.sender] + balanceOf[_to] == TokenERC20__transfer_3_previousBalances_0_0)) {
                revert();
            }
        }

        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        if (!(_value <= allowance[_from][msg.sender])) {
            revert();
        }
        allowance[_from][msg.sender] -= _value;
        {
            if (!(_to != address(0x0))) {
                revert();
            }
            if (!(balanceOf[_from] >= _value)) {
                revert();
            }
            if (!(balanceOf[_to] + _value > balanceOf[_to])) {
                revert();
            }
            uint TokenERC20__transfer_3_previousBalances_0_0 = balanceOf[_from] + balanceOf[_to];
            balanceOf[_from] -= _value;
            balanceOf[_to] += _value;
            emit Transfer(_from, _to, _value);
            if (!(balanceOf[_from] + balanceOf[_to] == TokenERC20__transfer_3_previousBalances_0_0)) {
                revert();
            }
        }

        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function approveAndCall(address _spender, uint256 _value, bytes _extraData) public returns (bool success) {
        tokenRecipient spender = tokenRecipient(_spender);
        bool TokenERC20_approve_TMP_34_0_0;
        {
            {
                allowance[msg.sender][_spender] = _value;
                emit Approval(msg.sender, _spender, _value);
                TokenERC20_approve_TMP_34_0_0 = true;
            }
        }
        if (TokenERC20_approve_TMP_34_0_0) {
            spender.receiveApproval(msg.sender, _value, this, _extraData);
            return true;
        }
    }

    function burn(uint256 _value) public returns (bool success) {
        if (!(balanceOf[msg.sender] >= _value)) {
            revert();
        }
        balanceOf[msg.sender] -= _value;
        totalSupply -= _value;
        emit Burn(msg.sender, _value);
        return true;
    }

    function burnFrom(address _from, uint256 _value) public returns (bool success) {
        if (!(balanceOf[_from] >= _value)) {
            revert();
        }
        if (!(_value <= allowance[_from][msg.sender])) {
            revert();
        }
        balanceOf[_from] -= _value;
        allowance[_from][msg.sender] -= _value;
        totalSupply -= _value;
        emit Burn(_from, _value);
        return true;
    }
}

contract MyAdvancedToken is owned, TokenERC20 {
    uint256 public sellPrice;
    uint256 public buyPrice;
    mapping(address => bool) public frozenAccount;
    event FrozenFunds(address target, bool frozen);

    function MyAdvancedToken(uint256 initialSupply, string tokenName, string tokenSymbol) public TokenERC20(initialSupply, tokenName, tokenSymbol) {}

    function _transfer(address _from, address _to, uint _value) internal {
        if (!(_to != address(0x0))) {
            revert();
        }
        if (!(balanceOf[_from] >= _value)) {
            revert();
        }
        if (!(balanceOf[_to] + _value >= balanceOf[_to])) {
            revert();
        }
        if (!(!frozenAccount[_from])) {
            revert();
        }
        if (!(!frozenAccount[_to])) {
            revert();
        }
        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(_from, _to, _value);
    }

    function mintToken(address target, uint256 mintedAmount) public onlyOwner {
        if (!(msg.sender == owner)) {
            revert();
        }
        balanceOf[target] += mintedAmount;
        totalSupply += mintedAmount;
        emit Transfer(0, this, mintedAmount);
        emit Transfer(this, target, mintedAmount);
    }

    function freezeAccount(address target, bool freeze) public onlyOwner {
        if (!(msg.sender == owner)) {
            revert();
        }
        frozenAccount[target] = freeze;
        emit FrozenFunds(target, freeze);
    }

    function setPrices(uint256 newSellPrice, uint256 newBuyPrice) public onlyOwner {
        if (!(msg.sender == owner)) {
            revert();
        }
        sellPrice = newSellPrice;
        buyPrice = newBuyPrice;
    }

    function buy() public payable {
        uint amount = msg.value / buyPrice;
        {
            if (!(msg.sender != address(0x0))) {
                revert();
            }
            if (!(balanceOf[this] >= amount)) {
                revert();
            }
            if (!(balanceOf[msg.sender] + amount >= balanceOf[msg.sender])) {
                revert();
            }
            if (!(!frozenAccount[this])) {
                revert();
            }
            if (!(!frozenAccount[msg.sender])) {
                revert();
            }
            balanceOf[this] -= amount;
            balanceOf[msg.sender] += amount;
            emit Transfer(this, msg.sender, amount);
        }
    }

    function sell(uint256 amount) public {
        address myAddress = this;
        if (!(myAddress.balance >= amount * sellPrice)) {
            revert();
        }
        {
            if (!(this != address(0x0))) {
                revert();
            }
            if (!(balanceOf[msg.sender] >= amount)) {
                revert();
            }
            if (!(balanceOf[this] + amount >= balanceOf[this])) {
                revert();
            }
            if (!(!frozenAccount[msg.sender])) {
                revert();
            }
            if (!(!frozenAccount[this])) {
                revert();
            }
            balanceOf[msg.sender] -= amount;
            balanceOf[this] += amount;
            emit Transfer(msg.sender, this, amount);
        }

        msg.sender.transfer(amount * sellPrice);
    }
}
