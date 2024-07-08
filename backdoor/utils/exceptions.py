class BaseError(Exception):
    def __init__(self, contract, function):
        self.contract = contract
        self.function = function


class TooManyUsersError(BaseError):
    def __init__(self, contract, function):
        super().__init__(contract, function)

    def __str__(self):
        return f'Too many address variables in {self.contract}.{self.function}.'


class TooComplexMapping(BaseError):
    def __init__(self, contract, function):
        super().__init__(contract, function)

    def __str__(self):
        return f'The mapping variable is too complex in {self.contract}.{self.function}.'
