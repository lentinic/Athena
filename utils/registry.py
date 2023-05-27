from dataclasses import dataclass

class Model(type):
    registry = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.registry[new_cls.__name__] = new_cls
        return new_cls

class Dataset(type):
    registry = {}

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)
        cls.registry[new_cls.__name__] = new_cls
        return new_cls

def CommandLineArgument(TVAL:type, *args):
    @dataclass
    class internal:
        name: str
        help: str
        default: TVAL
        required: bool = False
        choices: list[TVAL] = None
        value_type: type = TVAL
    return internal(*args)

TrainingArguments:list = []
