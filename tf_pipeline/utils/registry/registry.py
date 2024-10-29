from typing import Type, Union, Optional, Dict, Callable
from importlib import import_module


class Registry:
    def __init__(self, name: str, location: str):
        self._name = name
        self._location = location
        self._module_dict: Dict[str, Type] = dict()
        self._imported = False

    def _register_module(self, module_name: str, module: Type):
        if module_name in self._module_dict.keys():
            raise NameError(f'module_name: {module_name} already registered')
        self._module_dict[module_name] = module
        return module

    def register_module(self, module_name: Optional[str] = None, module: Optional[Type] = None) -> Union[type, Callable]:

        if not (module_name is None or isinstance(module_name, str)):
            raise TypeError(
                f'module_name must be None or str, but got {type(module_name)}')

        if module is not None:
            if module_name is not None:
                name = module_name
            else:
                name = module.__name__
            return self._register_module(name, module)

        def register_fun(module):
            if module_name is not None:
                name = module_name
            else:
                name = module.__name__
            return self._register_module(name, module)

        return register_fun

    def _import_modules(self):
        import_module(self._location)
        return

    def get(self, key: str):
        if not self._imported:
            self._import_modules()
            self._imported = True

        if key not in self.module_dict.keys():
            raise KeyError(
                f'Module: {key} is not registered. Please register before using')

        module = self.module_dict[key]
        return module

    @property
    def module_dict(self):
        return self._module_dict
