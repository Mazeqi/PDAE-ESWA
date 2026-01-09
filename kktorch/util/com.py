import os, copy, shutil, glob, re
from typing import List, Union


__all__ =[
    "CheckFunction",
    "check_type",
    "check_type_list",
    "correct_dirpath",
    "convert_1d_array",
    "makedirs",
    "get_file_list",
    "replace_sp_str_and_eval",
]


class CheckFunction:
    def __init__(self):
        self.is_check = True
    def __call__(self, *args, **kwargs):
        if self.is_check:
            self.check(*args, **kwargs)
            self.is_check = False
        return self.main(*args, **kwargs)
    def main(self, *args, **kwargs):
        raise NotImplementedError
    def check(self, *args, **kwargs):
        raise NotImplementedError


def check_type(instance: object, _type: Union[object, List[object]]):
    _type = [_type] if not (isinstance(_type, list) or isinstance(_type, tuple)) else _type
    is_check = [isinstance(instance, __type) for __type in _type]
    if sum(is_check) > 0:
        return True
    else:
        return False

def check_type_list(instances: List[object], _type: Union[object, List[object]], *args: Union[object, List[object]]):
    """
    Usage::
        >>> check_type_list([1,2,3,4], int)
        True
        >>> check_type_list([1,2,3,[4,5]], int, int)
        True
        >>> check_type_list([1,2,3,[4,5,6.0]], int, int)
        False
        >>> check_type_list([1,2,3,[4,5,6.0]], int, [int,float])
        True
    """
    if isinstance(instances, list) or isinstance(instances, tuple):
        for instance in instances:
            if len(args) > 0 and isinstance(instance, list):
                is_check = check_type_list(instance, *args)
            else:
                is_check = check_type(instance, _type)
            if is_check == False: return False
        return True
    else:
        return check_type(instances, _type)

def correct_dirpath(dirpath: str) -> str:
    if os.name == "nt":
        return dirpath if dirpath[-1] == "\\" else (dirpath + "\\")
    else:
        return dirpath if dirpath[-1] == "/" else (dirpath + "/")

def convert_1d_array(arrays: List[object]):
    """
    Usage::
        >>> convert_1d_array([1,2,3, [[1,1,23],2,3]])
        [1, 2, 3, 1, 1, 23, 2, 3]
    """
    import more_itertools as itr
    arrays = copy.deepcopy(arrays)
    for i, x in enumerate(arrays):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            arrays[i] = [x]
    arrays = list(itr.flatten(arrays))
    i = 0
    if len(arrays) > 0:
        while(1):
            if isinstance(arrays[i], list) or isinstance(arrays[i], tuple):
                arrays = convert_1d_array(arrays)
                i = 0
            else:
                i += 1
            if len(arrays) == i:
                break
    return arrays

def makedirs(dirpath: str, exist_ok: bool = False, remake: bool = False):
    dirpath = correct_dirpath(dirpath)
    if remake and os.path.isdir(dirpath): shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok = exist_ok)

def get_file_list(dirpath: str, regex_list: List[str]=[]) -> List[str]:
    dirpath = correct_dirpath(dirpath)
    file_list_org = glob.glob(dirpath + "**", recursive=True)
    file_list_org = list(filter(lambda x: os.path.isfile(x), file_list_org))
    file_list     = []
    for regstr in regex_list:
        file_list += list(filter(lambda x: len(re.findall(regstr, x)) > 0, file_list_org))
    return file_list if len(regex_list) > 0 else file_list_org

def replace_sp_str_and_eval(obj, dict_sp_str: dict):
    def work(_obj, _dict: dict):
        if isinstance(_obj, str):
            try:
                out = eval(_obj, copy.deepcopy(_dict)) # need copy.deepcopy
                if check_type(out, [int, float, str, list, dict, tuple]):
                    return out
                else:
                    return _obj
            except (NameError, SyntaxError, TypeError, AttributeError):
                for x, y in _dict.items():
                    if _obj.find(x) >= 0: _obj = _obj.replace(x, str(y)) # ex) "(___AAA)" -> "(10)"
                return _obj
        else:
            return _obj
    if   isinstance(obj, list) or isinstance(obj, tuple):
        return [replace_sp_str_and_eval(x, dict_sp_str) for x in obj]
    elif isinstance(obj, dict):
        return {x: replace_sp_str_and_eval(y, dict_sp_str) for x, y in obj.items()}
    else:
        return work(obj, dict_sp_str)
