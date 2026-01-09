import os
import pandas as pd
from typing import List
from tqdm import tqdm

from kktorch.util.com import get_file_list

__all__ = [
    "text_files_to_dataframe",
]


def text_files_to_dataframe(dirpath: str, regex_list: List[str]=[], encoding: str="utf8"):
    files = get_file_list(dirpath, regex_list=regex_list)
    df    = []
    for x in tqdm(files):
        se = pd.Series(dtype=object)
        se["path"]     = x
        se["filename"] = os.path.basename(x)
        with open(x, mode="r", encoding=encoding) as f:
            se["text"] = f.read()
        df.append(se)
    df = pd.concat(df, axis=1).T
    return df