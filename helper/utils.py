import os
from os.path import exists as file_exist
from os.path import join as pathjoin
from pytablewriter import MarkdownTableWriter
import pandas as pd
import pickle


def dump_pkl_file(obj,path):
    dir_path = os.path.dirname(path)
    mkdirs(dir_path)
    with open(path,'wb') as f:
        return pickle.dump(obj,f)

def load_pkl_file(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def mkdirs(dir_path: str):
    """
    make the dictionary according to the dir_path
    :param dir_path: the path of dir to be created
    :return:
    """
    if not file_exist(dir_path):
        os.makedirs(dir_path)


class EasyDict(dict):
    '''
    Convenience class that behaves exactly like dict(), but allows accessing
    the keys and values using the attribute syntax, i.e., "mydict.key = value".
    '''
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


def pandas2markdown(dataframe: pd.DataFrame) -> str:
    """
    translate the Dataframe in pandas into Markdown table
    :type dataframe: object
    """
    if not hasattr(pandas2markdown,'writer'):
        pandas2markdown.writer: MarkdownTableWriter = MarkdownTableWriter()
    pandas2markdown.writer.from_dataframe(
        dataframe,
        add_index_column=False,
    )
    return pandas2markdown.writer.dumps()



