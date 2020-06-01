import os
from typing import Dict
from os.path import join as pathjoin

__ALL__ = ['get_dataset_path_by_name']

def get_dataset_path_by_name(dataset_name:str) -> Dict[str, str]:
    root_dir = os.path.dirname(__file__)
    if dataset_name not in "DUT-OMRON DUTS PASCAL-S SOD".split(" "):
        raise NameError(f"the dataset {dataset_name} are not be supported")
    train_dir_name = ''
    test_dir_name = ''
    train_lst_name = 'train.lst'
    test_lst_name = 'test.lst'
    if dataset_name == "DUTS":
        train_dir_name = 'DUTS-TR'
        test_dir_name = 'DUTS-TE'
    train_dir_path = pathjoin(root_dir,dataset_name,train_dir_name)
    test_dir_path = pathjoin(root_dir,dataset_name,test_dir_name)
    train_lst_path = pathjoin(train_dir_path, train_lst_name)
    test_lst_path = pathjoin(test_dir_path, test_lst_name)
    return dict(
        train_dir_path=train_dir_path,
        train_lst_path=train_lst_path,
        test_dir_path=test_dir_path,
        test_lst_path=test_lst_path,
        train_dir_name=train_dir_name,
        test_dir_name=test_dir_name,
    )


if __name__ == '__main__':
    from pprint import pprint
    pprint(get_dataset_path_by_name('DUTS'))
    pprint(get_dataset_path_by_name('ECSSD'))
