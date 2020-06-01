"""
the tool to evaluate saliency map
"""

from .FmeasueEval import FmeasueEval
from .MAEEval import MAEEval
from .SMeasureEval import SMeasureEval
from collections import OrderedDict
from tqdm import tqdm
from .EvalDataloader import EvalDataloader
from typing import Dict
import torch

__all__ = ["get_measure"]


def get_measure( measure_list: list, img_root: str, label_root: str, device=torch.device('cpu') ) -> Dict[str, float or int]:
    """
    a faster function to get the evaluation of specific item

    :param measure_list:
    :param img_root:
    :param label_root:
    :param device:
    :return:
    """
    supported_measure_dict: Dict[str, type] = dict()
    for item in (FmeasueEval, MAEEval, SMeasureEval):
        for i in item.indexes:
            supported_measure_dict[i] = item
    ret_dict = {}
    dataloader = EvalDataloader(img_root, label_root)
    pbar = tqdm(total=len(measure_list))
    pbar.set_description("Start calculating")
    for item in measure_list:
        pbar.set_description(f"calculate {item}")
        assert item in supported_measure_dict, \
            f"{item} not be supported, supported list:{list(supported_measure_dict.keys())}"
        if item not in ret_dict:
            res = supported_measure_dict[item](dataloader, device).eval()
            ret_dict.update(res)
        pbar.update(1)

    return {item: ret_dict[item] for item in measure_list}
