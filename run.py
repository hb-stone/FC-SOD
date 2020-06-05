from config import Configuration
from helper.visual_helper import VisualHelper
from helper.visual_helper import ImgsType
from helper.model_saver import ModelSaver
from data_provider.sod_dataset import SaliencyDataSet
from helper.optim import Optimizer
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import os
from helper.trainer import Trainer
from helper.utils import pathjoin
from helper.utils import mkdirs
from helper.utils import pandas2markdown
import pandas as pd
from helper.tester import Tester
from helper.evaluator_tool import get_measure
from collections.abc import Sequence
from collections import OrderedDict
import time
import portalocker
from losses import BCELossWithLogits_Mask
from data_provider.sod_dataloader import SOD_Dataloader
from helper.trainer import TrainDataWrapper
from helper.utils import load_pkl_file
from helper.utils import dump_pkl_file
from torch.utils import data
from torch import optim
from networks.GAN_Unet import Generator,FCDiscriminator
from torch.nn import Module
import numpy as np
import random



def train(config:Configuration) -> None:

    if config.DISABLE_TRAIN:
        return
    if config.USE_GPU is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    batch_size = config.BATCH_SIZE
    crop_size = config.CROP_SIZE
    sal_dataset = SaliencyDataSet(config.DATASET_TRAIN_ROOT_DIR, config.DATASET_TRAIN_LIST_PATH, \
                                   ignore_value=255.0,\
                                   crop_size=(crop_size,crop_size),\
                                   is_random_flip=True,is_scale=False)
    dataset_size = len(sal_dataset)
    gt_dataset = SaliencyDataSet(config.DATASET_TRAIN_ROOT_DIR, config.DATASET_TRAIN_LIST_PATH, \
                                   ignore_value=sal_dataset.ignore_value,\
                                   crop_size=(crop_size,crop_size),\
                                   is_random_flip=True,is_scale=False)

    if batch_size > 1 and torch.cuda.device_count() > 0:
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True
    if config.PARTIAL_DATA:
        partial_size = int(config.PARTIAL_DATA * dataset_size)
    else:
        partial_size = dataset_size
    if partial_size < dataset_size:
        if config.DATA_IDX_PKL_PATH:
            idx = load_pkl_file(config.DATA_IDX_PKL_PATH)
            print(f'load index file {config.DATA_IDX_PKL_PATH} success!')
        else:
            idx = list(range(0,dataset_size))
            random.shuffle(idx)
            dump_pkl_file(idx,config.DATA_IDX_PKL_SAVE_PATH)
        train_sampler = data.sampler.SubsetRandomSampler(idx[:partial_size])
        train_gt_sampler = data.sampler.SubsetRandomSampler(idx[:partial_size])
        semi_sampler = data.sampler.SubsetRandomSampler(idx[partial_size:])

        train_full_loader = SOD_Dataloader(sal_dataset,batch_size=config.BATCH_SIZE, \
                                           sampler=train_sampler, num_workers=config.BATCH_SIZE // 2, pin_memory=True)
        gt_loader = SOD_Dataloader(gt_dataset,batch_size=config.BATCH_SIZE, \
                                           sampler=train_gt_sampler, num_workers=config.BATCH_SIZE // 2, pin_memory=True)
        semi_loader = SOD_Dataloader(sal_dataset,batch_size=config.BATCH_SIZE, \
                                           sampler=semi_sampler, num_workers=config.BATCH_SIZE // 2, pin_memory=True)

        train_full_wraper = TrainDataWrapper(
            train_full_loader,
            lambda_adv=config.LAMBDA_PRED_ADV,
            lambda_sal=config.LAMBDA_PRED_SAL,
        )
        gt_full_wraper = TrainDataWrapper(
            gt_loader,
            lambda_adv=None,
            lambda_sal=None,
        )
        semi_wrapper = TrainDataWrapper(
            semi_loader,
            lambda_adv=config.LAMBDA_SEMI_ADV,
            lambda_sal=config.LAMBDA_SEMI_SAL,
            mask_T=config.MASK_T,
            start_time=config.SEMI_START
        )
    else:
        train_full_loader = SOD_Dataloader(sal_dataset, batch_size=config.BATCH_SIZE, \
                                           num_workers=config.BATCH_SIZE // 2, pin_memory=True)
        gt_loader = SOD_Dataloader(gt_dataset, batch_size=config.BATCH_SIZE, \
                                   num_workers=config.BATCH_SIZE // 2, pin_memory=True)

        train_full_wraper = TrainDataWrapper(
            train_full_loader,
            lambda_adv=config.LAMBDA_PRED_ADV,
            lambda_sal=config.LAMBDA_PRED_SAL,
        )
        gt_full_wraper = TrainDataWrapper(
            gt_loader,
            lambda_adv=None,
            lambda_sal=None,
        )
        semi_wrapper = None

    sal_loss_function = BCELossWithLogits_Mask(ignore_value=sal_dataset.ignore_value).to(device)
    model_saver = ModelSaver(dataset_size // config.BATCH_SIZE, save_dir_path=config.MODEL_SAVE_PATH)
    adv_loss_func = BCELossWithLogits_Mask(ignore_value=sal_dataset.ignore_value).to(device)

    def G_optim_create_func( model: Generator, lr: float ):
        def get_params( sub_module: Module ):
            for module in sub_module.modules():
                if isinstance(module, nn.Conv2d):
                    for p in module.parameters():
                        yield p
        params = [
            {'params': get_params(model.base_model), 'lr': lr},
            {'params': get_params(model.aspp), 'lr': 10 * lr},
        ]
        optimizer = optim.SGD(
            params,
            lr=lr,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        return optimizer
    def G_adjust_lr( optimizer,lr, itr, max_itr ):
        now_lr = lr * (1 - itr / (max_itr + 1)) ** 0.9
        optimizer.param_groups[0]['lr'] = now_lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = 10 * now_lr
    G_optim_obj = Optimizer(G_optim_create_func, config.MAX_ITER,
                          step_time_interval=config.STEP_INTERVAL,
                            lr_schuduer=G_adjust_lr)
    def D_optim_create_func(model: FCDiscriminator, lr: float):
        return optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, betas=(0.9, 0.99))

    D_adjust_lr = G_adjust_lr
    D_optim_obj = Optimizer(D_optim_create_func, config.MAX_ITER,
                          step_time_interval=config.STEP_INTERVAL,
                            lr_schuduer=D_adjust_lr)

    trainer = Trainer(
        train_full_wraper,
        semi_wrapper,
        gt_full_wraper,
        adv_loss_func,
        sal_loss_function,
        config.MAX_ITER,
        ignore_value = sal_dataset.ignore_value,
        generator_optim_create_func = G_optim_obj,
        discriminator_optim_create_func = D_optim_obj,
        generate_lr = config.G_LEARNING_RATE,
        discriminator_lr= config.D_LEARNING_RATE,
        device=device,
        model_saver=model_saver,
        pretrained_model_path = config.PRETRAINED_MODEL_PATH,
        is_use_grab=config.USE_GRAB,
    )
    trainer.train()


def test(config: Configuration) -> None:
    if config.DISABLE_TEST:
        return
    dataset = SaliencyDataSet(config.DATASET_TEST_ROOT_DIR,config.DATASET_TEST_LIST_PATH,
                              is_scale=False,is_random_flip=False,crop_size=(None,None))
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    if config.USE_GPU is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tester = Tester(
        dataloader,config.TEST_IMG_SAVE_PATH,config.TEST_MODEL_PTH_PATH, \
        device=device,label_trans_func=lambda x:dataset.real_label(torch.sigmoid(x)).cpu().numpy().transpose((0,2,3,1) )
        )
    tester.test()



def evaluate(config: Configuration) -> None:
    if config.DISABLE_EVAL:
        return
    if config.USE_GPU is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sal_measure = get_measure(config.EVAL_MEASURE_LIST, config.EVALUATOR_DIR,config.DATASET_TEST_GT_DIR, device)
    measure = OrderedDict()
    measure['setting'] = config.MODEL_SAVE_DIR_NAME
    measure['dataset'] = config.DATASET_NAME
    measure.update(sal_measure)
    for key,item in measure.items():
        if not isinstance(item, Sequence):
            measure[key] = [item]
        else:
            measure[key] = [str(item)]
    table_content = pandas2markdown(pd.DataFrame(measure))
    record_file_dir = os.path.dirname(config.EVALUATOR_SUMMARY_FILE_PATH)
    mkdirs(record_file_dir)

    # Write xlsx report
    xlsx_filename = os.path.splitext(os.path.basename(config.EVALUATOR_SUMMARY_FILE_PATH))[0] + '.xlsx'
    xlsx_filepath = pathjoin(
        record_file_dir,
        xlsx_filename
    )
    data_dict = OrderedDict(**{
        'setting':str(config.MODEL_SAVE_DIR_NAME),
        'dataset':str(config.DATASET_NAME),
        'g_lr':str(config.G_LEARNING_RATE),
        'd_lr':str(config.D_LEARNING_RATE),
        'step_size':str(config.STEP_INTERVAL),
        'batch_size':str(config.BATCH_SIZE),
        'crop_size':str(config.CROP_SIZE),
        'partial_data':str(config.PARTIAL_DATA),
        'max_iter':str(config.MAX_ITER),
        'lambda_pred_sal':str(config.LAMBDA_PRED_SAL),
        'lambda_pred_adv':str(config.LAMBDA_PRED_ADV),
        'lambda_semi_sal':str(config.LAMBDA_SEMI_SAL),
        'lambda_semi_adv':str(config.LAMBDA_SEMI_ADV),
    })
    data_dict.update(sal_measure)
    record_dataframe = pd.DataFrame(data_dict,index=[0])
    if not os.path.exists(xlsx_filepath):
        record_dataframe.to_excel(xlsx_filepath,index=False)
    else:
        pd.concat([pd.read_excel(xlsx_filepath), record_dataframe],sort=False).to_excel(xlsx_filepath,index=False)

    title = config.MODEL_SAVE_DIR_NAME
    file_content = (f"\n"
                    f"# Experiment {title}  \n"
                    f"Time:{time.strftime('%Y-%m-%d %X')}  \n"
                    f"Dataset:{config.DATASET_NAME}  \n"
                    f"Test folder:{config.EVALUATOR_DIR}  \n"
                    f"Test index:{' '.join(config.EVAL_MEASURE_LIST)}  \n"
                    f"Command parameters:\n"
                    f"```bash\n"
                    f"{config.CMD_STR}\n"
                    f"```\n"
                    f"\n"
                    f"## Experimental results\n"
                    f"{table_content}\n")
    with portalocker.Lock(config.EVALUATOR_SUMMARY_FILE_PATH, 'a+', \
                          encoding='utf-8',timeout=600) as f:
        f.write(file_content)


def main():
    config = Configuration()
    if config.USE_GPU is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.USE_GPU
    if config.PROC_NAME is not None:
        from setproctitle import setproctitle
        setproctitle(config.PROC_NAME)
    train(config)
    for dataset_name in config.update_eval_list():
        print(f'handling {dataset_name}...')
        test(config)
        evaluate(config)
        print(f'evaluation on {dataset_name} has done')

if __name__ == '__main__':
    main()


