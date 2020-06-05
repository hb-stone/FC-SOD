from helper.utils import EasyDict
from helper.utils import pathjoin
from argparse import ArgumentParser
import os
from dataset.file_table import get_dataset_path_by_name
import threading


class Configuration(EasyDict):
    _instance_lock = threading.Lock()

    def __new__( cls, *args, **kwargs ):
        if not hasattr(Configuration, "_instance"):
            with Configuration._instance_lock:
                if not hasattr(Configuration, "_instance"):
                    Configuration._instance = EasyDict.__new__(cls)
                    Configuration._instance.init(*args, **kwargs)
        return Configuration._instance

    """
    Configuration class for system config
    """
    def init(self, use_arg_parser=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Run configuration
        self.DISABLE_TRAIN = False
        self.DISABLE_TEST = False
        self.DISABLE_EVAL = False


        # dataset configuration
        self.DATASET_TRAIN_ROOT_DIR = None
        self.DATASET_TRAIN_LIST_PATH = None
        self.DATASET_TEST_ROOT_DIR = None
        self.DATASET_TEST_LIST_PATH = None
        self.DATASET_NAME = None
        self.DATASET_TRAIN_DIR = None
        self.DATASET_TEST_DIR = None

        # Training configuration
        self.SEED = 2345
        self.BATCH_SIZE = 10
        self.CROP_SIZE = 321
        self.G_LEARNING_RATE = 2.5e-4
        self.D_LEARNING_RATE = 1e-4
        self.MAX_ITER = 25000
        self.MOMENTUM = 0.9
        self.PRETRAINED_MODEL_PATH = None
        self.STEP_INTERVAL = 1
        self.WEIGHT_DECAY = 0.0005
        self.DISABLE_VISUAL = False
        self.DATA_IDX_PKL_PATH = None
        self.SEMI_START = 5000
        self.MASK_T = 0.2
        self.PARTIAL_DATA = 0.125
        self.LAMBDA_SEMI_ADV = 0.007
        self.LAMBDA_SEMI_SAL = 1
        self.LAMBDA_PRED_SAL = 1
        self.LAMBDA_PRED_ADV = 0.01
        self.USE_GRAB = False
        # Graphics card use configuration
        self.USE_GPU = None

        # Model save settings
        self.MODEL_SAVE_ROOT_DIR = None
        self.MODEL_SAVE_DIR_NAME = None
        self.LOG_DIR = pathjoin(
            os.path.dirname(__file__),
            'logs',
        )
        self.DATA_IDX_PKL_SAVE_PATH = pathjoin(
            self.LOG_DIR,
            'idx.pkl',
        )

        # test setting
        self.TEST_MODEL_PTH_PATH = None

        # eval setting
        self.EVALUATOR_DIR = None
        self.EVAL_MEASURE_LIST = None
        self.EVALUATOR_SUMMARY_FILE_PATH = None
        self.EVALUATOR_DATASETS = None

        # proc_name setting
        self.PROC_NAME = None



        parser = ArgumentParser(description="Configuration of the System")

        parser.add_argument('-disable_test', '--DISABLE_TEST', action='store_true', \
                            default=self.DISABLE_TEST)
        parser.add_argument('-disable_train', '--DISABLE_TRAIN',  action='store_true', \
                            default=self.DISABLE_TRAIN)
        parser.add_argument('-disable_eval', '--DISABLE_EVAL', action='store_true', \
                            default=self.DISABLE_EVAL)
        parser.add_argument('-disable_visual', '--DISABLE_VISUAL', action='store_true', \
                            default=self.DISABLE_VISUAL)
        parser.add_argument('-seed','--SEED',type=int,default=self.SEED)
        parser.add_argument('-d', '--DATASET_NAME', required=True, type=str, \
                            default=self.DATASET_NAME, \
                            choices="DUT-OMRON DUTS PASCAL-S SOD".split(" "))

        parser.add_argument('-b', '--BATCH_SIZE', type=int, default=self.BATCH_SIZE)
        parser.add_argument('-crop', '--CROP_SIZE', type=int, default=self.CROP_SIZE)
        parser.add_argument('-g_lr', '--G_LEARNING_RATE',type=float, default=self.G_LEARNING_RATE)
        parser.add_argument('-d_lr', '--D_LEARNING_RATE',type=float, default=self.D_LEARNING_RATE)
        parser.add_argument('-max_iter', '--MAX_ITER', type=int, default=self.MAX_ITER)
        parser.add_argument('-step', '--STEP_INTERVAL', type=int, default=self.STEP_INTERVAL)
        parser.add_argument('-momentum', '--MOMENTUM', type=float, default=self.MOMENTUM)
        parser.add_argument('-weight_decay', '--WEIGHT_DECAY', type=float, default=self.WEIGHT_DECAY)
        parser.add_argument('-idx', '--DATA_IDX_PKL_PATH', type=str, default=self.DATA_IDX_PKL_PATH)
        parser.add_argument('-start', '--SEMI_START', type=int, default=self.SEMI_START)
        parser.add_argument('-mask_T', '--MASK_T', type=float, default=self.MASK_T)
        parser.add_argument('-part', '--PARTIAL_DATA', type=float, default=self.PARTIAL_DATA)
        parser.add_argument('-grab', '--USE_GRAB', type=bool, default=self.USE_GRAB)
        parser.add_argument('-l_semi_adv', '--LAMBDA_SEMI_ADV', type=float, default=self.LAMBDA_SEMI_ADV)
        parser.add_argument('-l_semi_sal', '--LAMBDA_SEMI_SAL', type=float, default=self.LAMBDA_SEMI_SAL)
        parser.add_argument('-l_pred_adv', '--LAMBDA_PRED_ADV', type=float, default=self.LAMBDA_PRED_ADV)
        parser.add_argument('-l_pred_sal', '--LAMBDA_PRED_SAL', type=float, default=self.LAMBDA_PRED_SAL)
        parser.add_argument('-pretrain', '--PRETRAINED_MODEL_PATH', type=str, \
                            default=self.PRETRAINED_MODEL_PATH)


        parser.add_argument('-gpu', '--USE_GPU', type=str, default=self.USE_GPU)
        parser.add_argument('-proc', '--PROC_NAME', type=str, default=self.PROC_NAME)
        parser.add_argument('-test_model', '--TEST_MODEL_PTH_PATH', type=str, default=self.TEST_MODEL_PTH_PATH)
        parser.add_argument('-save', '--MODEL_SAVE_DIR_NAME', required=True, type=str, \
                            default=self.MODEL_SAVE_DIR_NAME)
        parser.add_argument('-eval_dir', '--EVALUATOR_DIR', type=str, default=self.EVALUATOR_DIR)
        parser.add_argument('-eval_m_list', '--EVAL_MEASURE_LIST', nargs='+', \
                            type=str, choices="max-F mean-F MAE S precision recall".split(" "),default=self.EVAL_MEASURE_LIST)
        parser.add_argument('-eval_file', '--EVALUATOR_SUMMARY_FILE_PATH', \
                            type=str, default=self.EVALUATOR_SUMMARY_FILE_PATH)
        parser.add_argument('-eval_d', '--EVALUATOR_DATASETS', nargs='+', \
                            type=str, choices="ALL DUT-OMRON DUTS PASCAL-S SOD".split(" "), \
                            default=self.EVALUATOR_DATASETS)


        if use_arg_parser:
            args = parser.parse_args()
            self.update(args.__dict__)
        # handle MODEL_SAVE_PATH
        self.MODEL_SAVE_ROOT_DIR = pathjoin(
            self.LOG_DIR,
            self.MODEL_SAVE_DIR_NAME,
        )
        self.MODEL_SAVE_PATH = pathjoin(
            self.MODEL_SAVE_ROOT_DIR,
            'save_models',
        )
 
        if self.BATCH_SIZE != 1:
            assert self.CROP_SIZE is not None, "CROP_SIZE can't be null, if " \
                                               "BATCH_SIZE != 1, please use -crop to specify the paramter"
        # handle EVAL_MEASURE_LIST
        if self.EVAL_MEASURE_LIST is None:
            self.EVAL_MEASURE_LIST = ["max-F","MAE","S"]
        # handle DATASET_ROOT_DIR DATASET_LIST_PATH
        tmp_dict = get_dataset_path_by_name(self.DATASET_NAME)
        self.DATASET_TRAIN_ROOT_DIR = tmp_dict['train_dir_path']
        self.DATASET_TRAIN_LIST_PATH = tmp_dict['train_lst_path']
        self.DATASET_TEST_ROOT_DIR = tmp_dict['test_dir_path']
        self.DATASET_TEST_LIST_PATH = tmp_dict['test_lst_path']
        self.DATASET_TRAIN_DIR = tmp_dict['train_dir_name']
        self.DATASET_TEST_DIR = tmp_dict['test_dir_name']
        self.DATASET_TEST_GT_DIR = pathjoin(
            self.DATASET_TEST_ROOT_DIR,
            'GT'
        )
        if self.TEST_MODEL_PTH_PATH is None:
            self.TEST_MODEL_PTH_PATH = pathjoin(
                self.MODEL_SAVE_PATH,
                'G-final.pth'
            )
        self.TEST_IMG_SAVE_PATH = pathjoin(
            self.MODEL_SAVE_ROOT_DIR,
            'test',
            self.DATASET_TEST_DIR if len(self.DATASET_TEST_DIR) != 0 else self.DATASET_NAME
        )
        
        # handle EVALUATOR_DIR
        if self.EVALUATOR_DIR is None:
            self.EVALUATOR_DIR = self.TEST_IMG_SAVE_PATH
        if self.EVALUATOR_SUMMARY_FILE_PATH is None:
            self.EVALUATOR_SUMMARY_FILE_PATH = pathjoin(
                self.LOG_DIR,
                'ExperimentalNotes.md'
            )
        else:
            self.EVALUATOR_SUMMARY_FILE_PATH = pathjoin(
                self.LOG_DIR,
                self.EVALUATOR_SUMMARY_FILE_PATH
            )
        if self.EVALUATOR_DATASETS is not None:
            if "ALL" in self.EVALUATOR_DATASETS:
                self.EVALUATOR_DATASETS = "DUT-OMRON DUTS PASCAL-S SOD".split()
            self.EVALUATOR_GTS = []
            self.TEST_IMG_SAVE_PATHS = []
            self.DATASET_TEST_ROOT_DIRS = []
            self.DATASET_TEST_LIST_PATHS = []
            for dataset_name in self.EVALUATOR_DATASETS:
                dataset_info = get_dataset_path_by_name(dataset_name)
                self.EVALUATOR_GTS.append(pathjoin(dataset_info['test_dir_path'],'GT'))
                dataset_test_dir_name = dataset_info['test_dir_name']
                testpath = pathjoin(
                    self.MODEL_SAVE_ROOT_DIR,
                    'test',
                    dataset_test_dir_name if len(dataset_test_dir_name) != 0 else dataset_name
                )
                self.TEST_IMG_SAVE_PATHS.append(testpath)
                self.DATASET_TEST_ROOT_DIRS.append(dataset_info['test_dir_path'])
                self.DATASET_TEST_LIST_PATHS.append(dataset_info['test_lst_path'])
            self.EVALUATOR_DIRS = self.TEST_IMG_SAVE_PATHS
            if self.EVALUATOR_DIR not in self.EVALUATOR_DIRS:
                self.EVALUATOR_DIRS.append(self.EVALUATOR_DIR)
                self.EVALUATOR_GTS.append(self.DATASET_TEST_GT_DIR)
                self.DATASET_TEST_ROOT_DIRS.append(self.DATASET_TEST_ROOT_DIR)
                self.DATASET_TEST_LIST_PATHS.append(self.DATASET_TEST_LIST_PATH)
                self.EVALUATOR_DATASETS.append(self.DATASET_NAME)

        else:
            self.EVALUATOR_DIRS = [self.TEST_IMG_SAVE_PATH]
            self.EVALUATOR_GTS = [self.DATASET_TEST_GT_DIR]
            self.TEST_IMG_SAVE_PATHS = [self.TEST_IMG_SAVE_PATH]
            self.DATASET_TEST_ROOT_DIRS = [self.DATASET_TEST_ROOT_DIR]
            self.DATASET_TEST_LIST_PATHS = [self.DATASET_TEST_LIST_PATH]
            self.EVALUATOR_DATASETS = [self.DATASET_NAME]

        import sys
        # Command parameter acquisition
        self.CMD_STR = "python " + " ".join(sys.argv)

    def update_eval_list(self):
        for i,dataset_name in enumerate(self.EVALUATOR_DATASETS):
            self.DATASET_NAME = dataset_name
            self.EVALUATOR_DIR = self.EVALUATOR_DIRS[i]
            self.DATASET_TEST_GT_DIR = self.EVALUATOR_GTS[i]
            self.TEST_IMG_SAVE_PATH = self.TEST_IMG_SAVE_PATHS[i]
            self.DATASET_TEST_ROOT_DIR = self.DATASET_TEST_ROOT_DIRS[i]
            self.DATASET_TEST_LIST_PATH = self.DATASET_TEST_LIST_PATHS[i]
            yield dataset_name

