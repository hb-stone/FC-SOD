from torch.utils import data
import os
import cv2
import numpy as np

__ALL__ = ["EvalDataloader"]

class EvalDataset(data.Dataset):
    def __init__( self, img_root, label_root):
        self.image_path = list()
        self.label_path = list()
        exts = ['.jpg', '.png', '.bmp']
        for image_name in os.listdir(label_root):
            image_path = os.path.join(img_root, image_name)
            label_path = os.path.join(label_root, image_name)
            if os.path.exists(image_path):
                self.image_path.append(image_path)
                self.label_path.append(label_path)
            else:
                basename = os.path.splitext(image_name)[0]
                for ext in exts:
                    image_path = os.path.join(img_root, basename + ext)
                    if os.path.exists(image_path):
                        break
                if os.path.exists(image_path):
                    self.image_path.append(image_path)
                    self.label_path.append(label_path)
        assert len(self.label_path) != 0, "label's dir shouldn't be empty!"
        del_num = 0

        for image_path, label_path in zip(self.image_path, self.label_path):
            pred = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            if pred.shape != gt.shape:
                self.image_path.remove(image_path)
                self.label_path.remove(label_path)
                del_num += 1

        assert (del_num < 10), "to many pic is not paired"

    def __getitem__( self, item ):
        pred = cv2.imread(self.image_path[item],cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(self.label_path[item],cv2.IMREAD_GRAYSCALE)
        # if pred.shape != gt.shape:
        #     print(self.image_path[item],self.label_path[item],'is not paired')
        #     pred = cv2.resize(pred, gt.shape, interpolation=cv2.INTER_LINEAR)
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        pred /= 255.0
        gt /= 255.0
        return pred, gt

    def __len__( self ):
        return len(self.image_path)


class EvalDataloader(data.DataLoader):

    def __init__(self, img_root, label_root, *args, **kwargs):
        dataset = EvalDataset(img_root,label_root)
        super(EvalDataloader,self).__init__(dataset, batch_size=1, shuffle=False, num_workers=0)

