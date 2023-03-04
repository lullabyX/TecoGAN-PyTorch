import os
import os.path as osp

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset
from utils.base_utils import retrieve_files

import config


def center_crop(img, dim=(128,128)):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

class PairedFolderDataset(BaseDataset):
    """ Folder dataset for paired data. It supports both BI & BD degradation.
    """

    def __init__(self, data_opt, **kwargs):
        super(PairedFolderDataset, self).__init__(data_opt, **kwargs)

        # get keys
        gt_keys = sorted(os.listdir(self.gt_seq_dir))
        lr_keys = sorted(os.listdir(self.lr_seq_dir))
        self.keys = sorted(list(set(gt_keys) & set(lr_keys)))

        # filter keys
        # sel_keys = set(self.keys)
        # if hasattr(self, 'filter_file') and self.filter_file is not None:
        #     with open(self.filter_file, 'r') as f:
        #         sel_keys = {line.strip() for line in f}
        # elif hasattr(self, 'filter_list') and self.filter_list is not None:
        #     sel_keys = set(self.filter_list)
        # self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load gt frames
        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.gt_seq_dir, key)):
            frm = cv2.imread(frm_path)
            if config.center_crop == 'yes':
                 frm = center_crop(frm, (128, 128))
            else:     
                frm = cv2.resize(frm, (128, 128), interpolation=cv2.INTER_AREA)
            frm = frm[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            gt_seq.append(frm)
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8

        # load lr frames
        lr_seq = []
        for frm_path in retrieve_files(osp.join(self.lr_seq_dir, key)):
            frm = cv2.imread(frm_path)
            if config.center_crop == 'yes':
                 frm = center_crop(frm, (128, 128))
            else:     
                frm = cv2.resize(frm, (128, 128), interpolation=cv2.INTER_AREA)
            frm = frm[..., ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            lr_seq.append(frm)
        lr_seq = np.stack(lr_seq)  # thwc|rgb|float32

        # convert to tensor
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8 -> now float32
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq))  # float32

        # gt: thwc|rgb|uint8 | lr: thwc|rgb|float32
        return {
            'gt': gt_tsr,
            'lr': lr_tsr,
            'seq_idx': key,
            'frm_idx': sorted(os.listdir(osp.join(self.gt_seq_dir, key)))
        }
