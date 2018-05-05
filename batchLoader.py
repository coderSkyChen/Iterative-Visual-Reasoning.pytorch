# @Time    : 2018/3/27 10:59
# @File    : roibatchLoader.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

import cv2
import numpy as np
import numpy.random as npr
import torchvision.transforms as transforms
from scipy.misc import imread
import random
import time
import pdb


class BatchLoader(data.Dataset):
    def __init__(self, roidb, args, is_training):
        """Set the roidb to be used by this layer during training."""
        self.roidb = roidb
        self.args = args
        self.is_train_val = is_training

    def get_one_sample(self, index):
        num_images = 1
        # Sample random scales to use for each image in this batch
        scales = (600,)
        max_scale = 600
        random_scale_inds = npr.randint(0, high=len(scales), size=num_images)

        # Get the input image blob
        im_blob, im_scales = self.get_image_blob(index, random_scale_inds, scales, max_scale)

        blobs = {'data': im_blob}

        # gt boxes: (x1, y1, x2, y2, cls)

        gt_inds = np.where(self.roidb[index[0]]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = self.roidb[index[0]]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = self.roidb[index[0]]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        # height, width, scale
        blobs['im_info'] = np.array([im_blob.shape[1],
                                     im_blob.shape[2],
                                     im_scales[0]], dtype=np.float32)
        blobs['memory_size'] = np.ceil(blobs['im_info'][:2] / self.args.BOTTLE_SCALE).astype(np.int32)
        blobs['num_gt'] = np.int32(gt_boxes.shape[0])
        blobs['img'] = self.roidb[index[0]]['image']
        blobs['index'] = index[0]
        return blobs

    def get_image_blob(self, index, scale_inds, scales, max_scale):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = 1
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(self.roidb[index[i]]['image'])
            if self.roidb[index[i]]['flipped']:
                im = im[:, ::-1, :]
            target_size = scales[scale_inds[i]]
            im, im_scale = self.prep_im_for_blob_pytorch(im, target_size,
                                            max_scale)
            im_scales.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self.im_list_to_blob(processed_ims)

        return blob, im_scales

    def im_list_to_blob(self, ims):
        """Convert a list of images into a network input.
  
        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob

    def prep_im_for_blob_caffe(self, im, pixel_means, target_size, max_size):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    def prep_im_for_blob_pytorch(self, im, target_size, max_size):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im /= 255.  # Convert range to [0,1]
        im -= np.array([[[0.485, 0.456, 0.406]]])  # Minus mean
        im /= np.array([[[0.229, 0.224, 0.225]]])  # divide by stddev
        im = im[:, :, ::-1]  # BGR to RGB

        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    def __getitem__(self, index):
        indexes = [index]
        blobs = self.get_one_sample(indexes)
        data = torch.from_numpy(blobs['data'])
        im_info = torch.from_numpy(blobs['im_info'])
        mem_size = torch.from_numpy(blobs['memory_size'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)

        if self.is_train_val:
            # if the number of region is greater than 100 then random pick 100 regions
            # this opt can make the used memory of GPUs more stable.
            # only for train and val phase
            np.random.shuffle(blobs['gt_boxes'])
            if blobs['gt_boxes'].shape[0] > 100:
                print('sampling regions from %d to %d' % (blobs['gt_boxes'].shape[0], 100))
                blobs['gt_boxes'] = blobs['gt_boxes'][:100]
        else:
            pass
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])
        # permute trim_data to adapt to downstream processing
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)
        return data, im_info, gt_boxes, mem_size, blobs['data'], blobs['gt_boxes']

    def __len__(self):
        return len(self.roidb)
