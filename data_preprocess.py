# @Time    : 2018/4/27 9:47
# @File    : data_preprocess.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from voc_eval import voc_ap


try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
import cv2
import numpy as np


class ADE:
    def __init__(self, image_set, args, count=5):
        '''
        
        :param image_set: 
        :param count: threshold to filter the objects according to their counts 
        '''
        # imdb.__init__(self, 'ade_%s_%d' % (image_set, count))
        self.args = args
        self.cache_path = osp.join(self.args.DATA_DIR, 'cache')
        self._name = 'ade_%s_%d' % (image_set, count)
        self._image_set = image_set
        self._root_path = osp.join(self.args.DATA_DIR, 'ADE')
        self._name_file = osp.join(self._root_path, 'objectnames.txt')
        self._count_file = osp.join(self._root_path, 'objectcounts.txt')
        self._anno_file = osp.join(self._root_path, self._image_set + '.txt')  #train.txt
        with open(self._anno_file) as fid:
            image_index = fid.readlines()
            self._image_index = [ii.strip() for ii in image_index]
        with open(self._name_file) as fid:
            raw_names = fid.readlines()
            self._raw_names = [n.strip().replace(' ', '_') for n in raw_names]
            self._len_raw = len(self._raw_names)
        with open(self._count_file) as fid:
            raw_counts = fid.readlines()
            self._raw_counts = np.array([int(n.strip()) for n in raw_counts])

        # First class is always background
        self._ade_inds = [0] + list(np.where(self._raw_counts >= count)[0])
        self._classes = ['__background__']

        for idx in self._ade_inds:
            if idx == 0:
                continue
            ade_name = self._raw_names[idx]
            self._classes.append(ade_name)

        self._classes = tuple(self._classes)
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

        # load infos
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        self._roidb = self.gt_roidb()
        self.roidb = self._roidb
        if self._image_set == 'train':
            self.append_flipped_images()
        for i in range(len(self.image_index)):
            self.roidb[i]['image'] = self.image_path_at(i)

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def num_images(self):
        return len(self.image_index)

    @property
    def data_layer(self):
        return self._data_layer

    @property
    def minibatch(self):
        return self._minibatch

    def _load_text(self, text_path):
        class_keys = {}
        with open(text_path) as fid:
            lines = fid.readlines()
            for line in lines:
                columns = line.split('#')
                key = '%s_%s' % (columns[0].strip(), columns[1].strip())
                # Just get the class ID
                class_name = columns[4].strip().replace(' ', '_')
                if class_name in self._class_to_ind:
                    class_keys[key] = self._class_to_ind[class_name]
            total_num_ins = len(lines)

        return class_keys, total_num_ins

    def _load_annotation(self):
        gt_roidb = []

        for i in range(self.num_images):
            image_path = self.image_path_at(i)
            if i % 10 == 0:
                print(image_path)
            # Estimate the number of objects from text file
            text_path = image_path.replace('.jpg', '_atr.txt')
            class_keys, total_num_ins = self._load_text(text_path)

            valid_num_ins = 0
            boxes = np.zeros((total_num_ins, 4), dtype=np.uint16)
            gt_classes = np.zeros((total_num_ins), dtype=np.int32)
            seg_areas = np.zeros((total_num_ins), dtype=np.float32)

            # First, whole objects
            label_path = image_path.replace('.jpg', '_seg.png')
            seg = cv2.imread(label_path)
            height, width, _ = seg.shape

            # OpenCV has reversed RGB
            instances = seg[:, :, 0]
            unique_ins = np.unique(instances)

            for t, ins in enumerate(list(unique_ins)):
                if ins == 0:
                    continue
                key = '%03d_%d' % (t, 0)
                if key in class_keys:
                    ins_seg = np.where(instances == ins)
                    x1 = ins_seg[1].min()
                    x2 = ins_seg[1].max()
                    y1 = ins_seg[0].min()
                    y2 = ins_seg[0].max()
                    boxes[valid_num_ins, :] = [x1, y1, x2, y2]
                    gt_classes[valid_num_ins] = class_keys[key]
                    seg_areas[valid_num_ins] = ins_seg[0].shape[0]
                    valid_num_ins += 1

            # Then deal with parts
            level = 1
            while True:
                part_path = image_path.replace('.jpg', '_parts_%d.png' % level)
                if osp.exists(part_path):
                    seg = cv2.imread(part_path)
                    instances = seg[:, :, 0]
                    unique_ins = np.unique(instances)

                    for t, ins in enumerate(list(unique_ins)):
                        if ins == 0:
                            continue
                        key = '%03d_%d' % (t, level)
                        if key in class_keys:
                            ins_seg = np.where(instances == ins)
                            x1 = ins_seg[1].min()
                            x2 = ins_seg[1].max()
                            y1 = ins_seg[0].min()
                            y2 = ins_seg[0].max()
                            boxes[valid_num_ins, :] = [x1, y1, x2, y2]
                            gt_classes[valid_num_ins] = class_keys[key]
                            seg_areas[valid_num_ins] = ins_seg[0].shape[0]
                            valid_num_ins += 1

                    level += 1
                else:
                    break

            boxes = boxes[:valid_num_ins, :]
            gt_classes = gt_classes[:valid_num_ins]
            seg_areas = seg_areas[:valid_num_ins]

            gt_roidb.append({'width': width,
                             'height': height,
                             'boxes': boxes,
                             'gt_classes': gt_classes,
                             'flipped': False,
                             'seg_areas': seg_areas})
        return gt_roidb

    def image_path_at(self, i):
        return osp.join(self._root_path, self._image_index[i])

    def gt_roidb(self):
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        image_file = osp.join(self.cache_path, self.name + '_gt_image.pkl')
        if osp.exists(cache_file) and osp.exists(image_file):
            with open(cache_file, 'rb') as fid:
                gt_roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            with open(image_file, 'rb') as fid:
                self._image_index = pickle.load(fid)
            print('{} gt image loaded from {}'.format(self.name, image_file))
            return gt_roidb

        gt_roidb = self._load_annotation()
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        with open(image_file, 'wb') as fid:
            pickle.dump(self._image_index, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt image to {}'.format(image_file))
        return gt_roidb

    # Do some left-right flipping here
    def _find_flipped_classes(self):
        self._flipped_classes = np.arange(self.num_classes, dtype=np.int32)
        for i, cls_name in enumerate(self.classes):
            if cls_name.startswith('left_'):
                query = cls_name.replace('left_', 'right_')
                idx = self._class_to_ind[query]
                # Swap for both left and right
                self._flipped_classes[idx] = i
                self._flipped_classes[i] = idx

    def _get_widths(self):
        return [r['width'] for r in self.roidb]

    def append_flipped_images(self):
        print('Appending horizontally-flipped training examples...')
        self._find_flipped_classes()
        num_images = self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'width': widths[i],
                     'height': self.roidb[i]['height'],
                     'boxes': boxes,
                     'gt_classes': self._flipped_classes[self.roidb[i]['gt_classes']],
                     'flipped': True,
                     'seg_areas': self.roidb[i]['seg_areas']}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
        print('done')

    def filter_roidb(self):
        """Remove roidb entries that have no usable RoIs."""

        def is_valid(entry):
            # Valid images have at least one ground truth labeled
            valid = len(entry['gt_classes']) > 0
            return valid

        num = len(self.roidb)
        filtered_roidb = [entry for entry in self.roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                           num, num_after))
        self.roidb = filtered_roidb

    def _score(self, all_scores):
        scs = [0.] * self.num_classes
        scs_all = [0.] * self.num_classes
        valid = [0] * self.num_classes
        for i in range(1, self.num_classes):
            ind_this = np.where(self.gt_classes == i)[0]
            scs_all[i] = np.sum(all_scores[ind_this, i])
            if ind_this.shape[0] > 0:
                valid[i] = ind_this.shape[0]
                scs[i] = scs_all[i] / ind_this.shape[0]

        mcls_sc = np.mean([s for s, v in zip(scs, valid) if v])
        mins_sc = np.sum(scs_all) / self.gt_classes.shape[0]
        return scs[1:], mcls_sc, mins_sc, valid[1:]

    def _accuracy(self, all_scores):
        acs = [0.] * self.num_classes
        acs_all = [0.] * self.num_classes
        valid = [0] * self.num_classes

        # Need to remove the background class
        max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1
        max_scores = np.empty_like(all_scores)
        max_scores[:] = 0.
        max_scores[np.arange(self.gt_classes.shape[0]), max_inds] = 1.

        for i in range(1, self.num_classes):
            ind_this = np.where(self.gt_classes == i)[0]
            acs_all[i] = np.sum(max_scores[ind_this, i])
            if ind_this.shape[0] > 0:
                valid[i] = ind_this.shape[0]
                acs[i] = acs_all[i] / ind_this.shape[0]

        mcls_ac = np.mean([s for s, v in zip(acs, valid) if v])
        mins_ac = np.sum(acs_all) / self.gt_classes.shape[0]
        return acs[1:], mcls_ac, mins_ac

    def _average_precision(self, all_scores):
        aps = [0.] * self.num_classes
        valid = [0] * self.num_classes

        ind_all = np.arange(self.gt_classes.shape[0])
        num_cls = self.num_classes
        num_ins = ind_all.shape[0]

        for i, c in enumerate(self._classes):
            if i == 0:
                continue
            gt_this = (self.gt_classes == i).astype(np.float32)
            num_this = np.sum(gt_this)
            if i % 10 == 0:
                print('AP for %s: %d/%d' % (c, i, num_cls))
            if num_this > 0:
                valid[i] = num_this
                sco_this = all_scores[ind_all, i]

                ind_sorted = np.argsort(-sco_this)

                tp = gt_this[ind_sorted]
                max_ind = num_ins - np.argmax(tp[::-1])
                tp = tp[:max_ind]
                fp = 1. - tp

                tp = np.cumsum(tp)
                fp = np.cumsum(fp)
                rec = tp / float(num_this)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

                aps[i] = voc_ap(rec, prec)

        mcls_ap = np.mean([s for s, v in zip(aps, valid) if v])

        # Compute the overall score
        max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1
        max_scores = np.empty_like(all_scores)
        max_scores[:] = 0.
        max_scores[ind_all, max_inds] = 1.
        pred_all = max_scores[ind_all, self.gt_classes]
        sco_all = all_scores[ind_all, self.gt_classes]
        ind_sorted = np.argsort(-sco_all)

        tp = pred_all[ind_sorted]
        fp = 1. - tp

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(num_ins)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        mins_ap = voc_ap(rec, prec)
        return aps[1:], mcls_ap, mins_ap

    def evaluate(self, all_scores, roidb=None):
        if roidb is None:
            roidb = self.roidb
        all_scores = np.vstack(all_scores)
        # all_scores = np.minimum(all_scores, 1.0)
        self.gt_classes = np.hstack([r['gt_classes'] for r in roidb])

        scs, mcls_sc, mins_sc, valid = self._score(all_scores)
        acs, mcls_ac, mins_ac = self._accuracy(all_scores)
        aps, mcls_ap, mins_ap = self._average_precision(all_scores)

        for i, cls in enumerate(self._classes):
            if cls == '__background__' or not valid[i - 1]:
                continue
            print(('{} {:d} {:.4f} {:.4f} {:.4f}'.format(cls,
                                                         valid[i - 1],
                                                         scs[i - 1],
                                                         acs[i - 1],
                                                         aps[i - 1])))

        print('~~~~~~~~')
        # print('Scores | Accuracies | APs:')
        # for sc, ac, ap, vl in zip(scs, acs, aps, valid):
        #   if vl:
        #     print(('{:.3f} {:.3f} {:.3f}'.format(sc, ac, ap)))
        print(('mean-cls: {:.3f} {:.3f} {:.3f}'.format(mcls_sc, mcls_ac, mcls_ap)))
        print(('mean-ins: {:.3f} {:.3f} {:.3f}'.format(mins_sc, mins_ac, mins_ap)))
        print('~~~~~~~~')
        print(('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(mcls_sc,
                                                                  mcls_ac,
                                                                  mcls_ap,
                                                                  mins_sc,
                                                                  mins_ac,
                                                                  mins_ap)))
        print('~~~~~~~~')

        return mcls_sc, mcls_ac, mcls_ap, mins_sc, mins_ac, mins_ap
