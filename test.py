from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import time
import cv2

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

from data_preprocess import ADE
from batchLoader import BatchLoader
from model import vgg16, res50
from opts import parse_args
from visualization import *

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def check_rootfolders(trainid):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    if not os.path.exists('./data/results'):
        os.makedirs('./data/results')
    for folder in folders_util:
        if not os.path.exists(os.path.join('./data/results', trainid, folder)):
            print('creating folder ' + folder)
            os.makedirs(os.path.join('./data/results', trainid, folder))

if __name__ == '__main__':

    args = parse_args()
    check_rootfolders(args.train_id)
    summary_w = tf and tf.summary.FileWriter(\
        os.path.join('./data/results', args.train_id, args.root_log))  # tensorboard
    print('Called with args:')
    print(args)

    np.random.seed(args.RNG_SEED)

    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, \
        so you should probably run with --cuda")

    pd_test = ADE('mtest', args)  # without flipper append
    args.CLASSES = pd_test.classes
    print('{:d} test roidb entries'.format(len(pd_test.roidb)))

    pd_test.filter_roidb()

    test_size = len(pd_test.roidb)
    dataloader = torch.utils.data.DataLoader(BatchLoader(pd_test.roidb, args, is_training=False), batch_size=1, \
                                             num_workers=args.num_workers, shuffle=False)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    if True:
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda:
        args.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        basenet = vgg16(pd_test.classes, args, pretrained=True)
    elif args.net == 'res50':
        basenet = res50(pd_test.classes, args, pretrained=True)
    else:
        print("network is not defined")

    basenet.create_architecture()

    load_name = os.path.join('./data/results', args.train_id, 'model', args.model_name)
    print("loading checkpoint %s" % load_name)
    checkpoint = torch.load(load_name)
    basenet.load_state_dict(checkpoint['model'])
    print("loaded checkpoint %s" % load_name)

    if args.cuda:
        basenet.cuda()

    iters_per_epoch = int(test_size / args.batch_size)

    total_iters = 1

    # eval model every epoch
    data_iter_val = iter(dataloader)
    basenet.eval()
    loss_tt = 0.
    start = time.time()
    all_scores = [[] for _ in range(len(pd_test.roidb))]
    for step in range(len(pd_test.roidb)):
        data = next(data_iter_val)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])

        cls_prob, cls_loss = basenet(im_data, im_info, gt_boxes)
        all_scores[step] = cls_prob.data.cpu().numpy()
        loss = cls_loss.mean()
        loss_tt += loss.data[0]

        if True and step % 10 == 0:
            basename = os.path.basename(pd_test.image_path_at(step)).split('.')[0]
            im_vis, wrong = draw_predicted_boxes_test(data[4].cpu().numpy()[0], all_scores[step], data[5].cpu().numpy()[0], args)
            if not os.path.exists(os.path.join('./data/images', args.train_id)):
                os.makedirs(os.path.join('./data/images', args.train_id))
            out_image = os.path.join('./data/images', args.train_id, basename + '.jpg')
            cv2.imwrite(out_image, im_vis)

        if step % args.disp_interval == 0:
            end = time.time()
            loss_rcnn_cls = cls_loss.data[0]
            sys.stdout.write(
                "evaling: [iter %4d/%4d] ; time cost: %f; rcnn_cls: %.4f\r" % (
                step, len(pd_test.roidb), end - start, loss_rcnn_cls))
            start = time.time()
    sys.stdout.flush()

    res_file = os.path.join('./data/results', args.train_id, args.root_output, 'all_scores.pkl')
    import cPickle as pickle
    with open(res_file, 'wb') as f:
        pickle.dump(all_scores, f, pickle.HIGHEST_PROTOCOL)
    print('all_scores saved!')
    print('Evaluating detections')
    mcls_sc, mcls_ac, mcls_ap, mins_sc, mins_ac, mins_ap = pd_test.evaluate(all_scores)
    eval_file = os.path.join('./data/results', args.train_id, args.root_output, 'results.txt')
    with open(eval_file, 'w') as f:
        f.write('{:.3f} {:.3f} {:.3f} {:.3f}'.format(mins_ap, mins_ac, mcls_ap, mcls_ac))

