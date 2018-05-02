# @Time    : 2018/3/27 14:39
# @File    : model.py
# @Author  : Sky chen
# @Email   : dzhchxk@126.com
# @Personal homepage  : https://coderskychen.cn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import torchvision.models as models
from torch.autograd import Variable
from  torch.nn import init
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_align.crop_and_resize import CropAndResizeFunction

try:
    long  # Python 2
except NameError:
    long = int  # Python


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)


def compute_target_memory(memory_size, gt_boxes, feat_stride):
    memory_size = memory_size.data
    minus_h = memory_size[0] - 1.
    minus_w = memory_size[1] - 1.

    # gt_boxes = gt_boxes.detach()
    x1 = gt_boxes[:, [0]] / feat_stride
    y1 = gt_boxes[:, [1]] / feat_stride
    x2 = gt_boxes[:, [2]] / feat_stride
    y2 = gt_boxes[:, [3]] / feat_stride
    # h, w, h, w
    rois = torch.cat([y1, x1, y2, x2], 1)
    # normalize
    rois[:, 0::2] /= minus_h
    rois[:, 1::2] /= minus_w

    # h, w, h, w
    inv_rois = torch.zeros(*rois.size()).type_as(rois)
    # inv_rois = np.empty_like(rois)
    inv_rois[:, 0:2] = 0.
    inv_rois[:, 2] = minus_h
    inv_rois[:, 3] = minus_w
    inv_rois[:, 0::2] -= y1
    inv_rois[:, 1::2] -= x1

    # normalize coordinates
    inv_rois[:, 0::2] /= torch.max(y2 - y1, torch.ones_like(y1).type_as(y1) * (1e-14))
    inv_rois[:, 1::2] /= torch.max(x2 - x1, torch.ones_like(y1).type_as(y1) * (1e-14))
    return rois, inv_rois


class vgg16(nn.Module):
    def __init__(self, classes, args, pretrained=False):
        super(vgg16, self).__init__()
        self.model_path = args.backbone_path
        self.dout_base_model = 512
        self.pretrained = pretrained

        self.classes = classes
        self.n_classes = len(classes)
        # loss
        self._cls_loss = 0
        self.args = args

    def _crop_and_resize(self, bottom, rois, max_pool=False):
        # implement it using stn
        # box to affine
        # input (x1,y1,x2,y2)
        """
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1      ]
        """
        rois = rois.detach()

        x1 = rois[:, 1::4] / 16.0
        y1 = rois[:, 2::4] / 16.0
        x2 = rois[:, 3::4] / 16.0
        y2 = rois[:, 4::4] / 16.0

        height = bottom.size(2)
        width = bottom.size(3)

        pre_pool_size = self.args.POOLING_SIZE * 2 if max_pool else self.args.POOLING_SIZE
        crops = CropAndResizeFunction(pre_pool_size, pre_pool_size)(bottom,
                                                                    torch.cat([y1 / (height - 1), x1 / (width - 1),
                                                                               y2 / (height - 1), x2 / (width - 1)], 1),
                                                                    rois[:, 0].int())
        if max_pool:
            crops = F.max_pool2d(crops, 2, 2)

        return crops

    def forward(self, im_data, im_info, gt_boxes):
        gt_boxes = gt_boxes.data  # 3D
        # feed image data to base model to obtain base feature map
        base_feat = self.Conv_base(im_data)  # [1, 1024, 37, 50]

        # if self.training:
        if True:
            rois_label = Variable(gt_boxes[:, :, -1].contiguous().view(-1).long())
        else:
            rois_label = None

        rois = Variable(gt_boxes[:, :, 0:4])
        ss = gt_boxes.size()
        rr = torch.Tensor(ss[0], ss[1], 5).zero_().type_as(rois.data)
        rr[:, :, 1:5] = rois.data[:, :, 0:4]
        rr = Variable(rr)

        pooled_feat = self._crop_and_resize(base_feat, rr.contiguous().view(-1, 5), max_pool=False)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute object classification probability
        cls_score = self.Net_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        # classification loss
        loss = F.cross_entropy(cls_score, rois_label)

        cls_prob = cls_prob.view(rois.size(1), -1)

        return cls_prob, loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.Net_cls_score, 0, 0.01, self.args.TRUNCATED)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.Conv_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.Conv_base[layer].parameters(): p.requires_grad = False

        self.Conv_top = vgg.classifier

        # not using the last maxpool layer
        self.Net_cls_score = nn.Linear(4096, self.n_classes)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.Conv_top(pool5_flat)
        return fc7


class res50(nn.Module):
    def __init__(self, classes, args, pretrained=False):
        super(res50, self).__init__()
        self.model_path = args.backbone_path
        self.dout_base_model = 1024
        self.pretrained = pretrained

        self.classes = classes
        self.n_classes = len(classes)
        self.args = args

    def _crop_and_resize(self, bottom, rois, max_pool=False):
        # implement it using stn
        # box to affine
        # input (x1,y1,x2,y2)
        """
        [  x2-x1             x1 + x2 - W + 1  ]
        [  -----      0      ---------------  ]
        [  W - 1                  W - 1       ]
        [                                     ]
        [           y2-y1    y1 + y2 - H + 1  ]
        [    0      -----    ---------------  ]
        [           H - 1         H - 1      ]
        """
        rois = rois.detach()

        x1 = rois[:, 1::4] / 16.0
        y1 = rois[:, 2::4] / 16.0
        x2 = rois[:, 3::4] / 16.0
        y2 = rois[:, 4::4] / 16.0

        height = bottom.size(2)
        width = bottom.size(3)

        pre_pool_size = self.args.POOLING_SIZE * 2 if max_pool else self.args.POOLING_SIZE
        crops = CropAndResizeFunction(pre_pool_size, pre_pool_size)(bottom,
                                                                    torch.cat([y1 / (height - 1), x1 / (width - 1),
                                                                               y2 / (height - 1), x2 / (width - 1)], 1),
                                                                    rois[:, 0].int())
        if max_pool:
            crops = F.max_pool2d(crops, 2, 2)

        return crops

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.Conv_base.eval()
            self.Conv_base[5].train()
            self.Conv_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.Conv_base.apply(set_bn_eval)
            self.Conv_top.apply(set_bn_eval)

    def forward(self, im_data, im_info, gt_boxes):
        gt_boxes = gt_boxes.data  # 3D
        # feed image data to base model to obtain base feature map
        base_feat = self.Conv_base(im_data)  # [1, 1024, 37, 50]

        rois_label = Variable(gt_boxes[:, :, -1].contiguous().view(-1).long())

        rois = Variable(gt_boxes[:, :, 0:4])
        ss = gt_boxes.size()
        rr = torch.Tensor(ss[0], ss[1], 5).zero_().type_as(rois.data)
        rr[:, :, 1:5] = rois.data[:, :, 0:4]
        rr = Variable(rr)

        pooled_feat = self._crop_and_resize(base_feat, rr.contiguous().view(-1, 5), max_pool=False)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute object classification probability
        cls_score = self.Net_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        # classification loss
        loss = F.cross_entropy(cls_score, rois_label)
        cls_prob = cls_prob.view(rois.size(1), -1)  # *,classnum

        return cls_prob, loss

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.Net_cls_score, 0, 0.01, self.args.TRUNCATED)

    def _init_modules(self):
        resnet = models.resnet50(pretrained=True)

        self.Conv_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3)
        self.Conv_top = nn.Sequential(resnet.layer4)
        self.Net_cls_score = nn.Linear(2048, self.n_classes)

        # Fix blocks
        for p in self.Conv_base[0].parameters(): p.requires_grad = False
        for p in self.Conv_base[1].parameters(): p.requires_grad = False

        if 1 >= 1:
            for p in self.Conv_base[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.Conv_base.apply(set_bn_fix)
        self.Conv_top.apply(set_bn_fix)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _head_to_tail(self, pool5):
        fc7 = self.Conv_top(pool5).mean(3).mean(2)
        return fc7


class memory_res50(nn.Module):
    def __init__(self, classes, args, pretrained=False):
        super(memory_res50, self).__init__()
        self.model_path = args.backbone_path
        self.dout_base_model = 1024
        self.pretrained = pretrained

        self.classes = classes
        self.n_classes = len(classes)
        self.args = args

        self._predictions = {}
        self._predictions["cls_score"] = []
        self._predictions["cls_prob"] = []
        self._predictions["cls_pred"] = []
        self._predictions["confid"] = []
        self._predictions["weights"] = []

    def _crop_rois(self, bottom, rois):
        pre_pool_size = 7
        crops = CropAndResizeFunction(pre_pool_size, pre_pool_size)(bottom,
                                                                    Variable(rois),
                                                                    Variable(torch.zeros(rois.size(0), 1).cuda().int()))
        return crops

    def _inv_crops(self, pool5, inv_rois):
        inv_crops = CropAndResizeFunction(*(self._memory_size.data.cpu().int().numpy().tolist()))(pool5,
                                                                                                  Variable(inv_rois),
                                                                                                  Variable(torch.zeros(
                                                                                                      inv_rois.size(0),
                                                                                                      1).cuda().int()))
        # Add things up (make sure it is relu)
        inv_crop = torch.cumsum(inv_crops, dim=0)
        return inv_crop, inv_crops

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        def xavier_init(m):
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
                init.constant(m.bias.data, 0.)

        print('init weights...')
        normal_init(self._cls_init, 0, 0.01, self.args.TRUNCATED)
        self.context_n1.apply(xavier_init)
        self.context_n2.apply(xavier_init)
        self._fc_iter.apply(xavier_init)
        normal_init(self._cls_iter, 0, 0.01)
        normal_init(self._confidence_iter, 0, 0.01)
        normal_init(self._bottomtop_fc, 0, 0.01)
        normal_init(self._bottomtop_conv, 0, 0.01)
        self._input_conv.apply(xavier_init)
        normal_init(self._gate_p_input, 0, 0.01)
        normal_init(self._gate_p_reset, 0, 0.01)
        normal_init(self._gate_p_update, 0, 0.01)
        normal_init(self._gate_m_reset, 0, 0.01)
        normal_init(self._gate_m_update, 0, 0.01)
        normal_init(self._gate_m_input, 0, 0.01)


    def _init_modules(self):
        resnet = models.resnet50(pretrained=True)

        self.Conv_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                       resnet.layer2, resnet.layer3)
        self.Conv_top = nn.Sequential(resnet.layer4)
        self._cls_init = nn.Linear(2048, self.n_classes)

        # Fix blocks
        for p in self.Conv_base[0].parameters(): p.requires_grad = False
        for p in self.Conv_base[1].parameters(): p.requires_grad = False

        if 1 >= 1:
            for p in self.Conv_base[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.Conv_base.apply(set_bn_fix)
        self.Conv_top.apply(set_bn_fix)

        # define context conv related stuff
        self.context_n1 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.context_n2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding=1),
        )

        self._fc_iter = nn.Sequential(
            nn.Linear(512 * 49, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self._cls_iter = nn.Linear(4096, self.n_classes)
        self._confidence_iter = nn.Linear(4096, 1)
        self._bottomtop_fc = nn.Linear(self.n_classes, 512)
        self._bottomtop_conv = nn.Conv2d(1024, 512, (1, 1))
        self._input_conv = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU()
        )
        self._gate_p_input = nn.Conv2d(512, 512, (3, 3), padding=1)
        self._gate_p_reset = nn.Conv2d(512, 1, (3, 3), padding=1)
        self._gate_p_update = nn.Conv2d(512, 1, (3, 3), padding=1)
        self._gate_m_reset = nn.Conv2d(512, 1, (3, 3), padding=1)
        self._gate_m_update = nn.Conv2d(512, 1, (3, 3), padding=1)
        self._gate_m_input = nn.Conv2d(512, 512, (3, 3), padding=1)

    def _head_to_tail(self, pool5):
        fc7 = self.Conv_top(pool5).mean(3).mean(2)
        return fc7

    def _mem_init(self):
        mem_init = torch.zeros(1, self.args.MEM_C, self._memory_size.data.int()[0], self._memory_size.data.int()[1])
        mem_init = Variable(mem_init.cuda())
        return mem_init

    def _context(self, input):
        res1 = self.context_n1(input)
        res2 = self.context_n2(res1)
        return res1 + res2

    def _comb_conv_mem(self, cls_score_conv, cls_score_mem, iter):
        # take the output directly from each iteration
        if iter == 0:
            cls_score = cls_score_conv
        else:
            cls_score = cls_score_mem

        cls_prob = F.softmax(cls_score, dim=-1)

        self._predictions['cls_score'].append(cls_score)
        self._predictions['cls_prob'].append(cls_prob)

        return cls_score, cls_prob

    def _mem_pred(self, mem, cls_score_conv, rois, iter):
        '''
        Use memory to predict the output
        :return: 
        '''
        mem_net = self._context(mem)
        mem_ct_pool5 = self._crop_rois(mem_net, rois)  # n*512*7*7
        mem_fc7 = self._fc_iter(mem_ct_pool5.view(mem_ct_pool5.size(0), -1))
        cls_score_mem = self._cls_iter(mem_fc7)
        self._predictions['confid'].append(self._confidence_iter(mem_fc7))
        cls_score, cls_prob = self._comb_conv_mem(cls_score_conv, cls_score_mem, iter)
        return cls_score, cls_prob

    def _bottomtop(self, pool5, cls_prob):
        # just make the representation more dense
        map_prob = self._bottomtop_fc(cls_prob)
        map_comp = map_prob.view(-1, 512, 1, 1)
        pool5_comp = self._bottomtop_conv(pool5)

        pool5_comb = map_comp + pool5_comp
        pool5_comb = F.relu(pool5_comb)
        return pool5_comb

    def _input(self, net):
        # the first part is already done
        net = self._input_conv(net)
        return net

    def _input_module(self, pool5_nb, cls_score_nb):
        pool5_comb = self._bottomtop(pool5_nb, cls_score_nb)
        pool5_input = self._input(pool5_comb)
        return pool5_input

    def _mem_update(self, pool5_mem, pool5_input, iter):
        # compute the gates and features
        p_input = self._gate_p_input(pool5_input)
        p_reset = self._gate_p_reset(pool5_input)
        p_update = self._gate_p_update(pool5_input)
        # compute the gates and features from the hidden memory
        m_reset = self._gate_m_reset(pool5_mem)
        m_update = self._gate_m_update(pool5_mem)
        # get the reset gate, the portion that is kept from the previous step
        reset_gate = F.sigmoid(p_reset + m_reset)
        reset_res = pool5_mem * reset_gate
        m_input = self._gate_m_input(reset_res)
        # Non-linear activation
        pool5_new = F.relu(p_input + m_input)
        # get the update gate, the portion that is taken to update the new memory
        update_gate = F.sigmoid(p_update + m_update)
        # the update is done in a difference manner
        mem_diff = update_gate * (pool5_new - pool5_mem)
        return mem_diff

    def _mem_handle(self, mem, pool5_nb, cls_score, cls_prob, rois, inv_rois, iter):
        cls_score_nb = Variable(cls_score.data)
        pool5_mem = self._crop_rois(mem, rois)
        pool5_input = self._input_module(pool5_nb, cls_score_nb)
        mem_update = self._mem_update(pool5_mem, pool5_input, iter)
        mem_diff, _ = self._inv_crops(mem_update, inv_rois)
        # Update the memory
        mem_div = mem_diff / self._count_matrix_eps
        mem = mem + mem_div
        return mem

    def _update_weights(self, labels, cls_prob):
        num_gt = labels.size(0)
        index = torch.LongTensor([i for i in range(num_gt)]).cuda()
        cls_score = cls_prob[index, labels.data.type_as(index)]
        big_ones = cls_score >= 1. - self.args.MEM_BETA
        # Focus on the hard examples
        weights = 1. - cls_score
        weights[big_ones.type_as(index)] = self.args.MEM_BETA
        weights_sum = torch.sum(weights)
        weights /= torch.max(weights_sum, torch.ones_like(weights_sum).type_as(weights_sum) * (1e-14))

        weights = weights.contiguous().view(-1)
        self._predictions["weights"].append(weights)

    def _aggregate_pred(self):
        comb_confid = torch.stack(self._predictions['confid'], dim=2)
        comb_attend = F.softmax(comb_confid, dim=2)
        self._predictions['confid_prob'] = [comb_attend[:, :, i] for i in range(comb_attend.size(2))]
        comb_score = Variable(torch.stack(self._predictions["cls_score"], dim=2).data)
        # print('comb score', comb_score.size())
        # print('comb attend', comb_attend.size())
        # tt = comb_score*comb_attend
        # print('comb multiply', tt.size())

        cls_score = torch.sum(comb_score * comb_attend, dim=2)
        # print('cls score', cls_score.size())
        cls_prob = F.softmax(cls_score, dim = -1)

        self._predictions["attend_cls_score"] = cls_score
        self._predictions["attend_cls_prob"] = cls_prob
        self._predictions['attend'] = self._predictions['confid_prob']

        return cls_prob

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.Conv_base.eval()
            self.Conv_base[5].train()
            self.Conv_base[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.Conv_base.apply(set_bn_eval)
            self.Conv_top.apply(set_bn_eval)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def forward(self, im_data, im_info, gt_boxes, memory_size):
        self._predictions = {}
        self._predictions["cls_score"] = []
        self._predictions["cls_prob"] = []
        self._predictions["cls_pred"] = []
        self._predictions["confid"] = []
        self._predictions["weights"] = []

        gt_boxes = gt_boxes.data  # 3D
        self._count_base = Variable(torch.ones([1, 1, 7, 7]).cuda().float())
        self._memory_size = memory_size[0]
        # initialize memory
        mem = self._mem_init()  # [1, 512, h, w]

        # preparing baseconv related stuff
        base_feat = self.Conv_base(im_data)  # [1, 1024, 37, 50]
        rois, inv_rois = compute_target_memory(self._memory_size, gt_boxes[0, :, :], 16.)
        pool5 = self._crop_rois(base_feat, rois)
        pool5_nb = Variable(pool5.data)  # 1024 C

        # initialize the normalization vector, note here it is the batch ids
        count_matrix_raw, self._count_crops = self._inv_crops(self._count_base, inv_rois)
        self._count_matrix = Variable(count_matrix_raw.data)
        self._count_matrix_eps = torch.max(self._count_matrix,
                                           torch.ones_like(self._count_matrix).type_as(self._count_matrix) * (1e-14))

        fc7 = self._head_to_tail(pool5)
        # First iteration
        cls_score_conv = self._cls_init(fc7)
        # cls_prob_conv = F.softmax(cls_score_conv)

        # if self.training:
        if True:
            self._labels = Variable(gt_boxes[:, :, -1].contiguous().view(-1).long())
        else:
            self._labels = None

        for iter in range(self.args.MEM_ITER):
            # print('ITERATION: %02d' % iter)
            # Use memory to predict the output
            cls_score, cls_prob = self._mem_pred(mem, cls_score_conv, rois, iter)

            if iter == self.args.MEM_ITER - 1:
                break

            # Update the memory with all the regions
            mem = self._mem_handle(mem, pool5_nb, cls_score, cls_prob, rois, inv_rois, iter)

            # if training
            self._update_weights(self._labels, cls_prob)

        # Need to finalize the class scores, regardless of whether loss is computed
        cls_prob = self._aggregate_pred()

        # if training
        cross_entropy = []
        for iter in range(self.args.MEM_ITER):
            # RCNN, class loss
            cls_score = self._predictions["cls_score"][iter]
            ce = F.cross_entropy(cls_score, self._labels)
            ce = torch.mean(ce)
            cross_entropy.append(ce)

        ce_rest = torch.stack(cross_entropy[1:])
        cross_entropy_image = cross_entropy[0]
        cross_entropy_memory = torch.mean(ce_rest)
        cross_entropy = cross_entropy_image + 1 * cross_entropy_memory

        loss = cross_entropy

        cls_prob = cls_prob.view(rois.size(0), -1)  # *,classnum=2970?
        # print('cls prob', cls_prob.size())
        return cls_prob, loss
