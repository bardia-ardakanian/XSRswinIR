from collections import OrderedDict

import re
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from models.scorevgg import get_score_module

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
import torchvision.transforms as transforms
import random


class ModelPlainXSR(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlainXSR, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']  # training option

        # TODO: add aux module (score module)
        self.score_module = get_score_module(self.opt_train['score_module_path'])

        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

        self.crop_size = opt['scale'] * opt['netG']['img_size']
        self.step = self.crop_size // 2
        self.thresh_size = 0

        self.iou_threshold = 0.5
        self.sim_count = 5
        self.max_sim_attempt = 10

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netG.train()  # set training mode,for BN
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.coefficient = self.opt_train['coefficient']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                                            self.opt_train['G_scheduler_periods'],
                                                                            self.opt_train[
                                                                                'G_scheduler_restart_weights'],
                                                                            self.opt_train['G_scheduler_eta_min']
                                                                            ))
        else:
            raise NotImplementedError

    def find_best_from_sub_images(self):
        bests = []
        for i, (path, whole_image) in enumerate(zip(self.L_path, self.whole_H)):
            h_space, w_space = self.get_all_sub_images_starting_point(whole_image.shape)

            sim_sub_images_to_L = self.get_similar_sub_images_to_L(path, whole_image, h_space, w_space, self.H[i:i+1, ...])

            # Flatten and concatenate
            flattened_images = np.column_stack(sim_sub_images_to_L)

            bests.append(self.find_best_replacement(flattened_images, self.H[i:i+1, ...].reshape(-1, 1)).reshape(3, 96, 96))

        return torch.stack(bests, dim=0)

    def get_similar_sub_images_to_L(self, path, whole_image, h_space, w_space, H):
        idx, x_start, y_start = self.get_sub_image_starting_point(path, h_space, w_space)

        # # Exclude the starting point of L
        # h_space = np.delete(h_space, idx)
        # w_space = np.delete(w_space, idx)

        similar_images = []
        similar_indices = []
        for x in h_space:
            for y in w_space:
                sub_image = whole_image[:, x:x + self.crop_size, y:y + self.crop_size]

                if self.check_similarity(sub_image, H) == 1:
                    # if not any(self.calculate_iou((x, y), (sim_x, sim_y)) > self.iou_threshold for sim_x, sim_y in similar_images):
                    similar_images.append(sub_image.flatten())
                    similar_indices.append((x, y))

                    if len(similar_images) == self.sim_count:
                        return similar_images

        # If not enough similar images found, generate random sub-images
        attempt_count = 0
        while len(similar_images) < self.sim_count and attempt_count < self.max_sim_attempt:
            # Randomly select a starting point inside the image boundaries
            x = np.random.randint(0, whole_image.shape[1] - self.crop_size)
            y = np.random.randint(0, whole_image.shape[2] - self.crop_size)

            sub_image = whole_image[:, x:x + self.crop_size, y:y + self.crop_size]

            if self.check_similarity(sub_image, H) == 1:
                if not any(self.calculate_iou((x, y), (sim_x, sim_y)) > self.iou_threshold for sim_x, sim_y in
                           similar_indices):
                    similar_images.append(sub_image.flatten())
                    similar_indices.append((x, y))

            attempt_count += 1

        # If after all attempts still not enough similar images found, select random sub-images from h_space and w_space
        while len(similar_images) < self.sim_count:
            x, y = h_space[np.random.choice(len(h_space))], w_space[np.random.choice(len(h_space))]

            sub_image = whole_image[:, x:x + self.crop_size, y:y + self.crop_size]

            if not any(self.calculate_iou((x, y), (sim_x, sim_y)) > self.iou_threshold for sim_x, sim_y in similar_indices):
                similar_images.append(sub_image.flatten())
                similar_indices.append((x, y))

        return similar_images

    def get_sub_image_starting_point(self, sub_image_path, h_space, w_space):
        """
        Given the path of the sub-image, determine its starting point in the whole_H image.
        """

        # Extract the number from the L_path
        idx = int(re.search(r'_s(\d+)', sub_image_path).group(1))

        x = h_space[(idx - 1) // len(w_space)]
        y = w_space[(idx - 1)   % len(w_space)]
        return idx, x, y

    def get_all_sub_images_starting_point(self, whole_H_shape):
        h, w = whole_H_shape[-2], whole_H_shape[-1]  # assuming the shape is (channels, height, width)

        # Construct h_space and w_space like in the provided code
        h_space = np.arange(0, h - self.crop_size + 1, self.step)
        if h - (h_space[-1] + self.crop_size) > self.thresh_size:
            h_space = np.append(h_space, h - self.crop_size)

        w_space = np.arange(0, w - self.crop_size + 1, self.step)
        if w - (w_space[-1] + self.crop_size) > self.thresh_size:
            w_space = np.append(w_space, w - self.crop_size)

        return h_space, w_space

    def calculate_iou(self, boxA, boxB):
        """
        boxA and boxB format: (x, y), where x, y are the starting coordinates
        """
        # determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + self.crop_size, boxB[0] + self.crop_size)
        yB = min(boxA[1] + self.crop_size, boxB[1] + self.crop_size)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both boxes
        boxAArea = self.crop_size * self.crop_size
        boxBArea = self.crop_size * self.crop_size

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of both areas minus the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def check_similarity(self, x, y):
        return int(self.score_module(torch.cat((x, y), dim=0)).item() >= 0.5)

    # TODO: find the best by linear combination
    def find_best_replacement(self, X, y):
        XTX = X.T @ X
        inv_XTX = torch.inverse(XTX)
        XTy = X.T @ y
        return X @ (inv_XTX @ XTy)

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.L_path = data['L_path']  # 48

        # add Whole LR
        self.whole_H = data['whole_H'].to(self.device)

        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netE
    # ----------------------------------------
    def netE_forward(self):
        self.E = self.netE(self.L)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    def find_estimated(self):
        bests = self.find_best_from_sub_images().to(self.device)
        return self.coefficient * self.G_lossfn(self.E, bests)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = self.G_lossfn(self.E, self.H) + self.find_estimated()
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        with torch.no_grad():
            self.netE_forward()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
