# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
from datasets.oxford_dataset import OxfordRobotDataset
import networks
from IPython import embed
from tqdm import tqdm

from networks.MLPclassifier import MLPClassifier
from networks.ConvClassifier import NLayerDiscriminator

from networks.networks import GANLoss
from options import MonodepthOptions



class Evaluate_adfa:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        self.models = {}
        self.parameters_G_to_train = []
        self.parameters_D_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        #Night encoder:
        activation = "relu"
        if self.opt.leakyrelu:
            activation = "leakyrelu"
        self.models["night_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", activation=activation)
        self.models["night_encoder"].to(self.device)
        self.parameters_G_to_train += list(self.models["night_encoder"].parameters())
        
        #Day encoder:
        self.models["day_encoder"] = networks.ResnetEncoder(
            18, self.opt.weights_init == "pretrained")
        self.models["day_encoder"].to(self.device)
        
        #Discriminator:
        num_ch_enc = np.flip(self.models["day_encoder"].num_ch_enc)
        if self.opt.discriminator_mode == "mlp":
            for i_layer in range(self.opt.num_discriminator):
                self.models["domain_classifier_{}".format(i_layer)] = \
                    MLPClassifier(num_ch_enc[i_layer], 1024)
                self.models["domain_classifier_{}".format(i_layer)].to(self.device)
        elif self.opt.discriminator_mode == "conv":
            for i_layer in range(self.opt.num_discriminator):
                self.models["domain_classifier_{}".format(i_layer)] = \
                    NLayerDiscriminator(
                        input_nc = num_ch_enc[i_layer],
                        n_layers=i_layer + 1)
                self.models["domain_classifier_{}".format(i_layer)].to(self.device)
        else:
            raise NotImplementedError()
                           
        #Discriminator loss:
        self.domain_loss = torch.nn.BCELoss(reduction="mean")
        if self.opt.smooth_domain_label:
          self.criterionGAN = GANLoss(self.opt.gan_mode,
                                              target_real_label=0.9,
                                              target_fake_label=0.1).to(self.device)
        else:
          self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)

    
        
        #Load pretrained day model
        self.load_day_model()
        
        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.set_eval()
        
        print("Training model named:\n  ", self.opt.model_name)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "oxford": OxfordRobotDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        test_day_filenames = readlines(fpath.format("day_test"))
        test_night_filenames = readlines(fpath.format("night_test"))
                
        img_ext = '.png' if self.opt.png else '.jpg'

        num_day_val_samples = len(test_day_filenames)
        num_night_val_samples = len(test_night_filenames)
        
        val_day_dataset = self.dataset(
            self.opt.data_path,         # Folder container data: e.g kitti_data
            test_day_filenames,          # List of folders container data, e.g [//_sync]
            self.opt.height, 
            self.opt.width,             
            frame_idxs=[0],             # Use one image only
            num_scales=1,    
            is_train=False,              # Is train mode
            img_ext=img_ext,             # Image type: png / jpg
            )  
        
        self.val_day_loader = DataLoader(
            val_day_dataset, 
            batch_size=self.opt.batch_size, 
            shuffle=True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, 
            drop_last=True)
        self.val_day_iter = iter(self.val_day_loader)
        
        val_night_dataset = self.dataset(
            self.opt.data_path,          # Folder container data: e.g kitti_data
            test_night_filenames,         # List of folders container data, e.g [//_sync]
            self.opt.height, 
            self.opt.width,             
            frame_idxs=[0],              # Use one image only
            num_scales=1,    
            is_train=False,              # Is train mode
            img_ext=img_ext,             # Image type: png / jpg
            )  
        
        self.val_night_loader = DataLoader(
            val_night_dataset, 
            batch_size=self.opt.batch_size, 
            shuffle=True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, 
            drop_last=True)
        self.val_night_iter = iter(self.val_night_loader)

        print("Using split:\n  ", self.opt.split)
        print("There are day test {}, night test items {:d}\n".format(
            num_day_val_samples,
            num_night_val_samples))

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
                

    def val(self):
        """Validate the model on a single minibatch
        """
        print("Evaluating")
        
        self.set_eval()
        #self.val_day_iter = iter(self.val_day_loader)
        print(">> Night evaluating:")
        for batch_idx, night_inputs in enumerate(self.val_night_loader):          
            for key, ipt in night_inputs.items():
                night_inputs[key] = ipt.to(self.device)
            D_loss = 0
            with torch.no_grad():
                night_features = self.models["night_encoder"](night_inputs["color_aug", 0, 0])
                #print("night feature: {}".format(night_features[-1]))
                predict_night_dict = {}
                final_predict = 0
                for i_layer in range(self.opt.num_discriminator): # 0 = End layer.
                    night_feature = night_features[-(i_layer + 1)]
                    channel = night_feature.shape[1]
                    if self.opt.discriminator_mode == "mlp":
                        # Global Average pooling:
                        flatten_night_feature = torch.reshape(night_feature, (self.opt.batch_size, channel, -1))
                        mean_night_feature = torch.mean(flatten_night_feature, dim=2, dtype=torch.float32)
                        #Discriminator:
                        predict_night = self.models["domain_classifier_{}".format(i_layer)](mean_night_feature)
                    elif self.opt.discriminator_mode == "conv":
                        predict_night = self.models["domain_classifier_{}".format(i_layer)](night_feature)
                    loss = self.criterionGAN(predict_night, False)
                    night_class = torch.round(predict_night)
                    predict_night_dict["layer_{}".format(i_layer)] = torch.mean(predict_night)
                    final_predict += torch.mean(predict_night)
                    D_loss += loss
                #print(">> Sample {}: dis_0 = {}, dis_1={}. dis_2={}, final={}, loss = {},".format(
                print(">> Sample {}: dis_0 = {}, dis_1={}, dis_2={}, final={}, loss = {},".format(
                            batch_idx,
                            predict_night_dict["layer_0"],
                            predict_night_dict["layer_1"],
                            predict_night_dict["layer_2"],
                            final_predict / self.opt.num_discriminator,
                            D_loss / self.opt.num_discriminator
                            ))
            del night_inputs

        print(">> Day evaluating:")
        
        for batch_idx, day_inputs in enumerate(self.val_day_loader):          
            for key, ipt in day_inputs.items():
                day_inputs[key] = ipt.to(self.device)
            D_loss = 0
            with torch.no_grad():
                day_features = self.models["day_encoder"](day_inputs["color_aug", 0, 0])
                #print("day feature: {}".format(night_features[-1]))
                predict_day_dict = {}
                final_predict = 0
                for i_layer in range(self.opt.num_discriminator): # 0 = End layer.
                    day_feature = day_features[-(i_layer + 1)]
                    channel = day_feature.shape[1]
                    if self.opt.discriminator_mode == "mlp":
                        # Global Average pooling:
                        flatten_day_feature = torch.reshape(day_feature, (self.opt.batch_size, channel, -1))
                        mean_day_feature = torch.mean(flatten_day_feature, dim=2, dtype=torch.float32)
                        #Discriminator:
                        predict_day = self.models["domain_classifier_{}".format(i_layer)](mean_day_feature)
                    elif self.opt.discriminator_mode == "conv":
                        predict_day = self.models["domain_classifier_{}".format(i_layer)](day_feature)
                    #print(">> dis {}: {}".format(i_layer, torch.mean(predict_night)))
                    loss = self.criterionGAN(predict_day, True)
                    day_class = torch.round(predict_day)
                    predict_day_dict["layer_{}".format(i_layer)] = torch.mean(predict_day)
                    final_predict += torch.mean(predict_day)
                    D_loss += loss
                #print(">> Sample {}: dis_0 = {}, dis_1 = {}, dis_2={}, final={}, loss = {}".format(
                print(">> Sample {}: dis_0 = {}, dis_1={}, dis_2={}, final={} loss = {}".format(
                            batch_idx,
                            predict_day_dict["layer_0"],
                            predict_day_dict["layer_1"],
                            predict_night_dict["layer_2"],
                            final_predict / self.opt.num_discriminator,
                            D_loss / self.opt.num_discriminator
                            ))
            del day_inputs
        return D_loss

    def load_model(self):
        print(">> Loading model")
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def load_day_model(self):
        """Load model(s) from disk
        """
        print(">> Loading day model")
        self.opt.load_day_weights_folder = os.path.expanduser(self.opt.load_day_weights_folder)

        assert os.path.isdir(self.opt.load_day_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_day_weights_folder)
        print("loading model from folder {}".format(self.opt.load_day_weights_folder))

        for n in ["encoder"]:
            print("Loading day {} weights...".format(n))
            path = os.path.join(self.opt.load_day_weights_folder, "{}.pth".format(n))
            model_dict = self.models["day_{}".format(n)].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models["day_" + n].load_state_dict(model_dict)


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    evaluater = Evaluate_adfa(opts)
    evaluater.val()
    