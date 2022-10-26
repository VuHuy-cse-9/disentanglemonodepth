# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn as nn
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
from networks.image_pool import ImagePool



class Trainer_da_no_grl:
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
        if self.opt.day_depth_guide:
            self.models["day_depth"] = networks.DepthDecoder(
                self.models["day_encoder"].num_ch_enc, 
                scales=[0])
            self.models["day_depth"].to(self.device)
        
        #Discriminator:
        num_ch_enc = np.flip(self.models["day_encoder"].num_ch_enc)
        if self.opt.discriminator_mode == "mlp":
            for i_layer in range(self.opt.num_discriminator):
                self.models["domain_classifier_{}".format(i_layer)] = \
                    MLPClassifier(num_ch_enc[i_layer], 1024)
                self.models["domain_classifier_{}".format(i_layer)].to(self.device)
                self.parameters_D_to_train += list(self.models["domain_classifier_{}".format(i_layer)].parameters())
        elif self.opt.discriminator_mode == "conv":
            for i_layer in range(self.opt.num_discriminator):
                self.models["domain_classifier_{}".format(i_layer)] = \
                    NLayerDiscriminator(
                        input_nc = num_ch_enc[i_layer],
                        n_layers=i_layer + 1,
                        norm_layer = nn.InstanceNorm2d
                        )
                self.models["domain_classifier_{}".format(i_layer)].to(self.device)
                self.parameters_D_to_train += list(self.models["domain_classifier_{}".format(i_layer)].parameters())
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
          
        if self.opt.day_depth_guide:
            self.L1loss = torch.nn.L1Loss().to(self.device)

        #Freeze day encoder-decoder
        self.models["day_encoder"].eval()
        if self.opt.day_depth_guide:
            self.models["day_depth"].eval()

        self.model_G_optimizer = optim.Adam(self.parameters_G_to_train, self.opt.G_learning_rate)
        self.model_G_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_G_optimizer, 
            milestones=[25,33], 
            gamma=0.5)
        
        self.model_D_optimizer = optim.Adam(self.parameters_D_to_train, self.opt.learning_rate)
        self.model_D_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_D_optimizer, 
            milestones=[25,33], 
            gamma=0.5)
        
        #Load pretrained day model
        self.load_day_model()
        
        self.epoch = 0
        self.step = 0
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "oxford": OxfordRobotDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        
        train_day_filenames = readlines(fpath.format("train_day"))
        train_night_filenames = readlines(fpath.format("train_night"))
        
        assert len(train_day_filenames) == len(train_night_filenames), "Day, Night files does not equal."

        val_day_filenames = readlines(fpath.format("val_day"))
        val_night_filenames = readlines(fpath.format("val_night"))
        
        assert len(val_day_filenames) == len(val_night_filenames), "Day, Night val files does not equal."
        
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_day_filenames)
        num_val_samples = len(val_day_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_day_dataset = self.dataset(
            self.opt.data_path,         # Folder container data: e.g kitti_data
            train_day_filenames,        # List of folders container data, e.g [//_sync]
            self.opt.height, 
            self.opt.width,             
            frame_idxs=[0],             # Use one image only
            num_scales=1,    
            is_train=True,              # Is train mode
            img_ext=img_ext,            # Image type: png / jpg
            is_guassian_noise=self.opt.gauss)            
        
        self.train_day_loader = DataLoader(
            train_day_dataset, 
            self.opt.batch_size, True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, 
            drop_last=True,
        )
        
        train_night_dataset = self.dataset(
            self.opt.data_path,             # Folder container data: e.g kitti_data
            train_night_filenames,          # List of folders container data, e.g [//_sync]
            self.opt.height, 
            self.opt.width,             
            frame_idxs=self.opt.frame_ids,  #[0, -1, 1]
            num_scales=1,    
            is_train=True,                 
            img_ext=img_ext,                # Image type: png / jpg
            is_mcie=self.opt.mcie,
            is_guassian_noise=self.opt.gauss)            
        
        self.train_night_loader = DataLoader(
            train_night_dataset, 
            batch_size=self.opt.batch_size, 
            shuffle=True,
            num_workers=self.opt.num_workers, 
            pin_memory=True, 
            drop_last=True,
            )
        
        val_day_dataset = self.dataset(
            self.opt.data_path,         # Folder container data: e.g kitti_data
            val_day_filenames,          # List of folders container data, e.g [//_sync]
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
            val_night_filenames,         # List of folders container data, e.g [//_sync]
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

        self.writers = {}
        write_modes = ["train_D", "train_G", "val_D", "val_G"]
        if self.opt.day_depth_guide:
            write_modes += ["train_depth", "val_depth"]
        for mode in write_modes:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items, validation iterms {:d}\n".format(
            num_train_samples, num_val_samples))

        self.fake_pool = {}
        for ilayer in range(self.opt.num_discriminator):
            self.fake_pool["layer_{}".format(ilayer)] = ImagePool(self.opt.pool_size)

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
        self.models["day_encoder"].eval()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.early_phase = False
        self.start_time = time.time()
        while self.epoch < self.opt.num_epochs:
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            self.epoch += 1
            old_lr, new_lr = self.update_learning_rate()
            print("learning rate %.7f -> %.7f" % (old_lr, new_lr))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        
        print("Training:")
        self.set_train()
        #self.is_train_D = False
        day_iterator = iter(self.train_day_loader)

        for batch_idx, night_inputs in enumerate(self.train_night_loader):
            try:
                day_inputs = next(day_iterator)
            except StopIteration:
                day_iterator = iter(self.train_day_loader)
                day_inputs = next(day_iterator)
                
            before_op_time = time.time()
            for key, ipt in day_inputs.items():
                day_inputs[key] = ipt.to(self.device)

            for key, ipt in night_inputs.items():
                night_inputs[key] = ipt.to(self.device)
            
            self.set_G_train()
            
            self.forward(day_inputs, night_inputs)
            
            if self.step % self.opt.num_d_per_g == 0:
                if self.opt.day_depth_guide:
                    self.model_G_optimizer.zero_grad()
                    train_G_loss, train_G_dict_loss = self.G_backward(retain_graph=True)
                    train_depth_loss = self.depth_backward()
                    self.model_G_optimizer.step()
                else:
                    self.model_G_optimizer.zero_grad()
                    train_G_loss, train_G_dict_loss = self.G_backward()
                    self.model_G_optimizer.step()
                          
            self.set_D_train()
            train_D_loss, train_D_dict_loss = self.D_backward()
            self.model_D_optimizer.step()
            
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            if not self.early_phase:
              self.early_phase = batch_idx % self.opt.log_frequency == 0

            if self.early_phase and (self.step % self.opt.num_d_per_g == 0):
                self.early_phase = False
                self.log("train_D", train_D_loss, train_D_dict_loss)
                self.log("train_G", train_G_loss, train_G_dict_loss)
                if self.opt.day_depth_guide:
                    self.log("train_depth", train_depth_loss, None)
                
                val_D_loss, val_G_loss, val_depth_loss = self.val()
                if self.opt.day_depth_guide:
                    self.log_time(batch_idx, 
                                duration, 
                                train_D_loss.cpu().data,
                                train_G_loss.cpu().data,
                                val_D_loss.cpu().data,
                                val_G_loss.cpu().data,
                                train_depth_loss.cpu().data,
                                val_depth_loss.cpu().data)
                else:
                    self.log_time(batch_idx, 
                                duration, 
                                train_D_loss.cpu().data,
                                train_G_loss.cpu().data,
                                val_D_loss.cpu().data,
                                val_G_loss.cpu().data,
                                None,
                                None)
                del val_D_loss, val_G_loss
                if self.opt.day_depth_guide:
                    del val_depth_loss
            self.step += 1
            
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.model_G_optimizer.param_groups[0]['lr']
        self.model_G_lr_scheduler.step()
        self.model_D_lr_scheduler.step()  
        lr = self.model_G_optimizer.param_groups[0]['lr']
        return old_lr, lr            
        
    def forward(self, day_inputs, night_inputs):
        self.night_features = self.models["night_encoder"](night_inputs["color_aug", 0, 0])
        self.day_features   = self.models["day_encoder"](day_inputs["color_aug", 0, 0])
        if self.opt.day_depth_guide:
            self.night_from_denc_features = self.models["day_encoder"](night_inputs["color_aug", 0, 0])

    def set_G_train(self):
        #self.models["day_encoder"].eval()
        self.models["night_encoder"].train()
        for i in range(self.opt.num_discriminator):
            self.models["domain_classifier_{}".format(i)].eval()
            
    def set_D_train(self):
        #self.models["day_encoder"].eval()
        self.models["night_encoder"].eval()
        for i in range(self.opt.num_discriminator):
            self.models["domain_classifier_{}".format(i)].train()
    
    def G_backward(self, is_train=True, retain_graph=None):
        G_loss = 0
        G_dict_loss = {}
        for i_layer in range(self.opt.num_discriminator): # 0 = End layer.
            night_feature = self.night_features[-(i_layer + 1)]
            channel = night_feature.shape[1]
            predict_night = None
            if self.opt.discriminator_mode == "mlp":
                # Global Average pooling:
                flatten_night_feature = torch.reshape(night_feature, (self.opt.batch_size, channel, -1))
                mean_night_feature = torch.mean(flatten_night_feature, dim=2, dtype=torch.float32)
                #Discriminator:
                predict_night = self.models["domain_classifier_{}".format(i_layer)](mean_night_feature)
            elif self.opt.discriminator_mode == "conv":
                predict_night = self.models["domain_classifier_{}".format(i_layer)](night_feature)

            loss = self.criterionGAN(predict_night, True)
            G_loss += loss * (self.opt.num_discriminator  - i_layer) # loss close to the end is more important.
            G_dict_loss["Generator_{}".format(i_layer)] = loss
        
        G_loss /= (2*self.opt.num_discriminator)
                
        if is_train:
            self.model_G_optimizer.zero_grad()
            G_loss.backward(retain_graph=retain_graph)
        return G_loss, G_dict_loss
        
    def depth_backward(self, is_train=True):
        night_depth = self.models["day_depth"](self.night_features)[("disp", 0)]
        night_from_denc_depth = self.models["day_depth"](self.night_from_denc_features)[("disp", 0)]
        
        loss = self.L1loss(night_depth, night_from_denc_depth)
        if is_train:
            loss.backward()
        return loss
    
    def D_backward(self, is_train=True):
        D_loss = 0
        D_dict_loss = {}
        for i_layer in range(self.opt.num_discriminator): # 0 = End layer.
            day_feature = self.day_features[-(i_layer + 1)].detach()
            night_feature = self.night_features[-(i_layer + 1)].detach()
            if is_train:
              night_feature = self.fake_pool["layer_{}".format(i_layer)].query(night_feature)
            
            channel = night_feature.shape[1]
            predict_night, predict_day = None, None
            if self.opt.discriminator_mode == "mlp":
                # Global Average pooling:
                mean_night_feature = torch.mean(torch.reshape(night_feature, 
                                                            (self.opt.batch_size, channel, -1)),
                                                dim=2, 
                                                dtype=torch.float32)
                mean_day_feature = torch.mean(torch.reshape(day_feature, 
                                                            (self.opt.batch_size, channel, -1)),
                                                dim=2, 
                                                dtype=torch.float32)
                #Discriminator:
                predict_night = self.models["domain_classifier_{}".format(i_layer)](mean_night_feature)
                predict_day = self.models["domain_classifier_{}".format(i_layer)](mean_day_feature)
            elif self.opt.discriminator_mode == "conv":
                #print("compare layer {}: {}".format(i_layer, (night_feature - self.G_prev["layer{}".format(i_layer)]).sum()))
                predict_night = self.models["domain_classifier_{}".format(i_layer)](night_feature)
                predict_day = self.models["domain_classifier_{}".format(i_layer)](day_feature)
                # if is_train:
                #   print("D predict day: {}".format(predict_day))
                #   print("D predict night: {}".format(predict_night))
            loss = (self.criterionGAN(predict_day, True) + self.criterionGAN(predict_night, False)) * 0.5
            #print("los")
            D_loss += loss
            D_dict_loss["discriminator_{}".format(i_layer)] = loss
        
        D_loss /= self.opt.num_discriminator
        
        if is_train:
            self.model_D_optimizer.zero_grad()
            D_loss.backward()
        return D_loss, D_dict_loss


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            day_inputs = self.val_day_iter.next()
            night_inputs = self.val_night_iter.next()
        except StopIteration:
            self.val_day_iter = iter(self.val_day_loader)
            self.val_night_iter = iter(self.val_night_loader)
            day_inputs = self.val_day_iter.next()
            night_inputs = self.val_night_iter.next()

        for key, ipt in day_inputs.items():
            day_inputs[key] = ipt.to(self.device)

        for key, ipt in night_inputs.items():
            night_inputs[key] = ipt.to(self.device)

        with torch.no_grad():
            self.forward(day_inputs, night_inputs)
            G_loss, G_dict_loss = self.G_backward(is_train=False)
            depth_loss = None
            if self.opt.day_depth_guide:
                depth_loss = self.depth_backward(is_train=False)
                self.log("val_depth", depth_loss, None)
            D_loss, D_dict_loss = self.D_backward(is_train=False)

            self.log("val_D", D_loss, D_dict_loss)
            self.log("val_G", G_loss, G_dict_loss)
            
            
            del day_inputs, night_inputs

        self.set_train()
        return D_loss, G_loss, depth_loss

    def log_time(self, batch_idx, duration, \
                train_D_loss, train_G_loss, \
                    val_D_loss, val_G_loss,
                    train_depth_loss=None, val_depth_loss=None):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        if train_depth_loss is None:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | train D loss: {:.5f} | train G loss: {:.5f} | val D loss:{:.5f} | val G loss:{:.5f}| time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, 
                                    train_D_loss, train_G_loss, 
                                    val_D_loss, val_G_loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | train D loss: {:.5f} | train G loss: {:.5f} | train Depth loss: {:.5f} | val D loss:{:.5f} | val G loss:{:.5f}| val Depth loss: {:.5f} | time elapsed: {} | time left: {}"
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, 
                                    train_D_loss, train_G_loss, train_depth_loss, 
                                    val_D_loss, val_G_loss, val_depth_loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, total_loss, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("All mean loss", total_loss, self.step)
        if losses is not None:
            for key, value in losses.items():
                writer.add_scalar("{}".format(key), value, self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            if model_name == "day_encoder" or model_name == "day_depth":
                continue
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'night_encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("D_adam"))
        torch.save(self.model_D_optimizer.state_dict(), save_path)
        
        save_path = os.path.join(save_folder, "{}.pth".format("G_adam"))
        torch.save(self.model_G_optimizer.state_dict(), save_path)
        
        save_path = os.path.join(save_folder, "{}.txt".format("step"))
        file = open(save_path, "w")
        file.writelines("epoch {}\n".format(self.epoch))
        file.writelines("step {}".format(self.step))
        file.close()

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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "D_adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading D Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_D_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find D Adam weights so Adam is randomly initialized")
            
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "G_adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading G Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_G_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find G Adam weights so Adam is randomly initialized")
            
        save_path = os.path.join(self.opt.load_weights_folder, "{}.txt".format("step"))
        file = open(save_path, "r")
        lines = file.readlines()
        file.close()
        
        self.epoch = int(lines[0].split(" ")[1])
        self.step = int(lines[1].split(" ")[1])
            
    def load_day_model(self):
        """Load model(s) from disk
        """
        print(">> Loading day model")
        self.opt.load_day_weights_folder = os.path.expanduser(self.opt.load_day_weights_folder)

        assert os.path.isdir(self.opt.load_day_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_day_weights_folder)
        print("loading model from folder {}".format(self.opt.load_day_weights_folder))
        
        model_to_loads = ["encoder"]
        if self.opt.day_depth_guide:
            model_to_loads += ["depth"]
        for n in model_to_loads:
            print("Loading day {} weights...".format(n))
            path = os.path.join(self.opt.load_day_weights_folder, "{}.pth".format(n))
            model_dict = self.models["day_{}".format(n)].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models["day_" + n].load_state_dict(model_dict)
