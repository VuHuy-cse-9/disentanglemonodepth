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
from networks.gradient_reversal.module import GradientReversal


class Trainer_da:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        
        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        #Night encoder:
        self.models["night_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["night_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["night_encoder"].parameters())
        
        #Day encoder:
        self.models["day_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["day_encoder"].to(self.device)
        #self.parameters_to_train += list(self.models["day_encoder"].parameters())

        #Day depth, for validate only:
        # self.models["day_depth"] = networks.DepthDecoder(
        #     self.models["day_encoder"].num_ch_enc, self.opt.scales)
        # self.models["day_depth"].to(self.device)
        #self.parameters_to_train += list(self.models["depth"].parameters())
        
        #Gradient Reverse layer:
        self.models["gradient_reverse"] = GradientReversal(1.0)
        self.models["gradient_reverse"].to(self.device)
        self.parameters_to_train += list(self.models["gradient_reverse"].parameters())
        
        #Discriminator:
        num_ch_enc = np.flip(self.models["day_encoder"].num_ch_enc)
        if self.opt.discriminator_mode == "mlp":
            for i_layer in range(self.opt.num_discriminator):
                self.models["domain_classifier_{}".format(i_layer)] = \
                    MLPClassifier(num_ch_enc[i_layer], 1024)
                self.models["domain_classifier_{}".format(i_layer)].to(self.device)
                self.parameters_to_train += list(self.models["domain_classifier_{}".format(i_layer)].parameters())
        elif self.opt.discriminator_mode == "conv":
            for i_layer in range(self.opt.num_discriminator):
                self.models["domain_classifier_{}".format(i_layer)] = \
                    NLayerDiscriminator(num_ch_enc[i_layer])
                self.models["domain_classifier_{}".format(i_layer)].to(self.device)
                self.parameters_to_train += list(self.models["domain_classifier_{}".format(i_layer)].parameters())
        else:
            raise NotImplementedError()
                           
        #Discriminator loss:
        self.domain_loss = torch.nn.BCELoss(reduction="mean")
        
        #Freeze day encoder-decoder
        self.models["day_encoder"].eval()
        #self.models["day_depth"].eval()

        self.model_optimizer = optim.Adam(self.parameters_to_train, 0.0001)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, 
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
            is_train=False,                 # I don't do augmentation for night dataset
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
        for mode in ["train_D", "train_G", "val_D", "val_G"]:
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

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.start_time = time.time()
        while self.epoch < self.opt.num_epochs:
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            self.epoch += 1

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        day_iterator = iter(self.train_day_loader)
        for batch_idx, night_inputs in enumerate(self.train_night_loader):
            try:
                day_inputs = next(day_iterator)
            except StopIteration:
                day_iterator = iter(self.train_day_loader)
                day_inputs = next(day_iterator)
                
            before_op_time = time.time()
            
            train_total_D_loss, train_total_G_loss, \
                train_D_losses, train_G_losses = \
                    self.process_batch(day_inputs, night_inputs)
            
            self.model_optimizer.zero_grad()
            train_total_D_loss.backward()
            self.model_optimizer.step()
            
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0

            if early_phase:
                self.log("train_D", train_total_D_loss, train_D_losses)
                self.log("train_G", train_total_G_loss, train_G_losses)
                val_total_D_loss, val_total_G_loss = self.val()
                self.log_time(batch_idx, 
                            duration, 
                            train_total_D_loss.cpu().data,
                            train_total_G_loss.cpu().data,
                            val_total_D_loss.cpu().data,
                            val_total_G_loss.cpu().data)
                del val_total_D_loss, val_total_G_loss

            self.step += 1

    def process_batch(self, day_inputs, night_inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in day_inputs.items():
            day_inputs[key] = ipt.to(self.device)

        for key, ipt in night_inputs.items():
            night_inputs[key] = ipt.to(self.device)
        
        night_features = self.models["night_encoder"](night_inputs["color_aug", 0, 0])
        day_features   = self.models["day_encoder"](day_inputs["color_aug", 0, 0])
        
        losses, G_losses = {}, {}
        total_loss = 0 # Mean of all discriminator loss
        total_G_loss = 0 # Mean of all generator loss, as I use GRL, I only compute this for visualize
        for i_layer in range(self.opt.num_discriminator): # 0 = End layer.
            night_feature = night_features[-(i_layer + 1)]
            day_feature = day_features[-(i_layer + 1)]
        
            night_feature = self.models["gradient_reverse"](night_feature)
            channel = night_feature.shape[1]
        
            if self.opt.discriminator_mode == "mlp":
                # Global Average pooling:
                flatten_night_feature = torch.reshape(night_feature, (self.opt.batch_size, channel, -1))
                mean_night_feature = torch.mean(flatten_night_feature, dim=2, dtype=torch.float32)
                flatten_day_feature = torch.reshape(day_feature, (self.opt.batch_size, channel, -1))
                mean_day_feature = torch.mean(flatten_day_feature, dim=2, dtype=torch.float32)
            
                #Discriminator:
                predict_day = self.models["domain_classifier".format(i_layer)](mean_day_feature) 
                predict_night = self.models["domain_classifier_{}".format(i_layer)](mean_night_feature)
            elif self.opt.discriminator_mode == "conv":
                predict_day = self.models["domain_classifier_{}".format(i_layer)](day_feature)
                predict_night = self.models["domain_classifier_{}".format(i_layer)](night_feature)

            #Day = 1, night = 0
            loss, G_loss = None, None
            if self.opt.smooth_domain_label:
                loss = -torch.mean(torch.log(1 - torch.abs(predict_day - 0.9))) - torch.mean(torch.log(1 - torch.abs(predict_night - 0.1)))
                G_loss = -torch.mean(torch.log(1 - torch.abs(predict_night - 0.9)))
            else:   
                loss = -torch.mean(torch.log(predict_day)) - torch.mean(torch.log(1 - predict_night))
                G_loss = -torch.mean(torch.log(predict_night))
            total_loss += loss
            total_G_loss += G_loss
            losses["discriminator_{}".format(i_layer)] = loss
            G_losses["Generator_{}".format(i_layer)] = G_loss
        return total_loss / self.opt.num_discriminator, \
                total_G_loss / self.opt.num_discriminator, \
                 losses, G_losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]: #For prev and next frame.
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    #axisangle: (4, 2, 1, 3)
                    #translation: (4, 2, 1, 3)
                    axisangle, translation = self.models["pose"](pose_inputs)
                    
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    #Why only receive the first translation, axisangle???
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

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

        val_total_D_loss = 0
        val_total_G_loss = 0
        with torch.no_grad():
            val_total_D_loss, val_total_G_loss, \
                val_D_losses, val_G_losses = \
                    self.process_batch(day_inputs, night_inputs)

            self.log("val_D", val_total_D_loss, val_D_losses)
            self.log("val_G", val_total_G_loss, val_G_losses)
            del day_inputs, night_inputs

        self.set_train()
        return val_total_D_loss, val_total_G_loss

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate( #Upsample
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                        
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    #Take the matrix transformation mapping from current frame
                    #to frame_id frame
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)# Mean over all image
                    #Simarly calculation in predict pose:
                    #TODO: Reading how it work
                    T = transformation_from_parameters(
                            axisangle[:, 0], 
                            translation[:, 0] * mean_inv_depth[:, 0], 
                            frame_id < 0)

                #Project image in current frame into 3D coordinate (using depth)
                #TODO: Read how it work
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                #Project 3D coordinate into prev/next frame image plane.
                #Projection function in equation (2)
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                #Transform prev/next frame into current frame.
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        #Step 1: L1 Loss is first calculate
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True) #Mean of Image's channels

        #Step 2: Calculate on SSIM
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            #ssim process on pixel
            ssim_loss = self.ssim(pred, target).mean(1, True) #Mean of Image's channels
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
            

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            # Calculate reprojection loss between current image
            # and the mapping into current image of prev/next frame.
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            #shape (4, 2, width, height)
            reprojection_losses = torch.cat(reprojection_losses, 1)


            #AUTO_MASKING:
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses
            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.concat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            #Combined: (4, 4, 192, 640)                #idxs: index of min reprojection loss frame
            
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                #0, 1: Mapping frame
                #2, 3: Origin Next and prev frame
                #identity_reprojection_loss.shape[1] - 1 = 1
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
                

            #Only minimum loss of 4 preprojection loss are consider (If min is apply)
            # Or 
            loss += to_optimise.mean()

            #Mean-normalized depth: 
            mean_disp = disp.mean(2, True).mean(3, True)    
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        
        if self.opt.adversarial:
            losses["domain_loss"] = self.domain_loss(outputs["domain_pred"], inputs["domain_gt"])
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        Comment: Only involve in validation phase
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, \
                train_D_loss, train_G_loss, \
                    val_D_loss, val_G_loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | train D loss: {:.5f} | train G loss: {:.5f} | val D loss:{:.5f} | val G loss:{:.5f}| time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, train_D_loss, train_G_loss, val_D_loss, val_G_loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, total_loss, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar("All mean loss", total_loss, self.step)
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

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        
        save_path = os.path.join(save_folder, "{}.txt".format("step"))
        file = open(save_path, "w")
        file.writelines("epoch {}\n".format(self.epoch))
        file.writelines("step {}".format(self.step))
        file.close()

    def load_model(self):
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
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
            
        save_path = os.path.join(self.opt.load_weights_folder, "{}.txt".format("step"))
        file = open(save_path, "r")
        lines = file.readlines()
        file.close()
        
        self.epoch = int(lines[0].split(" ")[1]) + 1
        self.step = int(lines[1].split(" ")[1]) + 1
            
    def load_day_model(self):
        """Load model(s) from disk
        """
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
