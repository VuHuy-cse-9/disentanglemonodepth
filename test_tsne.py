# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
from tsne_torch import TorchTSNE as TSNE


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')
    parser.add_argument('--log_dir', type=str,
                       help='path to pretrain model')
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """

    device = torch.device("cuda")

    model_path = args.log_dir
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # print("   Loading pretrained decoder")
    # depth_decoder = networks.DepthDecoder(
    #     num_ch_enc=encoder.num_ch_enc, scales=range(4))

    # loaded_dict = torch.load(depth_decoder_path, map_location=device)
    # depth_decoder.load_state_dict(loaded_dict)

    # depth_decoder.to(device)
    # depth_decoder.eval()
    
    day_file = open("day_random.txt", "r")
    day_lines = day_file.readlines()
    night_file = open("night_random.txt", "r")
    night_lines = night_file.readlines()
    
    print("Number of day samples {}".format(len(day_lines)))
    print("Number of night samples {}".format(len(night_lines)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        day_features = []
        count_day = 0
        for idx, line in enumerate(day_lines):
            array = line.split( )
            folder_path, image_name = array[0], array[1]
            image_path = "Oxfordrobocar/" + folder_path + "/" + image_name + ".png"
            
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            feature = encoder(input_image)[-1]
            feature = feature.reshape(512, -1)
            feature = torch.mean(feature, dim=1)
            day_features.append(feature)
            # count_day += 1
            # if count_day == 10:
            #   break

        night_features = []
        count_night = 0
        for idx, line in enumerate(night_lines):
            array = line.split( )
            folder_path, image_name = array[0], array[1]
            image_path = "Oxfordrobocar/" + folder_path + "/" + image_name + ".png"
            
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            feature = encoder(input_image)[-1]
            feature = feature.reshape(512, -1)
            feature = torch.mean(feature, dim=1)
            night_features.append(feature)


            # count_night += 1
            # if count_night == 10:
            #   break

        night_features = torch.vstack(night_features)
        day_features = torch.vstack(day_features)
        features = torch.vstack([day_features, night_features])
        print("features: {}".format(features.shape))
        emb = TSNE(n_components=2, perplexity=30, initial_dims=512, n_iter=1000, verbose=True).fit_transform(features)
        day_emb = emb[:100]
        night_emb = emb[100:]

        # Get day embedding:
        # print(">> Generate day ebmeddding:")
        # print("day features: {}".format(day_features.shape))

        # day_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(day_features)  # returns shape (n_samples, 2)

        # print(">> Generate night embedding")
        # print("night features: {}".format(night_features.shape))
        # night_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(night_features)  # returns shape (n_samples, 2)
        
        plt.scatter(day_emb[:, 0],day_emb[:, 1], c='b', marker='o', label='day')
        plt.scatter(night_emb[:, 0], night_emb[:, 1], c='r', marker='o', label='night')
        plt.legend(loc='upper left')
        plt.savefig("result.png")


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
