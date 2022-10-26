import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import Image


from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class OxfordRobotDataset(MonoDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OxfordRobotDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.768, 0, 0.5, 0],
                           [0, 1.024, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        # 983.044006, 0, 643.646973 / 1280
        # 0, 983.044006, 493.378998 / 960
        self.full_res_shape = (1280, 640)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        
    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        #TODO:Replace this with grount truth png
        gt_dir = os.path.join(
            self.data_path,
            scene_name + "_gt")
        print("path: {}".format(gt_dir))

        return os.path.exists(gt_dir)

    def get_color(self, folder, frame_index, side, do_flip):
        #Load RGB image:
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if self.is_train: #Because valdiation + test have already been crop
            color = color.crop((0, 160, 1280, 960-160))
            color = color.resize((512, 256),Image.ANTIALIAS)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        #TODO: Check if data_path is corrected!.
        depth_path = os.path.join(self.data_path, folder+'_gt', f_str)
        img_file = Image.open(depth_path)
        depth_png = np.array(img_file, dtype=int)
        img_file.close()
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255, \
            "np.max(depth_png)={}, path={}".format(np.max(depth_png), depth_path)
        # print(np.min(depth_png), ' ',np.max(depth_png))

        depth_gt = depth_png.astype(np.float) / 256.
        # depth[depth_png == 0] = -1.
        # depth = np.expand_dims(depth, -1)

        # depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])

        depth_gt = depth_gt[160:960-160,:]

        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt