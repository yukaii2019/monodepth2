from __future__ import absolute_import, division, print_function
import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

class TumDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(TumDataset, self).__init__(*args, **kwargs)

        # self.K = np.array([[525.0, 0    , 319.5, 0],
        #                    [0    , 525.0, 239.5, 0],
        #                    [0    , 0    , 1    , 0],
        #                    [0    , 0    , 0    , 1]], dtype=np.float32)

        self.K = np.array([[525.0/640, 0    , 319.5/640, 0],
                           [0    , 525.0/480, 239.5/480, 0],
                           [0    , 0    , 1    , 0],
                           [0    , 0    , 0    , 1]], dtype=np.float32)
        
        self.full_res_shape = (640, 480)


    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(
            self.data_path,
            folder,
            "rgb",
            self.filenames_map[(folder, frame_index)]
        )
        return image_path

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color



def collect_image_paths(root_dir, sub_dirs, frame_interval=1, mode="train"):
    filenames = []
    filenames_map = {}

    for sub_dir in sub_dirs:
        full_path = os.path.join(root_dir, sub_dir)
        for subdir, _, files in os.walk(full_path):
            if "rgb.txt" in files:
                file_path = os.path.join(subdir, "rgb.txt")
                temp_filenames = []
                with open(file_path, 'r') as file:
                    i = 0
                    for line in file:
                        if line.strip() and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) == 2:
                                file_path = os.path.join(subdir, parts[1])
                                # dir_name = os.path.dirname(file_path)
                                base_name = os.path.basename(file_path)
                                seq = os.path.basename(os.path.normpath(subdir))
                                temp_filenames.append("{} {} {}".format(seq, i, "r"))
                                filenames_map[(seq, i)] = base_name
                                i+=1

                if mode == "test":
                    # Exclude the last image file 
                    filenames.extend(temp_filenames[:-frame_interval]) 
                else:
                    # Exclude first and last image file
                    if len(temp_filenames) > frame_interval*2:
                        filenames.extend(temp_filenames[frame_interval:-frame_interval])
    
    return filenames, filenames_map


if __name__ == "__main__":
    # Example usage
    root_directory = "/home/pylin/Research/dataset/"
    sub_directories = ["fr1_360", "fr2_xyz"]
    image_list, idx_map = collect_image_paths(root_directory, sub_directories)

    print(len(image_list))
    print(image_list[0])

    print(idx_map[("fr1_360", 1)])
