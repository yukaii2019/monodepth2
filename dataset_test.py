import datasets
from datasets.tum_dataset import collect_image_paths
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

if __name__ == "__main__":
    seq = ["fr1_360", "fr2_xyz", "fr2_rpy", "fr2_large_with_loop",
           "fr2_large_no_loop", "fr2_desk", "fr2_360_kidnap", "fr2_360_hemisphere",
           "fr1_room", "fr1_rpy", "fr1_floor", "fr1_desk2", "fr1_desk", "fr1_xyz"]
    val_seq = ["fr1_desk", "fr1_desk2", "fr1_room", "fr2_desk", "fr2_xyz"]
    train_seq = list(set(seq) - set(val_seq))
    print(val_seq)
    print(train_seq)

    data_path = "/home/remote/ykhsieh/tum_monodepth2/monodepth2/tum_dataset"

    train_filenames, train_filenames_map = \
        collect_image_paths(data_path, train_seq)
    
    val_filenames, val_filenames_map = \
        collect_image_paths(data_path, val_seq)
    num_train_samples = len(train_filenames)
    print("num_train_samples: ", num_train_samples)
    print("num_val_samples: ", len(val_filenames))

    train_dataset = datasets.TumDataset(
        data_path, train_filenames, 480, 640,
        [-1, 0, 1], 4, is_train=True, img_ext=None, 
        filenames_map=train_filenames_map)
    
    train_loader = DataLoader(
        train_dataset, 4, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    
    train_iter = iter(train_loader)
    train_batch = next(train_iter)
    
    for k, v in train_batch.items():
        print(k)

    print(train_batch[('color', -1, 0)][0].data.shape)
    print(train_batch[('color', 0, 0)][0].data.shape)
    print(train_batch[('color', 1, 0)][0].data.shape)

    # Save the last three images to PNG format
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.ToPILImage()
    
    for i, idx in enumerate([-1, 0, 1]):
        img_tensor = train_batch[('color', idx, 0)][0]  # Take first image in batch
        img_tensor = (img_tensor * 255).byte()  # Convert from 0-1 to 0-255
        img = transform(img_tensor.cpu())  # Convert to PIL image
        img.save(os.path.join(save_dir, f"image_{i}.png"))

    exit()
