import os
from glob import glob
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import defaultdict
from torchvision import transforms

# Define degradation type to index mapping dictionary
DEGRADATION_TYPE_MAP = {
    "hazy": 0,
    "rain": 1,
    "snow": 2,
}


# Data augmentation operations
class RandomRotate(object):
    """Randomly rotate image with angles 0°, 90°, 180°, or 270°"""

    def __call__(self, data):
        dirct = random.randint(0, 3)  # Randomly select rotation count (0-3)
        for key in data.keys():
            data[key] = np.rot90(data[key], dirct).copy()
        return data


class RandomFlip(object):
    """Randomly flip image horizontally and vertically"""

    def __call__(self, data):
        # 50% probability for horizontal flip
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.fliplr(data[key]).copy()

        # 50% probability for vertical flip
        if random.randint(0, 1) == 1:
            for key in data.keys():
                data[key] = np.flipud(data[key]).copy()
        return data


class RandomCrop(object):
    """Randomly crop image to specified size"""

    def __init__(self, Hsize, Wsize):
        self.Hsize = Hsize
        self.Wsize = Wsize

    def __call__(self, data):
        # Get dimensions of the first image (all images should have same dimensions)
        H, W, C = np.shape(list(data.values())[0])
        h, w = self.Hsize, self.Wsize

        # Ensure image is large enough for cropping
        assert H >= h and W >= w, f"Image size ({H},{W}) is smaller than crop size ({h},{w})"

        # Randomly select crop starting point
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)

        # Apply same crop position to all images
        for key in data.keys():
            data[key] = data[key][top:top + h, left:left + w].copy()

        return data


class ToTensor(object):
    """Convert images to PyTorch tensors and adjust channel order"""

    def __call__(self, data):
        for key in data.keys():
            # Convert numpy image from HWC to CHW format and create copy to avoid memory sharing issues
            data[key] = torch.from_numpy(data[key].transpose((2, 0, 1))).float().clone() / 255.0
        return data


class AllWeatherDataset(Dataset):
    def __init__(self, root_dir, crop_size=(256, 256), transform=None, input_list=None, gt_list=None,
                 degradation_types='snow'):
        """
        Initialize dataset class.
        :param root_dir: Root directory path of the dataset, containing 'input' and 'gt' folders.
        :param crop_size: Target size for random cropping.
        :param transform: Optional image augmentation transforms, uses default augmentations if None.
        :param input_list: Optional, list of input image paths (if specified, will use these file paths)
        :param gt_list: Optional, list of GT image paths (if specified, will use these file paths)
        :param degradation_types: Optional, specify which degradation types to load.
                                  Can be:
                                  - None: load all data (default behavior)
                                  - str: single type like 'rain', 'hazy', or 'snow'
                                  - list: multiple types like ['rain', 'snow']
        """
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.degradation_types = degradation_types

        # Convert single string to list for unified processing
        if isinstance(degradation_types, str):
            self.degradation_types = [degradation_types]

        # Validate degradation types if specified
        if self.degradation_types is not None:
            for dtype in self.degradation_types:
                if dtype not in DEGRADATION_TYPE_MAP:
                    raise ValueError(
                        f"Invalid degradation type: {dtype}. Valid types are: {list(DEGRADATION_TYPE_MAP.keys())}")

        # If file lists are provided, use them to load data; otherwise load entire dataset by default
        if input_list is not None and gt_list is not None:
            assert len(input_list) == len(gt_list), "Input and GT image list counts do not match"
            self.input_paths = input_list
            self.gt_paths = gt_list

            # Filter by degradation types if specified
            if self.degradation_types is not None:
                self._filter_by_degradation_types()
        else:
            # Load all images first
            all_input_paths = sorted(glob(os.path.join(root_dir, "input", '*.*')))
            all_gt_paths = sorted(glob(os.path.join(root_dir, "gt", '*.*')))

            # Filter by degradation types if specified
            if self.degradation_types is not None:
                self.input_paths = []
                self.gt_paths = []

                for input_path, gt_path in zip(all_input_paths, all_gt_paths):
                    # Extract degradation type from filename
                    degradation_type = os.path.basename(input_path).split('_')[0]

                    # Check if this type should be included
                    if degradation_type in self.degradation_types:
                        self.input_paths.append(input_path)
                        self.gt_paths.append(gt_path)

                print(f"Loaded {len(self.input_paths)} image pairs with degradation types: {self.degradation_types}")
            else:
                # Load all data if no specific types specified
                self.input_paths = all_input_paths
                self.gt_paths = all_gt_paths
                print(f"Loaded all {len(self.input_paths)} image pairs")

        # Validate counts
        assert len(self.input_paths) == len(self.gt_paths), "Number of input and GT images do not match"

        # Check if any data was loaded
        if len(self.input_paths) == 0:
            raise ValueError(f"No images found with degradation types: {self.degradation_types}")

        # If no transform is provided, use default augmentation sequence
        if transform is None:
            # Create default data augmentation sequence
            # Crop -> Flip -> Rotate -> ToTensor
            self.transform = transforms.Compose([
                RandomCrop(crop_size[0], crop_size[1]),
                RandomFlip(),
                RandomRotate(),
                ToTensor()
            ])
        else:
            self.transform = transform

    def _filter_by_degradation_types(self):
        """Filter already loaded paths by degradation types"""
        if self.degradation_types is None:
            return

        filtered_input_paths = []
        filtered_gt_paths = []

        for input_path, gt_path in zip(self.input_paths, self.gt_paths):
            # Extract degradation type from filename
            degradation_type = os.path.basename(input_path).split('_')[0]

            # Check if this type should be included
            if degradation_type in self.degradation_types:
                filtered_input_paths.append(input_path)
                filtered_gt_paths.append(gt_path)

        self.input_paths = filtered_input_paths
        self.gt_paths = filtered_gt_paths

        print(f"Filtered to {len(self.input_paths)} image pairs with degradation types: {self.degradation_types}")

    def __len__(self):
        """
        Return dataset size, i.e., number of samples.
        """
        return len(self.input_paths)

    def __getitem__(self, index):
        # Get input image and GT image paths
        input_path = self.input_paths[index]
        gt_path = self.gt_paths[index]

        # Use OpenCV to read images, avoiding PIL and numpy conversions
        input_image = cv2.imread(input_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        gt_image = cv2.imread(gt_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Ensure both images have same dimensions
        assert input_image.shape == gt_image.shape, "Input and GT image dimensions are inconsistent"

        # Create data dictionary containing input and GT images
        data = {
            'input': input_image,
            'gt': gt_image
        }

        # Apply data augmentation transform sequence
        if self.transform:
            data = self.transform(data)

        # Extract degradation type from filename
        degradation_type = os.path.basename(input_path).split('_')[0]
        degradation_type_index = DEGRADATION_TYPE_MAP.get(degradation_type, -1)  # 使用get避免键不存在错误

        # Return processed data
        return {
            'input_image': data['input'],
            'gt_image': data['gt'],
            'degradation_type': degradation_type_index,
            'input_path': input_path,  # Add path information for debugging convenience
            'gt_path': gt_path
        }


def create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=2, rank=0, world_size=1):
    # Use DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# Usage example
if __name__ == "__main__":
    # Define dataset path
    dataset_root = 'D:/pycharm_workspace/Data_Processor/New_Dataset/train'
    print(f"Testing dataset path: {dataset_root}")

    # Set output directory
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test images will be saved to: {output_dir} folder")

    print("\n" + "=" * 60)
    print("Testing Selective Loading Feature")
    print("=" * 60)

    # Test 1: Load only rain data
    print("\n1. Loading only 'rain' images:")
    rain_dataset = AllWeatherDataset(
        root_dir=dataset_root,
        crop_size=(256, 256),
        degradation_types='rain'
    )
    print(f"   Rain dataset size: {len(rain_dataset)}")
    if len(rain_dataset) > 0:
        sample = rain_dataset[0]
        print(f"   First sample degradation type: {sample['degradation_type']} (should be 1 for rain)")

    # Test 2: Load multiple degradation types
    print("\n2. Loading 'rain' and 'snow' images:")
    multi_dataset = AllWeatherDataset(
        root_dir=dataset_root,
        crop_size=(256, 256),
        degradation_types=['rain', 'snow']
    )
    print(f"   Multi-type dataset size: {len(multi_dataset)}")

    # Test 3: Load all data (default behavior)
    print("\n3. Loading all images (no filter):")
    all_dataset = AllWeatherDataset(
        root_dir=dataset_root,
        crop_size=(256, 256),
        degradation_types=None  # or simply omit this parameter
    )
    print(f"   Full dataset size: {len(all_dataset)}")

    # Test 4: Verify degradation type distribution
    print("\n4. Checking degradation type distribution in filtered dataset:")
    if len(rain_dataset) > 0:
        type_counts = {0: 0, 1: 0, 2: 0, -1: 0}
        # Sample a few items to check (up to 10 or dataset size)
        sample_size = min(10, len(rain_dataset))
        for i in range(sample_size):
            sample = rain_dataset[i]
            dtype = sample['degradation_type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        print(f"   Sampled {sample_size} items from rain dataset:")
        print(f"   - Hazy (0): {type_counts[0]}")
        print(f"   - Rain (1): {type_counts[1]}")
        print(f"   - Snow (2): {type_counts[2]}")
        print(f"   - Unknown (-1): {type_counts[-1]}")

    print("\n" + "=" * 60)
    print("Original Test: Default Processing Pipeline")
    print("=" * 60)

    # Test default processing pipeline
    dataset = AllWeatherDataset(root_dir=dataset_root, crop_size=(256, 256))
    sample = dataset[0]

    # Verify that returned images are Tensors
    input_img = sample['input_image']
    gt_img = sample['gt_image']

    print(f"Processed input image type: {type(input_img)}")
    print(f"Input image shape: {input_img.shape}")  # Should be [C,H,W] format
    print(f"Input image data range: {input_img.min().item():.4f} - {input_img.max().item():.4f}")

    # Convert back to numpy array for visualization
    input_np = input_img.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    gt_np = gt_img.permute(1, 2, 0).numpy()

    # Adjust data range (from [0,1] back to [0,255])
    input_np = (input_np * 255).astype(np.uint8)
    gt_np = (gt_np * 255).astype(np.uint8)

    # Save images for visual verification
    cv2.imwrite(os.path.join(output_dir, "processed_input.jpg"), cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "processed_gt.jpg"), cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))
    print(f"Saved processed image pair to {output_dir} folder")

    print("\nTesting complete!")