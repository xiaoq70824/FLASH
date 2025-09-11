import torch
import os
import cv2
import numpy as np
from model.FLASH import FLASH
from glob import glob
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

# 测试配置参数
test_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_dir': 'D:/pycharm_workspace/FLASH/weights',
    'checkpoint_file': 'last.pth',
    'test_data_dir': 'D:/pycharm_workspace/Data_Processor/New_Dataset/test/input',
    'output_dir': 'D:/pycharm_workspace/FLASH/output',
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 32,
    'num_blocks': [4, 6, 6, 8],
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'drop_out_rate': 0.,
    'batch_size': 1,
    'num_workers': 2,
    'pad_factor': 32
}

os.makedirs(test_config['output_dir'], exist_ok=True)



class TestDataset:
    def __init__(self, root_dir, pad_factor=64, transform=None):
        self.root_dir = root_dir
        self.pad_factor = pad_factor
        self.input_paths = sorted(glob(os.path.join(root_dir, '*.*')))
        if len(self.input_paths) == 0:
            raise ValueError(f"No input images found in directory {root_dir}")
        print(f"Found {len(self.input_paths)} input images in {root_dir}")

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        input_path = self.input_paths[index]

        input_image = cv2.imread(input_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        original_height, original_width = input_image.shape[:2]

        if self.transform:
            input_tensor = self.transform(input_image)

        return {
            'input_image': input_tensor,
            'input_path': input_path,
            'original_size': (original_height, original_width)
        }


def create_test_dataloader(dataset, batch_size=1, shuffle=False, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def load_model(config):
    device = config['device']
    model = FLASH(
        inp_channels=config['inp_channels'],
        out_channels=config['out_channels'],
        dim=config['dim'],
        num_blocks=config['num_blocks'],
        num_refinement_blocks=config['num_refinement_blocks'],
        heads=config['heads'],
        ffn_expansion_factor=config['ffn_expansion_factor'],
        bias=config['bias'],
        LayerNorm_type=config['LayerNorm_type'],
        drop_out_rate=config['drop_out_rate']
    ).to(device)

    checkpoint_path = os.path.join(config['checkpoint_dir'], config['checkpoint_file'])
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully from checkpoint.")
    else:
        raise ValueError("No model state dict found in checkpoint")

    model.eval()
    return model


def enhance_and_save_image(model, input_tensor, original_size, output_path, device, pad_factor=64):
    _, h, w = input_tensor.shape

    # Calculate padding to make dimensions multiples of pad_factor
    H, W = ((h + pad_factor) // pad_factor) * pad_factor, ((w + pad_factor) // pad_factor) * pad_factor
    padh = H - h if h % pad_factor != 0 else 0
    padw = W - w if w % pad_factor != 0 else 0

    input_padded = F.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')
    input_padded = input_padded.to(device).float()

    with torch.no_grad():
        restored_padded = model(input_padded)

    restored_image = restored_padded[:, :, :h, :w]
    restored_image = torch.clamp(restored_image, 0, 1)
    restored_image = restored_image.squeeze().cpu()

    if (h, w) != original_size:
        restored_np = restored_image.permute(1, 2, 0).numpy()  # (H,W,C)
        restored_np = cv2.resize(restored_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
        restored_image = torch.from_numpy(restored_np.transpose(2, 0, 1))

    restored_image = restored_image.mul(255).clamp(0, 255).byte()
    restored_image = restored_image.permute(1, 2, 0).numpy()

    restored_image_bgr = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, restored_image_bgr)
    print(f"Enhanced image has been saved to: {output_path}")


if __name__ == "__main__":
    # Load model
    models = load_model(test_config)
    print("Load models Successfully")

    # Create test dataset and data loader
    test_dataset = TestDataset(root_dir=test_config['test_data_dir'], pad_factor=test_config['pad_factor'])
    test_loader = create_test_dataloader(test_dataset, batch_size=test_config['batch_size'],
                                         num_workers=test_config['num_workers'], shuffle=False)
    print("Create Dataset Successfully")
    print(f"Length of test_loader: {len(test_loader)}")

    # Enhance each test image and save results
    for batch_idx, batch_data in enumerate(test_loader):
        input_images = batch_data['input_image']
        input_paths = batch_data['input_path']
        original_sizes = batch_data['original_size']

        print(f"Batch {batch_idx} Start!")
        print(f"Input images batch size: {input_images.size(0)}")

        for i in range(input_images.size(0)):
            input_filename = os.path.basename(input_paths[i])
            output_path = os.path.join(test_config['output_dir'], input_filename)
            original_size = (original_sizes[0][i].item(), original_sizes[1][i].item())

            # Call enhance and save function
            enhance_and_save_image(models, input_images[i], original_size, output_path,
                                   test_config['device'], test_config['pad_factor'])

        # Output information about current batch completion
        print(f"Batch {batch_idx + 1}/{len(test_loader)} completed, enhanced {input_images.size(0)} images in total")

    print("Enhancement of all test images completed.")