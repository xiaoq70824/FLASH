import os
import re
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from collections import defaultdict
import csv
from glob import glob


def parse_filename(filename):
    """
    Parse filename to extract degradation type and index.
    For example, "rain_001.png" -> ("rain", "001")
    """
    pattern = r'^(.*?)_(\d+)\.(png|jpg|jpeg|bmp|tiff)$'
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        degradation_type, index, _ = match.groups()
        return degradation_type, index
    else:
        return None, None


def load_image(image_path):
    """
    Load image and convert to RGB format.

    Args:
    - image_path: Image file path

    Returns:
    - image: Loaded image (H x W x C) normalized to [0, 1]
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image_rgb


def main():
    """
    Calculate average PSNR and SSIM of generated images, categorized by degradation type.
    Directly compare images at original resolution without resizing.
    """
    # Set paths for generated and reference image folders
    generated_images_dir = "D:/pycharm_workspace/FLASH/output"  # Replace with your generated images folder path
    reference_images_dir = "D:/pycharm_workspace/Data_Processor/New_Dataset/test/gt"  # Replace with your reference images folder path

    # Whether to save detailed results to CSV file
    save_csv = True  # Set to False if saving is not needed
    csv_path = "image_quality_results.csv"  # Specify the CSV file save path

    # Check if folders exist
    if not os.path.isdir(generated_images_dir):
        print(f"Generated images folder does not exist: {generated_images_dir}")
        return
    if not os.path.isdir(reference_images_dir):
        print(f"Reference images folder does not exist: {reference_images_dir}")
        return

    # Initialize statistical variables
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    # Use defaultdict to store PSNR and SSIM by degradation type
    degradation_stats = defaultdict(lambda: {'psnr': 0.0, 'ssim': 0.0, 'count': 0})

    # Initialize list for detailed results if needed
    results = [] if save_csv else None

    # Iterate through generated images folder
    for filename in os.listdir(generated_images_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue  # Skip non-image files

        degradation_type, index = parse_filename(filename)
        if not degradation_type or not index:
            print(f"Unable to parse filename: {filename}")
            continue

        generated_image_path = os.path.join(generated_images_dir, filename)
        reference_image_path = os.path.join(reference_images_dir, filename)  # Use same filename

        if not os.path.exists(reference_image_path):
            print(f"Reference image does not exist: {reference_image_path}")
            continue

            # Load generated image
        try:
            generated_image = load_image(generated_image_path)
        except Exception as e:
            print(f"Error loading generated image ({filename}): {e}")
            continue

            # Load reference image
        try:
            reference_image = load_image(reference_image_path)
        except Exception as e:
            print(f"Error loading reference image ({filename}): {e}")
            continue

        # Ensure generated and reference images have consistent dimensions (should already be consistent since test.py restores output to original size)
        if generated_image.shape != reference_image.shape:
            print(f"Image dimensions mismatch ({filename}): "
                  f"Generated image size {generated_image.shape}, Reference image size {reference_image.shape}")
            continue

        # Print image dimensions and channels for debugging
        print(f"Processing image: {filename}")

        # Calculate PSNR and SSIM
        try:
            psnr = compute_psnr(reference_image, generated_image, data_range=1.0)
            # For SSIM, use channel_axis=-1
            ssim = compute_ssim(reference_image, generated_image, channel_axis=-1, data_range=1.0)
        except Exception as e:
            print(f"Error calculating PSNR/SSIM ({filename}): {e}")
            continue

        # Update overall statistics
        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # Update statistics by degradation type
        degradation_stats[degradation_type]['psnr'] += psnr
        degradation_stats[degradation_type]['ssim'] += ssim
        degradation_stats[degradation_type]['count'] += 1

        # Save detailed results if needed
        if save_csv:
            results.append({
                'filename': filename,
                'degradation_type': degradation_type,
                'psnr': psnr,
                'ssim': ssim
            })

    if count == 0:
        print("No valid image pairs found for evaluation.")
        return

    # Calculate overall averages
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"\nOverall Average PSNR: {avg_psnr:.4f}")
    print(f"Overall Average SSIM: {avg_ssim:.4f}\n")

    # Calculate and print averages for each degradation type
    print("Average PSNR and SSIM by degradation type:")
    print("-----------------------------------")
    for degradation, stats in degradation_stats.items():
        if stats['count'] == 0:
            continue
        avg_degradation_psnr = stats['psnr'] / stats['count']
        avg_degradation_ssim = stats['ssim'] / stats['count']
        print(f"Degradation type: {degradation}")
        print(f"  Average PSNR: {avg_degradation_psnr:.4f}")
        print(f"  Average SSIM: {avg_degradation_ssim:.4f}")
        print("-----------------------------------")

    # Save detailed results to CSV file if needed
    if save_csv and results:
        try:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # Create directory if it doesn't exist
            with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['filename', 'degradation_type', 'psnr', 'ssim']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            print(f"Detailed results saved to {csv_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")


if __name__ == "__main__":
    main()