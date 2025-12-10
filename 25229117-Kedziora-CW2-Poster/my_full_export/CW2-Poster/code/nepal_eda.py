# Import of necessary libraries for EDA stage of the 'Himalayan Glacial Lakes: Sentinel-2 Image Collection Dataset'
# Specificities of dataset: false colour bands, specifically Bands 8, 4, and 3, with a resolution of 400 x 400 pixels.
# I also curated my own target dataset that I did not input in the training of the model, and will conduct a validation experiment on unseen data to push my learning further.
# This self-curated dataset isn't ground truth to reality as I had to generate some of the masks and used photoshop for 2 of the 10 images.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import random
from sklearn.model_selection import train_test_split
from collections import Counter

# Setup and Dynamic Path Configuration for reproducibility.
SEED = 42
IMG_SIZE = 256
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)

# Poster Colours.
COLOR_TRAIN = '#00008b'
COLOR_VAL = '#bbe4ed'

TRAIN_IMG_DIR = os.path.join(project_root, "images", "train", "raw")
TRAIN_MASK_DIR = os.path.join(project_root, "images", "train", "masks")
VAL_IMG_DIR = os.path.join(project_root, "images", "val", "raw")
VAL_MASK_DIR = os.path.join(project_root, "images", "val", "masks")

# Data Loading and Splitting.
def get_file_list(dir_path):
    if not os.path.exists(dir_path):
        return []
    # Simplified to strictly look for .png as per your confirmation
    return sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.png')])

all_train_files = get_file_list(TRAIN_IMG_DIR)
train_files, internal_val_files = train_test_split(all_train_files, test_size=0.2, random_state=SEED)
external_test_files = get_file_list(VAL_IMG_DIR)

# Sanity Check
print("Dataset Split Overview")
print(f"Training Set:       {len(train_files)} images")
print(f"Internal Val Set:   {len(internal_val_files)} images")
print(f"External Test Set:  {len(external_test_files)} images")
print()

# Verifying that no filenames training set appears in external test set.
train_set = set(train_files)
test_set = set(external_test_files)
intersection = train_set.intersection(test_set)

if len(intersection) > 0:
    print(f"Found {len(intersection)} duplicate filenames.")
    print(f"Files: {list(intersection)[:5]}...")
else:
    print("No filename leakage detected.")

# Hashing image files to detect identical images that may have different filenames - good to keep for experiments with other datasets.
# The following code might need review in case future datasets are larger.
def check_content_duplicates(img_dir, file_list, label):
    print(f"Checking content duplicates in {label}")
    hashes = {}
    duplicates = []
    
    for filename in file_list:
        path = os.path.join(img_dir, filename)
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        if file_hash in hashes:
            duplicates.append((filename, hashes[file_hash]))
        else:
            hashes[file_hash] = filename
            
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} pairs of identical images.")
        for new_file, original in duplicates[:3]:
            print(f"- {new_file} is identical to {original}")
    else:
        print(f"No identical image content found in {label}.")

check_content_duplicates(TRAIN_IMG_DIR, train_files, "Training Set")
print()

# Analysing the Red channel to quantify the domain shift between datasets.
def analyse_pixels(img_dir, file_list, label):
    print(f"Analysing {label} (Red Channel)")
    pixel_values = []
    
    # Randomly pick 100 images to get a quick statistical snapshot.
    sample_files = file_list if len(file_list) < 100 else random.sample(file_list, 100)
    
    for f in sample_files:
        img = cv2.imread(os.path.join(img_dir, f))
        if img is not None:
            # Extract only the Red channel as it highlights sediment/water differences well.
            pixel_values.append(img[:, :, 2].flatten())
    if len(pixel_values) == 0:
        return 0, 0
    
    all_pixels = np.concatenate(pixel_values)
    # Calculate the 'Center' (Mean) and 'Spread' (Std) of brightness.
    mean_val = np.mean(all_pixels)
    std_val = np.std(all_pixels)
    
    print(f"Mean: {mean_val:.2f} | Std: {std_val:.2f}")
    
    plt.figure(figsize=(10, 4))
    plot_color = COLOR_TRAIN if 'Train' in label else COLOR_VAL   
    sns.histplot(all_pixels, bins=50, color=plot_color, kde=True) # If the curves for Train and Test don't overlap, the AI will likely fail -> solved with CLAHE, Gaussian Blur, Grayscale, Brightness Augmentaiton later in adaptation.
    plt.title(f"Pixel Intensity Distribution ({label})")
    plt.xlabel("Pixel Value (0-255)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"EDA_Intensity_{label.replace(' ', '_')}.png")
    return mean_val, std_val

mean_train, std_train = analyse_pixels(TRAIN_IMG_DIR, train_files, "Training Data")
mean_test, std_test = analyse_pixels(VAL_IMG_DIR, external_test_files, "External Test Data")

print("Domain Shift Analysis Conclusion.")
# Calculate the absolute gap in average brightness between Training and Test data.
diff = abs(mean_train - mean_test)
print(f"Difference in Mean Brightness: {diff:.2f}")

# Threshold check: Is the difference big enough to confuse the model?
if diff > 20:
    print("Significant shift detected.")
else:
    print("Minor shift detected.")
print()

# Calculating the ratio of lake pixels to background to determine the necessity of weighted loss functions.
def analyse_masks(mask_dir, file_list, label):
    print(f"Analysing Class Balance for {label}")
    total_pixels = 0
    lake_pixels = 0

    # Direct path linking masks to filenames.
    for f in file_list:
        mask_path = os.path.join(mask_dir, f)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load as Greyscale for structural data.
            if mask is not None:
                mask = (mask > 127).astype(np.uint8) # Forcing grey edges to be strictly Black (0) or White (1) for accurate counting.
                total_pixels += mask.size # total pixel in img.
                lake_pixels += np.sum(mask) # total lake pixels.

    if total_pixels == 0:
        print("No masks found.")
        return

    # Calculating exactly what percentage of the data is actually 'Lake'.
    # If low -> weighted loss.
    lake_ratio = (lake_pixels / total_pixels) * 100
    bg_ratio = 100 - lake_ratio
    
    print(f"Lake: {lake_ratio:.2f}% | Background: {bg_ratio:.2f}%")
    
    # Pie chart for visual proof.
    plt.figure(figsize=(6, 6))
    plt.pie([lake_ratio, bg_ratio], labels=['Lake', 'Background'], autopct='%1.1f%%', colors=[COLOR_TRAIN, COLOR_VAL])
    plt.title(f"Class Balance: {label}")
    plt.savefig(f"EDA_ClassBalance_{label.replace(' ', '_')}.png")

analyse_masks(TRAIN_MASK_DIR, train_files, "Training Set")
analyse_masks(VAL_MASK_DIR, external_test_files, "External Test Set")
print()

# Visualising samples side-by-side to visually prove domain shift.
def plot_visual_samples(train_files, test_files, num_samples=3):
    plt.figure(figsize=(10, 6))
    
    train_samples = random.sample(train_files, num_samples)
    for i, f in enumerate(train_samples):
        img = cv2.imread(os.path.join(TRAIN_IMG_DIR, f))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(img)
            plt.title("Training (RGB)")
            plt.axis('off')

    test_samples = random.sample(test_files, num_samples)
    for i, f in enumerate(test_samples):
        img = cv2.imread(os.path.join(VAL_IMG_DIR, f))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(img)
            plt.title("Nepal (Target)")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("EDA_Domain_Gap_Visual.png")

# Boxplot.
# Checking how big real lakes are so I can delete noise in a later stage. 
def get_lake_sizes(mask_dir, file_list):
    sizes = []
    for f in file_list:
        mask_path = os.path.join(mask_dir, f)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.uint8) # Forces pixels to be 0 or 1.
                num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask) # Finding all separate lakes and calculating their stats.
                
                # Checks that object exists - 0 is background.
                if num > 1:
                    lake_areas = stats[1:, 4] # in stats table 4 is the Area. takes information from row 1 (skip background).
                    sizes.extend(lake_areas) # Adding to list of sizes.
    return sizes

train_sizes = get_lake_sizes(TRAIN_MASK_DIR, train_files)
test_sizes = get_lake_sizes(VAL_MASK_DIR, external_test_files)

if len(train_sizes) > 0 and len(test_sizes) > 0:
    plt.figure(figsize=(8, 6))
    data = [train_sizes, test_sizes]
    bplot = plt.boxplot(data, labels=['Training Lakes', 'Nepal Lakes'], patch_artist=True)
    box_colors = [COLOR_TRAIN, COLOR_VAL]
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)

    # As lake sizes vary wildly, a linear plot would squash everything - log scale is used to see small and big lakes.
    plt.yscale('log')
    plt.ylabel('Lake Size (Pixels) - Log Scale')
    plt.title('Distribution of Lake Object Sizes')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("EDA_Lake_Sizes.png")
    print(f"Avg Train Lake Size: {np.mean(train_sizes):.0f} pixels")
    print(f"Avg Tsho Rolpa Size: {np.mean(test_sizes):.0f} pixels")