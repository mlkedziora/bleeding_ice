# BONUS: Tsho Rolpa Glacier Lake evolution over years monitoring
# This is extra, and needs heavy refinement, but since Tsho Rolpa is my target to help local populations move when necessary, a time tracking system is necessary.
# Further improvements should include a ground physics approach (cubic meters of water), not just satellite images, as well as a ratio between rising global temperatures and glacier meltdown in that region. 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU, Dropout
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
# [NEW] Necessary imports for metric calculation (F1 & IoU).
from sklearn.metrics import f1_score, jaccard_score

# Dynamically locating the project root to ensure robustness across different environments.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) # Fallback if running from root

if not os.path.exists(os.path.join(project_root, "images")): 
    project_root = current_script_dir

TRAIN_IMG_DIR = os.path.join(project_root, "images", "train", "raw")
TRAIN_MASK_DIR = os.path.join(project_root, "images", "train", "masks")
VAL_IMG_DIR = os.path.join(project_root, "images", "val", "raw")
VAL_MASK_DIR = os.path.join(project_root, "images", "val", "masks")

# Output Directory for Temporal Results.
OUTPUT_DIR = os.path.join(project_root, "final_results_temporal")
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(TRAIN_IMG_DIR): 
    print("Paths not found.")
    exit()

# Specific chronological files for the longitudinal study phase.
MONITORING_FILES = {
    "2016-12-09": "2016-12-09-00_00_2016-12-09-23_59_Sentinel-2_L2A_True_color.png",
    "2017-12-14": "2017-12-14-00_00_2017-12-14-23_59_Sentinel-2_L2A_True_color.png",
    "2018-12-24": "2018-12-24-00_00_2018-12-24-23_59_Sentinel-2_L2A_True_color.png",
    "2019-11-19": "2019-11-19-00_00_2019-11-19-23_59_Sentinel-2_L2A_True_color.png",
    "2020-11-13": "2020-11-13-00_00_2020-11-13-23_59_Sentinel-2_L2A_True_color.png",
    "2021-12-08": "2021-12-08-00_00_2021-12-08-23_59_Sentinel-2_L2A_True_color.png",
    "2022-12-18": "2022-12-18-00_00_2022-12-18-23_59_Sentinel-2_L2A_True_color.png",
    "2023-12-18": "2023-12-18-00_00_2023-12-18-23_59_Sentinel-2_L2A_True_color.png",
    "2024-11-07": "2024-11-07-00_00_2024-11-07-23_59_Sentinel-2_L2A_True_color.png",
    "2025-11-22": "2025-11-22-00_00_2025-11-22-23_59_Sentinel-2_L2A_True_color.png"
}

# Model Constants matching the training phase.
IMG_SIZE = 256
CHANNELS = 1
DROPOUT_RATE = 0.5 

# Architecture (John & Zhang, 2022).
# Re-implementing the exact Attention U-Net structure used in Nepal_repro.keras to ensure weight compatibility.
# Reminder: g -> what to look for | s -> raw feature to look at.
def attention_gate(g, s, num_filters):
    # 1x1 convolutions to align dimensions.
    Wg = Conv2D(num_filters, 1, padding="same")(g); Wg = BatchNormalization()(Wg)
    Ws = Conv2D(num_filters, 1, padding="same")(s); Ws = BatchNormalization()(Ws)
    # ReLU activates regions where signals agree.
    out = Activation("relu")(Wg + Ws)
    # Sigmoid squashes values (important = 1 | irrelevant = 0).
    out = Conv2D(1, 1, padding="same")(out); out = Activation("sigmoid")(out)
    # Supressing noise in the skip connection.
    return out * s

def conv_block(input, num_filters):
    # Standard Double Convolution Block with added Dropout for regularisation.
    x = Conv2D(num_filters, 3, padding="same")(input); x = BatchNormalization()(x); x = ReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv2D(num_filters, 3, padding="same")(x); x = BatchNormalization()(x); x = ReLU()(x)
    return x

def decoder_block(input, skip, num_filters):
    # Upsampling via Transposed Convolution.
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # Attention Gate applied to the skip connection (John & Zhang, 2022).
    skip = attention_gate(x, skip, num_filters)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def get_model_architecture():
    # Constructs the Attention U-Net used for the Nepal Adaptation.
    inputs = Input((IMG_SIZE, IMG_SIZE, CHANNELS))
    
    # Encoder - compressing image to extract features.
    s1 = conv_block(inputs, 16); p1 = MaxPooling2D((2, 2))(s1)
    s2 = conv_block(p1, 32); p2 = MaxPooling2D((2, 2))(s2)
    s3 = conv_block(p2, 64); p3 = MaxPooling2D((2, 2))(s3)
    s4 = conv_block(p3, 128); p4 = MaxPooling2D((2, 2))(s4)
    
    # Bottom of the U - most abstract features.
    b1 = conv_block(p4, 256) 
    
    # Decoder - expanding image with attention-gated skip connections.
    d1 = decoder_block(b1, s4, 128); d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32); d4 = decoder_block(d3, s1, 16)
    
    # Sigmoid activation for binary classification (Lake/Non-Lake).
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid", dtype='float32')(d4)
    
    return Model(inputs, outputs, name="Nepal_Model")

# Post-Processing Strategy - Tsho Rolpa.
# Since the model might pick up small noise blobs, we filter for the largest connected component.
def post_process_mask(pred_mask):
    pred_mask_uint = (pred_mask * 255).astype(np.uint8)

    # Morphological Opening to remove small white noise.
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(pred_mask_uint, cv2.MORPH_OPEN, kernel)
    
    # Keep Largest Connected Component (Tsho Rolpa Aim)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8) 
    
    if num_labels > 1: # Index 0 is background.
        # Get largest component (skipping background at index 0).
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
        cleaned_mask = np.zeros_like(opening)
        cleaned_mask[labels == largest_label] = 255
        return cleaned_mask.astype(np.float32) / 255.0
    
    return opening.astype(np.float32) / 255.0

# Executes the longitudinal study on Tsho Rolpa.
# Tracks lake surface area (pixel count) over time with a specific threshold > 0.60.
# Generates PLOT 4.
def monitor_tsho_rolpa_evolution(model, monitoring_files, image_dir, mask_dir=None):
    dates = []
    lake_pixel_counts = []
    visualisation_data = [] # Stores (Image, True Mask, Raw Pred, Clean Pred)
    
    # Sort files by date key to ensure chronological order in the graph
    sorted_keys = sorted(monitoring_files.keys())

    for date in sorted_keys:
        filename = monitoring_files[date]
        img_path = os.path.join(image_dir, filename)
        
        # Load Image
        if not os.path.exists(img_path):
            print(f"Warning: File {filename} not found in {image_dir}. Skipping.")
            continue
            
        # Loading and basic preprocessing.
        # Note: Model expects Grayscale (1 channel) as per training config.
        img_raw = cv2.imread(img_path)
        
        # Conversion to Grayscale to match Training Pipeline.
        # Deep shadows in valleys confused the RGB model; Grayscale preserves texture better.
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        
        # CLAHE Application (Critical for Nepal context).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_gray)
        
        # Resizing and Normalisation.
        img_resized = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=-1) # Add channel dim
        img_batch = np.expand_dims(img_batch, axis=0) # Add batch dim

        # Load True Mask (If available, for comparison)
        true_mask = np.zeros((IMG_SIZE, IMG_SIZE)) # Default empty
        has_ground_truth = False
        current_f1 = "N/A"
        current_iou = "N/A"
        
        if mask_dir:
            mask_path = os.path.join(mask_dir, filename) 
            if os.path.exists(mask_path):
                t_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Binarise ground truth to strictly 0 or 1 for metric calculation.
                true_mask = (cv2.resize(t_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
                has_ground_truth = True

        # Predict & Apply Custom Threshold (0.60)
        # We predict raw, then apply the strict threshold you requested.
        pred_prob = model.predict(img_batch, verbose=0).squeeze()
        
        # Custom Thresholding: Only count if confidence > 0.60.
        pred_thresholded = (pred_prob > 0.60).astype(np.float32)

        # Post-Process (Largest Connected Component)
        pred_clean = post_process_mask(pred_thresholded)
        
        # [NEW] Metric Calculation Loop.
        if has_ground_truth and np.sum(true_mask) > 0:
            # Flatten arrays to 1D lists as required by sklearn metrics.
            y_true_f = true_mask.flatten()
            y_pred_f = pred_clean.flatten()
            
            # Binary metrics because we care about the "Lake" class specifically.
            val_f1 = f1_score(y_true_f, y_pred_f, average='binary', zero_division=1)
            val_iou = jaccard_score(y_true_f, y_pred_f, average='binary', zero_division=1)
            
            current_f1 = f"{val_f1:.4f}"
            current_iou = f"{val_iou:.4f}"

        # Track Metrics
        # Summing white pixels (value 1.0).
        white_pixels = np.sum(pred_clean)
        dates.append(date)
        lake_pixel_counts.append(white_pixels)
        
        print(f"Processed {date}: Size={white_pixels:.0f} px | F1: {current_f1} | IoU: {current_iou}")
        
        # Store for plotting - Saving RGB version for visual context, but model used Gray.
        img_viz = cv2.resize(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        visualisation_data.append({
            'date': date,
            'img': img_viz,
            'true': true_mask,
            'raw': pred_prob, # Keep raw probability for heat map visualisation.
            'clean': pred_clean,
            'f1': current_f1, # Store F1 for title
            'iou': current_iou # Store IoU for title
        })

    # PLOT 4: Longitudinal Analysis
    if len(dates) > 0:
        n_samples = len(dates)
        
        # Nepal_Final_Viz layout
        fig = plt.figure(figsize=(12, 5 + (3.5 * n_samples))) 
        
        # Create a GridSpec: Top is the graph, Bottom rows are the images.
        gs = gridspec.GridSpec(n_samples + 1, 4, height_ratios=[1.5] + [1]*n_samples)
        
        # Subplot 1: The Evolution Graph (Spanning all columns at the top).
        ax_graph = fig.add_subplot(gs[0, :])
        ax_graph.plot(dates, lake_pixel_counts, marker='o', linestyle='-', color='b', linewidth=2)
        ax_graph.set_title(f"Tsho Rolpa Surface Evolution (Cleaned Prediction) - {dates[0]} to {dates[-1]}")
        ax_graph.set_ylabel("Lake Area (Pixels)")
        ax_graph.grid(True, linestyle='--', alpha=0.7)
        plt.setp(ax_graph.get_xticklabels(), rotation=45, ha="right")

        # Subplot 2: Image Rows.
        for i, data in enumerate(visualisation_data):
            # Column 1: Original Image.
            ax1 = fig.add_subplot(gs[i+1, 0])
            ax1.imshow(data['img'])
            ax1.set_title(f"{data['date']}\nSatellite Input")
            ax1.axis('off')

            # Column 2: True Mask (Ground Truth).
            ax2 = fig.add_subplot(gs[i+1, 1])
            ax2.imshow(data['true'], cmap='gray')
            ax2.set_title("True Mask")
            ax2.axis('off')

            # Column 3: Raw Prediction (Heatmap).
            ax3 = fig.add_subplot(gs[i+1, 2])
            pos = ax3.imshow(data['raw'], cmap='jet', vmin=0, vmax=1)
            ax3.set_title("Raw Prediction (Prob)")
            ax3.axis('off')

            # Column 4: Final Cleaned Mask (Threshold > 0.6 + Post-Process).
            ax4 = fig.add_subplot(gs[i+1, 3])
            ax4.imshow(data['clean'], cmap='gray')
            
            # [CHANGE 2] Updated title to display F1 and IoU metrics for performance comparison.
            ax4.set_title(f"Final Output\n(F1: {data['f1']} | IoU: {data['iou']})") 
            ax4.axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "Nepal_Longitudinal_Analysis.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")
        
        # Print Summary for user.
        print("\nSummary of Evolution")
        for d, p in zip(dates, lake_pixel_counts):
            print(f"Date: {d} | Lake Size: {p:.0f} pixels")

    return dates, lake_pixel_counts

# Main Execution Pipeline.
if __name__ == "__main__":
    model = get_model_architecture()
    # Load Weights (Assuming weights are in project root or current dir)
    weights_path = 'Nepal_repro.keras' 
    print(f"Loading Weights from {weights_path}...")
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print("Weights file not found!")
        exit()

    # Run Monitoring
    # Using VAL_IMG_DIR here because Tsho Rolpa is your external validation set.
    monitor_tsho_rolpa_evolution(
        model=model,
        monitoring_files=MONITORING_FILES,
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR
    )