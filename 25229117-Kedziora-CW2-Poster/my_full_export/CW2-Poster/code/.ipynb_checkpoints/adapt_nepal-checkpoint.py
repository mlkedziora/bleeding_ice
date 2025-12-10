# Adaptation of Attention U-Net for Glacial Lake Outburst Flood (GLOF) detection in Nepal - attempt of monitoring unseen data: Tsho Rolpa.

import os
import random 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU, Dropout
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scipy import stats

# Seed for reproducibility and transferability analysis.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' 
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed Precision for performance optimisation on T4 GPU.
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configuration adapted for the Himalayan context.
# Image size reduced to 256 from the original 512 due to the smaller resolution of the specific glacial lake 'region of interest'.
# CHANNELS set to 1 (Grayscale) instead of 3 (RGB) to focus on structural contrast rather than spectral signature (material fingerprint).
print("Nepal Adaptation Optimised")
IMG_SIZE = 256 
CHANNELS = 1 
LEARNING_RATE = 0.001 
BATCH_SIZE = 16 
EPOCHS = 250 
DROPOUT_RATE = 0.5 # Increased Dropout to 0.5 to force shape learning over texture, because the training dataset has different colours from the aimed Tsho Rolpa.
STEPS_PER_EPOCH = 150

COLOR_TRAIN = 'darkblue'
COLOR_VAL = '#bbe4ed'

# Setup for reproducibility and locating the dataset root.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) 
TRAIN_IMG_DIR = os.path.join(project_root, "images", "train", "raw")
TRAIN_MASK_DIR = os.path.join(project_root, "images", "train", "masks")
VAL_IMG_DIR = os.path.join(project_root, "images", "val", "raw")
VAL_MASK_DIR = os.path.join(project_root, "images", "val", "masks")

# Safety guard to ensure paths exist.
if not os.path.exists(TRAIN_IMG_DIR):
    print("Paths not found.")
    exit()

# Custom Loss Functions to address class imbalance.
# Similar to the original paper's use of weighted metrics, we combine Binary Cross Entropy and Dice Loss.
# BCE because checks Lake vs Land.
# Dice Loss because it prevents from a model.
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Weighted combination to penalise missing the lake (False Negatives) more heavily than False Positives.
def weighted_combo_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    weight_lake = 15.0 
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = - (weight_lake * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)


# Adapted pipeling to address the specific domain shift challenge of the Himalayas dataset.
class NepalGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, mask_dir, file_ids=None, augment=False, is_validation=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.is_validation = is_validation
        
        # Since my images are all .png I am using this - for adapting the model to another dataset that doesn't have .png please change the following code snippet.
        if file_ids is None:
            self.img_ids = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
        else:
            self.img_ids = file_ids
        
        self.valid_pairs = []
        for img_name in self.img_ids:
             base = os.path.splitext(img_name)[0]
             # Direct pairing since masks are .png too.
             mask_name = base + ".png"
             mask_path = os.path.join(mask_dir, mask_name)
             
             # Pragmatic check to ensure every image has a corresponding mask before training.
             if os.path.exists(mask_path):
                 self.valid_pairs.append((img_name, mask_name))
             else:
                 print(f"Mask not found for {img_name} - skipping.")

    # Calculating exact number of batches needed.
    def __len__(self):
        if self.is_validation: return int(np.ceil(len(self.valid_pairs) / float(BATCH_SIZE)))
        return STEPS_PER_EPOCH

    def __getitem__(self, index):
        if self.is_validation:
            start = index * BATCH_SIZE
            end = min((index + 1) * BATCH_SIZE, len(self.valid_pairs))
            batch_pairs = self.valid_pairs[start : end]
        else:
            if len(self.valid_pairs) > 0: batch_pairs = random.choices(self.valid_pairs, k=BATCH_SIZE)
            else: batch_pairs = []

        images, masks = [], []
        # Loading and processing files.
        for img_name, mask_name in batch_pairs:
            img = cv2.imread(os.path.join(self.img_dir, img_name))
            if img is None: continue

            # Conversion to Grayscale. 
            # The original paper uses RGB channels, but the 'Red Channel' trick fails in the Himalayas because deep shadows in the valleys have low Red values, confusing the model.
            # Grayscale averages the channels, preserving texture better in shadow/snow contrast for the aimed Tsho Rolpa.
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalisation).
            # Normalises lighting differences between bright snow and dark terrain to assist feature extraction.
            # CLAHE works through tiled focus on the img and boosts the contrast in that tile, in this use case, it helps to locate lake boundaries within deep shadows, differentiating smooth water from rough rock texture.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_enhanced = clahe.apply(img_gray)
            
            img = cv2.resize(img_enhanced, (IMG_SIZE, IMG_SIZE))
            # Reshaping to for Keras compatibility.
            img = np.expand_dims(img, axis=-1)
            
            mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            # Data augmentation as described in the paper (rotation, reflection, zooming).
            # Additional intensity augmentations added to simulate environmental changes.
            if self.augment:
                # Geometry Augmentations.
                if np.random.rand() > 0.5: img = np.fliplr(img); mask = np.fliplr(mask)
                if np.random.rand() > 0.5: img = np.flipud(img); mask = np.flipud(mask)
                if np.random.rand() > 0.5:
                    k = np.random.randint(1, 4); img = np.rot90(img, k); mask = np.rot90(mask, k)
                
                # Intensity Augmentations - critical for current domain shift.
                if np.random.rand() > 0.2:
                    contrast = np.random.uniform(0.7, 1.3)
                    brightness = np.random.uniform(-40, 40)
                    img = img.astype(np.float32) * contrast + brightness
                    img = np.clip(img, 0, 255)

                # Zooming and Shearing.
                if np.random.rand() > 0.3: 
                    # Randomises geometry to make the model invariant to minor satellite altitude or angle changes.
                    scale = np.random.uniform(0.85, 1.15); shear = np.random.uniform(-0.1, 0.1)
                    rows, cols = IMG_SIZE, IMG_SIZE                  
                    M = np.float32([[scale, shear, cols*(1-scale)/2], [shear, scale, rows*(1-scale)/2]]) # Constructing the Affine Matrix - (cols*(1-scale)/2) ensures the zoom remains centered.
                    img = cv2.warpAffine(img.squeeze(), M, (cols, rows), borderMode=cv2.BORDER_REFLECT) # BORDER_REFLECT fills empty space with mirrored pixels to avoid black borders.
                    if len(img.shape) == 2: img = np.expand_dims(img, axis=-1) # OpenCv removes the channel dimension for grayscale - adding it back.
                    mask = cv2.warpAffine(mask.squeeze().astype(np.float32), M, (cols, rows), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST) # INTER_NEAREST used for the mask to prevent interpolation from creating decimal values.
                    mask = np.expand_dims(mask, axis=-1)
                
                # As Tsho Rolpa is milky/cloudy compared to clear training lakes, using Gaussian Blur on training forces the model to rely less on sharp textures and more on shape.
                if np.random.rand() > 0.3:
                    img_sq = img.squeeze()
                    img_sq = cv2.GaussianBlur(img_sq, (3, 3), 0)
                    img = np.expand_dims(img_sq, axis=-1)

            # Normalisation of pixel values.
            images.append(img.astype(np.float32) / 255.0)
            mask = mask.astype(np.float32) / 255.0
            # Ensuring strict binary masks.
            mask[mask > 0.5] = 1.0; mask[mask <= 0.5] = 0.0
            masks.append(mask)
            
        return np.array(images), np.array(masks)

# Architecture (John & Zhang, 2022).
# 'Skip connection' modified with attention gates to filter relevant features.
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

# Standard Double Convolution Block with added Dropout for regularisation.
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input); x = BatchNormalization()(x); x = ReLU()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv2D(num_filters, 3, padding="same")(x); x = BatchNormalization()(x); x = ReLU()(x)
    return x

# Upscaling Block with Attention Mechanism.
def decoder_block(input, skip, num_filters):
    # Upsampling via Transposed Convolution.
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # Attention Gate applied to the skip connection (John & Zhang, 2022).
    skip = attention_gate(x, skip, num_filters)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

# Assembling the full model (Fig. 3 - John & Zhang, 2022).
def build_model():
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

# Training Pipeline.
if __name__ == "__main__":
    model = build_model()
    
    # Check if weights exist to avoid retraining.
    weights_path = 'Nepal_repro.keras'
    weights_loaded = False
    
    if os.path.exists(weights_path):
        print("Loading existing weights from 'Nepal_repro.keras' - Skipping Training.")
        model.load_weights(weights_path)
        weights_loaded = True
        history = None
        
    # Using 'Adam' optimiser as per the original paper's recommendation.
    # Compilation happens after weight loading to ensure the optimizer wrapper does not interfere.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=weighted_combo_loss, metrics=['accuracy'])

    # Loading file lists for splitting.
    # Uniform .png Glacier Lake dataset - change if other datasets are tested.
    all_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith('.png')])
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=SEED)
    
    print(f"Split check | Training Files: {len(train_files)} | Validation Files: {len(val_files)}")
    
    # Initialising generators.
    train_gen = NepalGenerator(TRAIN_IMG_DIR, TRAIN_MASK_DIR, file_ids=train_files, augment=True, is_validation=False)
    val_gen = NepalGenerator(TRAIN_IMG_DIR, TRAIN_MASK_DIR, file_ids=val_files, augment=False, is_validation=True)
    test_gen = NepalGenerator(VAL_IMG_DIR, VAL_MASK_DIR, augment=False, is_validation=True)

    # Callbacks for optimal training - delete Nepal_repro.keras in CW2-Poster or move to holder if checking full training - count about 45min - 1h on T4.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('Nepal_repro.keras', monitor='val_loss', save_best_only=True, mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6), # Changes step size after patience 15not improved models.
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True) # Play around with patience if necessary for new adaptation.
    ]

    # Execute training only if weights were not loaded.
    if not weights_loaded:
        print(f"Training {len(train_files)} (Steps: {STEPS_PER_EPOCH})")
        history = model.fit(train_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=val_gen, callbacks=callbacks)

    # PLOT 1: Training Curves.
    if history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss', color=COLOR_TRAIN, linewidth=2)
        plt.plot(history.history['val_loss'], label='Val Loss', color=COLOR_VAL, linewidth=2)
        plt.title('Loss Curve', color='black', fontweight='bold')
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
        plt.grid(True, linestyle=':', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy', color=COLOR_TRAIN, linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color=COLOR_VAL, linewidth=2)
        plt.title('Accuracy Curve', color='black', fontweight='bold')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.savefig("Nepal_Training_Curves.png", dpi=300)
        print("Saved Nepal_Training_Curves.png")

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
    
    # BONUS: Tsho Rolpa Glacier Lake evolution over years monitoring
    # This is extra, and needs heavy refinement, but since Tsho Rolpa is my target to help local populations move when necessary, a time tracking system is necessary.
    # Further improvements should include a ground physics approach (cubic meters of water), not just satellite images, as well as a ratio between rising global temperatures and glacier meltdown in that region. 
    # MOVED THE CODE TO -> monitor_tsho_rolpa.py

    # Evaluation loop.
    def evaluate(generator, name, use_post_processing=False):
        print(f"\nEvaluating {name} {'(with Post-Processing)' if use_post_processing else ''}")
        # Note: redundant weight loading removed here to prevent optimizer conflict.
        y_true, y_pred, scores = [], [], []
        
        for i in range(len(generator)):
            imgs, masks = generator[i]
            if len(imgs) == 0: continue
            
            # Test Time Augmentation (TTA) to improve robustness.
            p1 = model.predict(imgs, verbose=0) # Standard Prediction
            p2 = np.fliplr(model.predict(np.fliplr(imgs), verbose=0)) # Horizontal Flip
            p3 = np.flipud(model.predict(np.flipud(imgs), verbose=0)) # Vertical Flip
            preds = (p1 + p2 + p3) / 3.0
            preds_bin = np.round(preds)

            # Apply Post-Processing per image if requested.
            if use_post_processing:
                processed_preds = []
                for p in preds_bin:
                    p_clean = post_process_mask(p.squeeze())
                    processed_preds.append(np.expand_dims(p_clean, axis=-1))
                preds_bin = np.array(processed_preds)

            for j in range(len(masks)):
                # Calculating F1 score per image for CI.
                f1 = f1_score(masks[j].flatten(), preds_bin[j].flatten(), average='binary', zero_division=1)
                scores.append(f1)
            
            # Reminder: flattening is to put into 1D list to calculate global metrics.
            y_true.extend(masks.flatten())
            y_pred.extend(preds_bin.flatten())

        mean_f1 = np.mean(scores)
        if len(scores) > 1:
            ci = stats.t.interval(0.95, len(scores)-1, loc=mean_f1, scale=stats.sem(scores))
        else:
            ci = (mean_f1, mean_f1)
        
        print(f"   Binary F1:   {f1_score(y_true, y_pred, average='binary'):.4f}")
        # "Weighted metrics were used as they account for class imbalance" (John & Zhang, 2022).
        print(f"   Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
        # IoU (Jaccard Index) used as a stricter metric for shape overlap.
        print(f"   IoU:         {jaccard_score(y_true, y_pred, average='binary'):.4f}")
        # Quantifying stability of the model by estimating range where true mean score falls.
        print(f"   95% CI:       {ci[0]:.4f} - {ci[1]:.4f}")

        return scores, y_true, y_pred

    # Executing Evaluation on all 4 sets.
    train_scores, _, _ = evaluate(train_gen, "Train")
    val_scores, _, _ = evaluate(val_gen, "Internal Val")
    test_scores_raw, _, _ = evaluate(test_gen, "External Test (Tsho Rolpa) - RAW")
    test_scores_clean, test_true_flat, test_pred_flat = evaluate(test_gen, "External Test (Tsho Rolpa) - CLEANED", use_post_processing=True)
    
    # PLOT 2: Confusion Matrix.
    cm = confusion_matrix(test_true_flat, test_pred_flat, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Background', 'Lake'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='.3f')
    plt.title('Pixel-Wise Normalized Confusion Matrix', color='black', fontweight='bold')
    plt.savefig("Nepal_Confusion_Matrix.png", dpi=300)
    print("Saved Nepal_Confusion_Matrix.png")

    # PLOT 3: Box Plot of F1 Scores.
    plt.figure(figsize=(8, 6))
    box = plt.boxplot([test_scores_raw, test_scores_clean], labels=['Raw', 'Post-Processed'], patch_artist=True)
    for patch in box['boxes']:
        patch.set_facecolor(COLOR_VAL)
        patch.set_edgecolor(COLOR_TRAIN)
    for median in box['medians']:
        median.set_color(COLOR_TRAIN)
        median.set_linewidth(2)

    plt.title('F1 Score Distribution (Statistical Significance)', color='black', fontweight='bold')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("Nepal_F1_Boxplot.png", dpi=300)
    print("Saved Nepal_F1_Boxplot.png")

    # Visualisation of results.
    imgs, masks = test_gen[0]
    preds = model.predict(imgs, verbose=0)
    plt.figure(figsize=(10, 10))
    for i in range(min(3, len(imgs))):
        p_clean = post_process_mask(np.round(preds[i]).squeeze())
        
        # Calculating Accuracy (IoU) for a specific image.
        intersection = np.logical_and(masks[i].squeeze(), p_clean).sum()
        union = np.logical_or(masks[i].squeeze(), p_clean).sum()
        iou_score = intersection / (union + 1e-6) 

        plt.subplot(3, 4, i*4+1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title("Input (Gray)", color='black', fontweight='bold')
        plt.axis('off')

        plt.subplot(3, 4, i*4+2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Ground Truth", color='black', fontweight='bold')
        plt.axis('off')

        plt.subplot(3, 4, i*4+3)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title("Raw Pred", color='black', fontweight='bold')
        plt.axis('off')
        
        plt.subplot(3, 4, i*4+4)
        plt.imshow(p_clean, cmap='gray')
        plt.title(f"Cleaned (IoU: {iou_score:.2f})", color='black', fontweight='bold')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("Nepal_Final_Viz.png", dpi=300)
    print("Saved Nepal_Final_Viz.png")