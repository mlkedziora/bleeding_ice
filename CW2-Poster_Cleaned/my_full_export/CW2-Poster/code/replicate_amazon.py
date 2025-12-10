# Replication of Attention U-Net (RGB) with modern env - documented in requirements.txt
import os
import random 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, ReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix, ConfusionMatrixDisplay
from scipy import stats

# Seed for reproducibility.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Verifying GPU Configuration - if running for the first time Tesla T4 or more performant GPU recommended (lengthy run).
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Detected: {len(gpus)} device(s) active.")
    except RuntimeError as e:
        print(e)

# Configuration from original paper.
IMG_SIZE = 512
CHANNELS = 3 
LEARNING_RATE = 0.0005 
BATCH_SIZE = 4 
EPOCHS = 50 
STEPS_PER_EPOCH = 100 # Defining 1 Epoch = 4 batch * 100 steps to achieve sufficient graient updates for convergence - this is necessary because of the data augmentation strategy. 

COLOR_TRAIN = 'darkblue'
COLOR_VAL = '#bbe4ed'

# Setup for reproducibility (I encountered many errors so keeping for future reference). 
# Search function to locate the dataset regardless of the current working directory.
def find_repo_location(start_path, target_folder_name="baseline_repo"):
    print(f"Crawling {start_path} looking for '{target_folder_name}'...")
    for root, dirs, files in os.walk(start_path):
        if target_folder_name in dirs:
            full_path = os.path.join(root, target_folder_name)
            print(f"Found '{target_folder_name}' at: {full_path}")
            return full_path
    return None

# Locating the dataset root.
current_dir = os.getcwd()
baseline_repo = find_repo_location(current_dir)

if baseline_repo is None:
    print("Could not find 'baseline_repo'.")
    print("Unzip dataset inside your project.")
    exit()

# Defining absolute paths for Training and Validation.
dataset_root = os.path.join(baseline_repo, "Amazon Forest Dataset")
TRAIN_IMG_DIR = os.path.join(dataset_root, "Training", "images")
TRAIN_MASK_DIR = os.path.join(dataset_root, "Training", "masks")
VAL_IMG_DIR = os.path.join(dataset_root, "Validation", "images")
VAL_MASK_DIR = os.path.join(dataset_root, "Validation", "masks")

# Handling potential nested folder structures.
if not os.path.exists(TRAIN_IMG_DIR):
    print(f"Error: Path found but 'Training/images' is missing: {TRAIN_IMG_DIR}")
    nested_path = os.path.join(dataset_root, "Amazon Forest Dataset", "Training", "images")
    if os.path.exists(nested_path):
        print("Found nested folder structure. Adjusting path...")
        dataset_root = os.path.join(dataset_root, "Amazon Forest Dataset")
        TRAIN_IMG_DIR = os.path.join(dataset_root, "Training", "images")
        TRAIN_MASK_DIR = os.path.join(dataset_root, "Training", "masks")
        VAL_IMG_DIR = os.path.join(dataset_root, "Validation", "images")
        VAL_MASK_DIR = os.path.join(dataset_root, "Validation", "masks")

# Loading data. 
class AmazonGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_dir, mask_dir, augment=False, is_validation=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.is_validation = is_validation
        
        if not os.path.exists(img_dir):
            print(f"{img_dir} does not exist.")
            self.img_ids = []
        else:
            self.img_ids = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
        
        self.valid_pairs = []
        for img_name in self.img_ids:
            base = os.path.splitext(img_name)[0]
            for ext in [".png", ".tif", ".tiff", ".jpg"]:
                if os.path.exists(os.path.join(mask_dir, base + ext)):
                    self.valid_pairs.append((img_name, base + ext))
                    break
        # Sanity check of paired images.
        if len(self.valid_pairs) > 0:
            print(f"Loaded {len(self.valid_pairs)} images from {os.path.basename(os.path.dirname(img_dir))}")
        else:
            print(f"No pairs found in {img_dir}")

    # Calculating exact number of batches needed to see all data once - during training 1 Epoch = 100 steps.
    def __len__(self):
        if self.is_validation:
            return int(np.ceil(len(self.valid_pairs) / float(BATCH_SIZE)))
        return STEPS_PER_EPOCH

    def __getitem__(self, index):
        if self.is_validation:
            start = index * BATCH_SIZE
            end = min((index + 1) * BATCH_SIZE, len(self.valid_pairs))
            batch_pairs = self.valid_pairs[start:end]
        else:
            if len(self.valid_pairs) > 0:
                batch_pairs = random.choices(self.valid_pairs, k=BATCH_SIZE)
            else:
                batch_pairs = []

        images, masks = [], []
        
        # Loading and processing files.
        for img_name, mask_name in batch_pairs:
            img = cv2.imread(os.path.join(self.img_dir, img_name))
            if img is None: continue 
            
            # Convert colour frpm OpenCV to Keras
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            # Read the mask (the black/white map showing where the forest is).
            mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            # Resizing and using Nearest to keep sharpness of the mask. 
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            # From (512, 512) to (512, 512, 1) so Keras accepts it.
            mask = np.expand_dims(mask, axis=-1)

            # Data augmentation as described in the paper (rotation, reflection, zooming, and shearing) because of small dataset.
            if self.augment:
                # Horizontal Reflection.
                if np.random.rand() > 0.5:
                    img = np.fliplr(img); mask = np.fliplr(mask)
                # Vertical Reflection.
                if np.random.rand() > 0.5:
                    img = np.flipud(img); mask = np.flipud(mask)
                # Spins the image by 90c k times. 
                if np.random.rand() > 0.5:
                    k = np.random.randint(1, 4)
                    img = np.rot90(img, k); mask = np.rot90(mask, k)
                # Zooming and Shearing.
                if np.random.rand() > 0.3:
                    scale = np.random.uniform(0.8, 1.2)
                    shear = np.random.uniform(-0.1, 0.1)
                    h, w = IMG_SIZE, IMG_SIZE
                    # Combining zoom and shear.
                    M = np.float32([[scale, shear, w*(1-scale)/2], [shear, scale, h*(1-scale)/2]])
                    # Apply the matrix to the Image.
                    # BORDER_REFLECT fills empty space created with mirrored pixels.
                    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    # Apply matrix to the Mask.
                    mask_sq = mask.squeeze()
                    mask_warped = cv2.warpAffine(mask_sq.astype(np.float32), M, (w, h), 
                                                 borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
                    mask = np.expand_dims(mask_warped, axis=-1)

            # Converting pixel values from range 0-255 to range 0.0-1.0. (normalisation)
            images.append(img.astype(np.float32) / 255.0)
            # Ensuring the mask is strictly black (0) or white (1).
            mask = mask.astype(np.float32) / 255.0
            mask[mask > 0.5] = 1.0
            mask[mask <= 0.5] = 0.0
            masks.append(mask)

        return np.array(images), np.array(masks)

# Architecture (John & Zhang, 2022)
# 'skip connection' to only let relevant features through.
def attention_gate(g, s, num_filters):
    # g -> what to look for | s -> raw feature to look at
    # 1x1 convolutions to make 'g' and 's' compatible sizes.
    Wg = Conv2D(num_filters, 1, padding="same")(g)
    Wg = BatchNormalization()(Wg)
    Ws = Conv2D(num_filters, 1, padding="same")(s)
    Ws = BatchNormalization()(Ws)
    # Combining s and g and ReLU only activates region where both signals agree.
    out = Activation("relu")(Wg + Ws)
    # Squashing values between 0 and 1 with sigmoid. (important = 1 | irrelevant = 0)
    out = Conv2D(1, 1, padding="same")(out)
    out = Activation("sigmoid")(out)
    # By multiplying s by out, noise is supressed and important features boosted.
    return out * s

# Standard Double Convolution Block to learn more complex features.
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Upscaling Block.
def decoder_block(input, skip, num_filters):
    # Increasing image size by 2.
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # Calling the attention_gate to not merge noisy background inforamtion (John & Zhang, 2022).
    skip = attention_gate(x, skip, num_filters)
    # Joining the upsampled features with the filtered skip features.
    x = Concatenate()([x, skip])
    # Refining combined features from above function.
    x = conv_block(x, num_filters)
    return x

# Assembling the full model (Fig. 3 - John & Zhang, 2022).
def build_attention_unet(input_shape):
    inputs = Input(input_shape)
    # Encoder - compressing image to find the what - starting with 16 filters instead of standard 64.
    s1 = conv_block(inputs, 16); p1 = MaxPooling2D((2, 2))(s1)
    s2 = conv_block(p1, 32); p2 = MaxPooling2D((2, 2))(s2)
    s3 = conv_block(p2, 64); p3 = MaxPooling2D((2, 2))(s3)
    s4 = conv_block(p3, 128); p4 = MaxPooling2D((2, 2))(s4)
    
    # Bottom of the U - most abstract features.
    b1 = conv_block(p4, 256) 

    # Decoder - expanding image to find the where - with above decoded block for attention gates.
    d1 = decoder_block(b1, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 16)
    
    # Sigmoid activation pushes output to 0 (No Forest) or 1 (Forest).
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    
    model = Model(inputs, outputs, name="Attention_UNet_Paper")
    return model
    
model = build_attention_unet((IMG_SIZE, IMG_SIZE, CHANNELS))

# Choice of training engine from section 2.1.2 (John & Zhang, 2022):
# Using 'Adam' because the paper found it provided better validation accuracy than SGD.
# Learning Rate: 0.0005 is used, which was the specific optimal rate found for Attention U-Net (see Table 2 in paper).
#'BinaryCrossentropy' is selected as the standard loss function for binary (Forest/Non-Forest).
# Tracking pixel-wise 'accuracy' during training loops (though F1/IoU are used for the final strict evaluation).
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss=tf.keras.losses.BinaryCrossentropy(), 
              metrics=['accuracy'])

print(f"Starting Training (Steps: {STEPS_PER_EPOCH})")

# Switch to change fow the data loader behaves for training/testing.
train_gen = AmazonGenerator(TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=True, is_validation=False)
validation_gen = AmazonGenerator(VAL_IMG_DIR, VAL_MASK_DIR, augment=False, is_validation=True)

# Safety guard for reproducibility. 
if len(train_gen.valid_pairs) == 0:
    print("No valid i mage/mask paris found.")
    exit()

# Saves the best model from the run.
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint('Amazon_repro.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
]

# Check if weights exist to avoid retraining. 
if os.path.exists('Amazon_repro.keras'):
    print("Loading existing weights from 'Amazon_repro.keras' - Skipping Training.")
    model.load_weights('Amazon_repro.keras')
    history = None
else:
    # Capturing the scores (Accuracy/Loss) for every epoch to draw graphs later.
    history = model.fit(
        train_gen, 
        epochs=EPOCHS, 
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_gen,
        callbacks=callbacks_list
    )

# PLOT 1: Training Curves (Only if training happened)
if history:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color=COLOR_TRAIN, linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color=COLOR_VAL, linewidth=2)
    plt.title('Amazon Baseline: Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.grid(True, linestyle=':', alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', color=COLOR_TRAIN, linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', color=COLOR_VAL, linewidth=2)
    plt.title('Amazon Baseline: Accuracy Curve')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.savefig("Amazon_Training_Curves.png", dpi=300)
    print("Saved Amazon_Training_Curves.png")

# Evaluation following paper.
print()
print("Calculating Metrics")
# Ensuring we are using the best weights (to avoid full training again).
# Note for user: if you want to see the model training values, delete the Amazon_repro.keras file in the CW2-Poster directory.
model.load_weights('Amazon_repro.keras')

y_true_all = []
y_pred_all = []
per_image_f1 = []
per_image_indices = [] 

# Running through validation set to get metrics and find failure cases
for i in range(len(validation_gen)):
    imgs, masks = validation_gen[i]
    if len(imgs) == 0: continue
    preds = model.predict(imgs, verbose=0)
    preds_bin = np.round(preds)
    
    for j in range(len(masks)):
        # 2.1.3: "Weighted metrics were used as they account for class imbalance" (John & Zhang, 2022)
        f1 = f1_score(masks[j].flatten(), preds_bin[j].flatten(), average='weighted', zero_division=1)
        per_image_f1.append(f1)
        
        # Store index to retrieve image for failure analysis
        per_image_indices.append((i, j)) 
    # Reminder: flattening is to put into 1D list to calculate F1 score (two long lists of numbers compared see if they match)
    y_true_all.extend(masks.flatten())
    y_pred_all.extend(preds_bin.flatten())

# Global weighted F1 to match paper
weighted_f1 = f1_score(y_true_all, y_pred_all, average='weighted')
# IoU only rewards the model when the predicted shape actually overlaps with the true shape: "they account for class imbalance between forest and non-forest pixels".
iou = jaccard_score(y_true_all, y_pred_all, average='binary')
mean_score = np.mean(per_image_f1)
# Checks if model is consistently good.
confidence_interval = stats.t.interval(0.95, len(per_image_f1)-1, loc=mean_score, scale=stats.sem(per_image_f1))

print(f"FINAL RESULTS (Target F1: 0.9550)")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"IoU:         {iou:.4f}")
print(f"95% CI:      {confidence_interval[0]:.4f} - {confidence_interval[1]:.4f}")

# PLOT 2: Confusion Matrix
cm = confusion_matrix(y_true_all, y_pred_all, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Forest', 'Forest'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, values_format='.3f')
plt.title('Amazon Baseline: Pixel-Wise Confusion Matrix')
plt.savefig("Amazon_Confusion_Matrix.png", dpi=300)

# PLOT 3: Box Plot
plt.figure(figsize=(6, 6))
box = plt.boxplot(per_image_f1, patch_artist=True, tick_labels=['Amazon Baseline'])
for patch in box['boxes']:
    patch.set_facecolor(COLOR_VAL)
    patch.set_edgecolor(COLOR_TRAIN)
for median in box['medians']:
    median.set_color(COLOR_TRAIN)
    median.set_linewidth(2)

plt.title('Amazon F1 Score Distribution')
plt.ylabel('Weighted F1 Score')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("Amazon_F1_Boxplot.png", dpi=300)
print("Saved Amazon_F1_Boxplot.png")

# PLOT 4: Best vs Worst
def visualise_performance(generator, model, indices, title_prefix, filename):
    plt.figure(figsize=(12, 4 * len(indices)))
    for plot_idx, (batch_idx, img_idx) in enumerate(indices):
        # Retrieve the specific batch containing the interesting image.
        imgs, masks = generator[batch_idx]
        target_img = imgs[img_idx]
        target_mask = masks[img_idx]
        
        # Run prediction and converts soft probabilities to hard binary decisions.
        pred = model.predict(np.expand_dims(target_img, axis=0), verbose=0)[0]
        pred = np.round(pred)
        
        # Manual IoU Calculation for Title
        intersection = np.logical_and(target_mask.squeeze(), pred.squeeze()).sum()
        # Counting pixels where the Truth or Prediction say it is Forest.
        union = np.logical_or(target_mask.squeeze(), pred.squeeze()).sum()
        # IoU = Overlap / Total Area. 1e-6 prevents a crash if union is 0.
        iou_score = intersection / (union + 1e-6)
        
        # Plots.
        # Raw Satellite Input.
        plt.subplot(len(indices), 3, plot_idx*3 + 1)
        plt.title(f"{title_prefix} Input", color='black', fontweight='bold')
        plt.imshow(target_img.squeeze(), cmap='gray')
        plt.axis('off')

        # Ground Truth.
        plt.subplot(len(indices), 3, plot_idx*3 + 2)
        plt.title("Ground Truth", color='black', fontweight='bold')
        plt.imshow(target_mask.squeeze(), cmap='gray')
        plt.axis('off')

        # AI Prediction.
        plt.subplot(len(indices), 3, plot_idx*3 + 3)
        plt.title(f"Pred (IoU: {iou_score:.2f})", color='black', fontweight='bold')
        plt.imshow(pred.squeeze(), cmap='gray')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    
# Finding best 2 and worst 2 images based on F1 score
sorted_indices = np.argsort(per_image_f1)
worst_indices = [per_image_indices[i] for i in sorted_indices[:2]] # Lowest scores
best_indices = [per_image_indices[i] for i in sorted_indices[-2:]] # Highest scores

visualise_performance(validation_gen, model, best_indices, "Best Case", "Amazon_Best_Cases.png")
visualise_performance(validation_gen, model, worst_indices, "FAILURE CASE", "Amazon_Failure_Cases.png")