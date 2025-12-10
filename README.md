# Attention U-Net Adaptation: From Amazon Deforestation to Glacial Lake Monitoring

## Project Overview
This repository contains the replication of an Attention U-Net model originally designed for deforestation detection in the Amazon and its adaptation for monitoring Glacial Lake Outburst Flood (GLOF) risks in the Nepal Himalayas (Tsho Rolpa). This work addresses **SDG 13 (Climate Action)** and **SDG 11 (Sustainable Cities and Communities)** by transferring deep learning architectures to a new, high-stakes environmental context.

## Quick Start & Grading Information
For a streamlined review of the results, please open:
* **`Untitled.ipynb`**: This notebook contains sequential cells executing the key scripts. It displays the final model outputs, training curves, and evaluation metrics for both the Amazon replication and Nepal adaptation without requiring a full retrain.

### Reproducing the Training
The repository includes pre-trained weights to facilitate quick evaluation.
* **Weights Location:**
    * `25229117-Kedziora-CW2-Poster/my_full_export/CW2-Poster/Nepal_repro.keras`
    * `25229117-Kedziora-CW2-Poster/my_full_export/CW2-Poster/Amazon_repro.keras`

**To Retrain from Scratch:**
If you wish to reproduce the full training pipeline for grading, please move the `.keras` files mentioned above to the `25229117-Kedziora-CW2-Poster/my_full_export/CW2-Poster/keras_holder_folder`.

> **Note:** Training is computationally intensive. It may take approximately **45–60 minutes** on a standard T4 GPU due to the high number of steps per epoch configured for robust convergence.

## Repository Structure
The core implementation files are located in `25229117-Kedziora-CW2-Poster/my_full_export/CW2-Poster/code/`:

* **`../code/replicate_amazon.py`**
    * **Purpose:** Replicates the baseline Attention U-Net methodology on the Amazon RGB dataset.
    * **Key Action:** Implements the Attention Gate mechanism and U-Net architecture exactly as described in the original paper.

* **`../code/nepal_eda.py`**
    * **Purpose:** Performs Exploratory Data Analysis (EDA) on the Himalayan dataset.
    * **Key Action:** Analyses pixel intensity distributions to identify domain shifts between the high-contrast "false-colour" training data and the natural "true-colour" target data.

* **`../code/adapt_nepal.py`**
    * **Purpose:** Adapts the model for the Himalayan context.
    * **Key Adaptation:** Modifies the architecture (mixed precision, grayscale input, CLAHE preprocessing) to handle the specific challenges of the Tsho Rolpa region.

* **`../code/monitor_tsho_rolpa.py`**
    * **Purpose:** **(Bonus Work)** Executes a longitudinal analysis of the Tsho Rolpa glacial lake.
    * **Key Action:** Tracks the lake's surface area evolution over time using the adapted model to quantify expansion risks.

## Methodology Notes
* **Baseline Replication:** The original baseline code was cloned into `CW2-Poster/original_paper_code`. Consistent with the methodology described in the paper, **5 images were manually moved** from the training set to the testing/validation set to create a strictly unseen evaluation split.
* **Poster:** The academic poster summarising this work is included in the submission `.zip` file (located in a separate root folder not visible in the main export).

## Future Improvements
To enhance the robustness of the Glacial Lake Outburst Flood (GLOF) warning system, future work will focus on:
1.  **3D Volumetric Analysis:** Integrating Lidar and Ground Penetrating Radar (GPR) data to measure water volume and dam stability, rather than relying solely on 2D surface area.
2.  **Multimodal Fusion:** Combining optical satellite imagery with SAR (Synthetic Aperture Radar) to monitor the lake through cloud cover, which is frequent in the Himalayas.

## Open Source
This project will be open-sourced shortly on my GitHub to contribute to the community of AI for Earth Observation.

## References

### Academic Paper Replicated
[1] John, D., & Zhang, C. (2022). An attention-based U-Net for detecting deforestation within satellite sensor imagery. *International Journal of Applied Earth Observation and Geoinformation*, 107, 102685. https://doi.org/10.1016/j.jag.2022.102685 [Accessed: 10 December 2025].

### Datasets
* **Amazon Dataset:** Bragagnolo, L., da Silva, R. V., & Grzybowski, J. M. V. (2019). Amazon Rainforest dataset for semantic segmentation. https://doi.org/10.5281/zenodo.3233081
* **Himalayas Dataset:** Shrestha, A. (n.d.). Glacial Lake Dataset. Kaggle. https://www.kaggle.com/datasets/aatishshresthaa/glacial-lake-dataset [Accessed: 10 December 2025].

### Original Code Repository
* John, D. (2022). *attention-mechanism-unet* [Source Code]. GitHub. https://github.com/davej23/attention-mechanism-unet