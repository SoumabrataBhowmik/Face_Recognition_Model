### **Robust Face Recognition via ResNet-50 Feature Extraction**

## **Facial Verification Under Distortion**

This project implements a highly accurate facial recognition and verification system designed to match distorted or augmented query images against clean reference images. By leveraging a fine-tuned ResNet-50 architecture and Cosine Similarity on 2048-dimensional embeddings, the model achieves near-perfect verification accuracy.

### **Key Features**

**Transfer Learning Strategy:** Utilizes a pre-trained ResNet-50, freezing early layers and fine-tuning only the deep semantic layers (layer4 and fc) for optimal feature extraction.

**Distortion Robustness:** Evaluates model performance by specifically querying distorted/augmented faces against clean reference embeddings.

**Cosine Similarity Matching:** Employs an $L_2$-normalized embedding space to match query identities, bypassing standard softmax classification for final verification.

**Exceptional Performance:** Achieved an outstanding Validation F1-Score of 0.9989 across a highly diverse set of identities.

### **Model Architecture**
The system operates in two phases:Training (Classification) and Inference (Embedding Verification).

**Technical Specifications**

**Backbone:** ResNet-50 (Pretrained on ImageNet).

**Training Head:** Fully connected layer adapted for 877 distinct classes.

**Embedding Extractor:** Strips the final classification layer to output a 2048-dimensional feature vector.

**Normalization:** Embeddings are $L_2$-normalized (p=2, dim=1) to project features onto a unit hypersphere, optimizing them for cosine similarity calculations.

<p align="center">
  <img src="architecture diagram face recognition.png" width="600" title="Face_Recognition_Model Architecture">
</p>

### **Dataset & Preprocessing**

The dataset is structured into distinct person folders, with a specific focus on handling image distortions.

Total Identities (Train): 877 unique classes.

Data Split Structure: Zero identity overlap between the training and validation sets (Overlap: set()).

Image Categories: * Clean Images: Used to generate ground-truth "Reference Embeddings".

Distorted Images: Stored in subfolders and used exclusively as "Query Images" during evaluation.

Augmentations (Training): Resize (224x224), Random Horizontal Flip, Random Rotation (10°), ImageNet Normalization.

**Training Configuration & Performance**

The model was trained using Label Smoothing to prevent overconfidence and an adaptive learning rate scheduler.

### **Training Hyperparameters**

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | AdamW (`lr = 1e-4`) |
| **Loss Function** | CrossEntropyLoss with Label Smoothing (0.1) |
| **Scheduler** | StepLR (`step_size = 5`, `gamma = 0.1`) |
| **Batch Size** | 32 |
| **Epochs** | 10 |

### **Epoch Progression (Classification Loss)**

| Epoch | Train Loss | Train Accuracy |
| :---: | :---: | :---: |
| 1 | 4.9673 | 0.2665 |
| 3 | 1.7365 | 0.9127 |
| 5 | 1.1948 | 0.9957 |
| 10 | 1.1041 | 0.9999 |

### **Final Verification Metrics**

Instead of evaluating the raw classification output, the final evaluation tests the model's true verification power: extracting the embedding of a distorted image and matching it to the closest clean reference embedding via Cosine Similarity.

| Metric | Training Set (Queries) | Validation Set (Queries) |
| :--- | :---: | :---: |
| **Accuracy** | 1.0000 | 0.9990 |
| **Precision** | 1.0000 | 0.9989 |
| **Recall** | 1.0000 | 0.9991 |
| **F1-Score** | 1.0000 | 0.9989 |

**Installation & Usage**

Clone the Repository:

Bash

git clone https://github.com/SoumabrataBhowmik/Face_Recognition_Model.git

cd Face_Recognition_Model

Dependencies:

Ensure you have the required libraries installed:

Bash

pip install torch torchvision pillow scikit-learn numpy

Run the Implementation:

Open the provided Jupyter Notebook (face_recognition.ipynb file) to view the data loading, training loop, and evaluation logic.

The trained weights are saved in face_recognition_model.pt. To run inference, load this state dictionary into a ResNet-50 model.
