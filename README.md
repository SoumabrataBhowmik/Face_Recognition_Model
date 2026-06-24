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


## System Architecture

The face recognition system is divided into two primary phases: a model training pipeline utilizing a fine-tuned ResNet-50, and an inference pipeline that extracts L2-normalized 2048-dimensional feature embeddings for cosine similarity matching.

```mermaid
flowchart TD
    %% Node Class Definitions for Aesthetics
    classDef inputNode fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1,rx:8px,ry:8px
    classDef processNode fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20,rx:8px,ry:8px
    classDef modelNode fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#b71c1c,rx:8px,ry:8px
    classDef embedNode fill:#fff8e1,stroke:#fbc02d,stroke-width:2px,color:#f57f17
    classDef matchNode fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef outputNode fill:#e0f7fa,stroke:#0097a7,stroke-width:2px,color:#006064,rx:8px,ry:8px
    
    %% Global Edge styling
    linkStyle default stroke:#78909c,stroke-width:2px

    subgraph TrainingPipeline ["⚙️ Model Training Pipeline"]
        direction TB
        
        %% Inputs
        TrainData[/"📁 Input Data (Train Images)<br/><b>person_00X/</b><br/>├── img1.png, img2.png<br/>└── <b>distortion/</b><br/>&nbsp;&nbsp;&nbsp;&nbsp;└── distorted_img.png"/]:::inputNode
            
        %% Processing
        TrainPrep("🔄 Train Preprocessing<br/>• Resize (224x224)<br/>• RandomHorizontalFlip<br/>• RandomRotation (10°)<br/>• ToTensor<br/>• Normalize"):::processNode
        
        %% Model
        TrainModel{{"🧠 Model Training (ResNet-50)<br/>• Pretrained base<br/>• Freeze all except layer4 & fc<br/>• Modify fc Num Classes<br/>• Loss: CrossEntropy (Label Smoothing 0.1)<br/>• Optimizer: AdamW<br/>• Scheduler: StepLR"}}:::modelNode
        
        %% Output file
        ModelWeights[("💾 Output Weights<br/>face_recognition_model.pt")]:::modelNode
        
        %% Flow
        TrainData ==> TrainPrep ==> TrainModel ==> ModelWeights
    end

    subgraph InferencePipeline ["🔍 Validation & Matching Pipeline"]
        direction TB
        
        %% Inputs
        ValData[/"📁 Input Data (Val Images)<br/>Clean & Distorted Data"/]:::inputNode
        
        %% Processing
        ValPrep("🔄 Val Preprocessing<br/>• Resize (224x224)<br/>• ToTensor<br/>• Normalize"):::processNode
        
        %% Model Loading
        LoadModel("📥 Loaded ResNet-50<br/>(Pre-trained Weights)"):::modelNode
        
        %% Extractor
        Extractor{{"🛠️ FaceEmbeddingExtractor<br/>• ResNet50 up to avgpool<br/>• (Drop final FC layer)<br/>• Output: 2048-dim vector<br/>• Normalize with L2 norm"}}:::modelNode
        
        %% Embeddings
        CleanEmb[/"✨ Clean Embeddings<br/>(Reference / Person)"/]:::embedNode
        DistEmb[/"🌀 Distorted Embeddings<br/>(Query)"/]:::embedNode
        
        %% Matching
        Match{"⚖️ Cosine Similarity<br/>Compare query to all references<br/>Pick Highest Similarity"}:::matchNode
        
        %% Final Evaluation
        Metrics("🎯 Final Prediction<br/>Most Similar Person<br/>(Evaluate: Acc, Prec, Rec, F1)"):::outputNode
        
        %% Flow
        ValData ==> ValPrep ==> LoadModel ==> Extractor
        Extractor -->|Extract| CleanEmb
        Extractor -->|Extract| DistEmb
        
        CleanEmb <==>|Compare| Match
        DistEmb ==>|Query| Match
        Match ==> Metrics
    end

    %% State Dict Transfer
    ModelWeights -.->|"torch.load()"| LoadModel
