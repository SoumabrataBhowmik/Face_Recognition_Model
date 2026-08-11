# Robust Open-Set Face Recognition Under Severe Visual Distortions
## Facial Verification via ResNet-50 Proxy-Task Training

This project implements a two-phase, embedding-based face verification system designed to match severely distorted query images against clean reference images, generalizing to identities **never seen during training** (a strict open-set protocol). By leveraging proxy-task classification training followed by an $L_2$-normalized 2048-dimensional cosine-similarity embedding space, the model achieves near-perfect open-set verification accuracy — substantially outperforming three off-the-shelf pretrained face embedding baselines evaluated under the identical protocol.

### Key Features

- **Two-Phase Proxy-Task Strategy:** Phase 1 trains a ResNet-50 as a closed-set 877-way identity classifier (with aggressive layer freezing) purely as an intermediate objective; Phase 2 discards the classification head and repurposes the backbone as an open-set embedding extractor.
- **Open-Set Evaluation:** Training and validation identities are strictly disjoint ($ID_{train} \cap ID_{val} = \emptyset$, verified — zero identity overlap, zero duplicate or near-duplicate images across splits).
- **Distortion Robustness:** Evaluated across seven distortion categories (blur, fog, low-light, noise, rain, resizing, sunlight overexposure), matching distorted queries against clean reference embeddings.
- **Benchmarked Against Real Baselines:** Compared under an identical protocol against an ImageNet-pretrained ResNet-50 (no fine-tuning), InceptionResnetV1 (VGGFace2), and ArcFace (buffalo_l, InsightFace project) — not just an internal ablation.
- **Cosine Similarity Matching:** Multi-template gallery-query matching in an $L_2$-normalized embedding space, bypassing softmax classification entirely at inference time.

---

## Model Architecture

The system operates in two phases: **Phase 1 (Closed-Set Classification)** and **Phase 2 (Open-Set Embedding Verification)**.

<p align="center">
  <img src="architecture_diagram_face_recognition.png" width="600" title="Face_Recognition_Model Architecture">
</p>

### Technical Specifications

| Component | Detail |
| :--- | :--- |
| **Backbone** | ResNet-50 (ImageNet-pretrained, `torchvision.models.resnet50(pretrained=True)`) |
| **Freezing Strategy** | Layers 1–3, initial conv, and batch norm frozen; only Layer 4 + FC trainable |
| **Trainable / Total Params (Phase 1)** | 16.76M / 25.31M |
| **Embedding-Extractor Params (Phase 2, inference)** | 23.51M (FC head removed after Global Average Pooling) |
| **Training Head** | Fully connected layer over 877 identity classes |
| **Embedding Dimension** | 2048-d, $L_2$-normalized ($p=2$, $\dim=1$) onto the unit hypersphere |
| **Matching Rule** | Multi-template cosine similarity, $S_i(q) = \max_j S(q, r_{i,j})$, forced top-1 (argmax) — **no rejection threshold is implemented in this inference pipeline** |

---

## Dataset & Preprocessing

Dataset: **FACECOM** (Face Attributes in Challenging Environments), structured by identity into person folders with a nested `distortion/` subfolder per identity.

| Split | Identities | Clean Reference Images | Distorted Query Images |
| :--- | :---: | :---: | :---: |
| Train | 877 | 1,926 | 13,482 |
| Validation | 250 | 422 | 2,954 |
| **Total** | **1,127** | **2,348** | **16,436** |

**Total dataset size: 18,784 images.** Identity overlap between train/validation splits: **verified zero** (via MD5 hash and embedding-similarity checks — no exact duplicates, no near-duplicate images across the split boundary).

Distorted queries span **seven categories**: blur, fog, low-light, noise, rain, resizing, and sunlight-induced overexposure — present in both splits (1,926 train / 422 validation images per category).

**Augmentations (training only):** Resize(224×224), RandomHorizontalFlip, RandomRotation(10°), ImageNet normalization. `ColorJitter` is deliberately **not** used, to preserve consistent appearance between clean templates and distorted queries for reliable cosine-similarity matching.

---

## Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | AdamW (`lr = 1e-4`) |
| **Loss Function** | CrossEntropyLoss with Label Smoothing ($\epsilon = 0.1$) |
| **Scheduler** | StepLR (`step_size = 5`, `gamma = 0.1`) |
| **Batch Size** | 32 |
| **Epochs** | 10 |

### Epoch Progression (Phase 1 Classification Loss)

| Epoch | Train Loss | Train Accuracy |
| :---: | :---: | :---: |
| 1 | 4.9673 | 0.2665 |
| 3 | 1.7365 | 0.9127 |
| 5 | 1.1948 | 0.9957 |
| 10 | 1.1041 | 0.9999 |

---

## Results: Benchmark Comparison

The final evaluation tests true **open-set verification power** — extracting the embedding of a distorted query and matching it to the closest clean reference embedding via cosine similarity — not raw classification accuracy. This model is benchmarked against three pretrained alternatives under an identical gallery-query protocol on the same validation split.

### Model Comparison (Validation, Open-Set)

| Model | Inference Params | Accuracy | Precision | Recall | F1-Score | AUC | EER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ImageNet Baseline (no fine-tuning) | 23.51M | 70.58% | 85.55% | 72.98% | 75.53% | 0.8422 | 0.2468 |
| InceptionResnetV1 (VGGFace2) | 27.91M | 79.05% | 89.52% | 78.45% | 81.70% | 0.9410 | 0.1228 |
| ArcFace (buffalo_l) | 43.59M | 79.96% | 91.07% | 79.55% | 83.10% | 0.9594 | 0.0944 |
| **This Model (Proposed)** | **23.51M** | **99.90%** | **99.89%** | **99.91%** | **99.89%** | **1.0000** | **0.0027** |

Notably, this model's inference-time footprint (23.51M) is **identical** to the untrained ImageNet baseline and smaller than both pretrained face-specific alternatives — the performance gap comes entirely from the proxy-task training strategy, not additional model capacity.

### Training vs. Validation (This Model)

| Metric | Training Set | Validation Set |
| :--- | :---: | :---: |
| **Accuracy** | 1.0000 | 0.9990 |
| **Precision** | 1.0000 | 0.9989 |
| **Recall** | 1.0000 | 0.9991 |
| **F1-Score** | 1.0000 | 0.9989 |

### Accuracy by Distortion Type (Validation, n=422 per category)

| Distortion | ImageNet Baseline | InceptionResnetV1 | ArcFace | This Model |
| :--- | :---: | :---: | :---: | :---: |
| Rainy | 25.12% | 78.67% | 86.73% | 100.00% |
| Noisy | 25.83% | 84.12% | 76.30% | 100.00% |
| **Sunny** | 56.16% | **1.42%** | **2.84%** | 99.29% |
| Foggy | 91.47% | 91.94% | 95.02% | 100.00% |
| Low-light | 95.97% | 97.39% | 99.76% | 100.00% |
| Blurred | 99.53% | 99.76% | 99.05% | 100.00% |
| Resized | 100.00% | 100.00% | 100.00% | 100.00% |

**Notable finding:** under the "sunny" (overexposure) distortion, both pretrained face-specific models collapse to near-chance accuracy — likely because this distortion severely occludes the internal facial landmarks (eyes, nose, mouth) that VGGFace2/ArcFace-style training data typically exposes. This model retains 99.29% accuracy under the same condition, consistent with Phase 1 training including pre-distorted samples across all seven categories.

---

## Open-Set Rejection: Threshold Analysis (Not Yet Deployed)

> **Important:** the deployed inference pipeline in this repository performs a **forced nearest-neighbor match** — every query is assigned to the highest-similarity gallery identity, regardless of how low that similarity is. **No rejection threshold is implemented in the current codebase.**

A **post-hoc analysis** was run on the model's existing top-1 similarity scores to evaluate whether a threshold-based rejection rule (`accept only if max similarity ≥ τ, else label "unknown"`) would be viable if implemented:

- Accuracy on accepted queries remains **≥0.999** across nearly the full threshold range tested.
- Rejection rate stays **below 5%** for τ ≤ 0.90, rising sharply only as τ → 1.0.
- The EER-optimal threshold (τ ≈ 0.797) falls within this low-rejection region — a plausible operating point *if* this mechanism were integrated into deployment.

**This is analysis, not a shipped feature.** Integrating actual threshold-based rejection into the inference pipeline, and validating it against a dedicated set of imposter identities entirely absent from the gallery, remains future work.

---

## Installation & Usage

### Dependencies

Core dependencies:
```bash
pip install torch torchvision pillow scikit-learn numpy
```

For reproducing the baseline comparison (InceptionResnetV1 and ArcFace):
```bash
pip install facenet-pytorch --break-system-packages -q
pip install insightface onnxruntime opencv-python-headless --break-system-packages -q
```

> Note: installing the above may pull in NumPy 2.x / Pillow 11.x, which can break an existing PyTorch/torchvision install compiled against older ABI versions. If you hit `ImportError` or `"Numpy is not available"` errors, pin back with:
> ```bash
> pip install "numpy<2" "Pillow<11" --break-system-packages -q
> ```
> and restart your kernel/session.

### Running the Implementation

Open `face_recognition.ipynb` to view the data loading, Phase 1/Phase 2 training loop, and evaluation logic. Trained weights are saved as `face_recognition_model.pt`. To run inference, load this state dictionary into a ResNet-50 model with the final FC layer set to 877 output classes, then strip the FC layer post-load to use the model as an embedding extractor (see Phase 2 in the notebook).

---

## Limitations

- **Alignment dependency:** performance depends on accurate face localization/cropping prior to embedding extraction; severe occlusions that break upstream face detection will break the pipeline.
- **No rejection mechanism deployed:** see the Threshold Analysis section above.
- **Single-benchmark evaluation:** all results are obtained on FACECOM; generalization to other face recognition benchmarks or real-world surveillance imagery has not been empirically tested.

## Citation

If you use this work, please cite the accompanying paper (details to be added upon publication).

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
