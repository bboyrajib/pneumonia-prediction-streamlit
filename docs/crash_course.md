# Crash Course: Understanding Your Project End-to-End
## Automated Pneumonia and COVID-19 Detection from Chest X-Ray Images

> This document explains every concept you used — what it is, why you used it, and exactly how your code implements it. Read this once and you will be able to answer any question about your own work.

---

## 1. The Big Picture

Your pipeline has 10 stages in sequence:

```
Raw X-ray images
      ↓
Label unification + deduplication
      ↓
Stratified train/val/test split
      ↓
Data augmentation + normalisation → DataLoader
      ↓
Focal Loss + class weights (handle imbalance)
      ↓
Two-phase transfer learning (× 3 models)
      ↓
AUC-weighted ensemble (soft voting)
      ↓
MC Dropout uncertainty → referral flag
      ↓
Grad-CAM saliency maps
      ↓
Streamlit app
```

Each stage is explained below.

---

## 2. Dataset: Why Two Sources and How You Merged Them

### What the datasets are

- **Kermany dataset**: 5,863 chest X-rays. Two classes — Normal and Pneumonia (mostly bacterial).
- **COVID-19 Radiography Database**: 21,165 images. Four classes — COVID, Viral Pneumonia, Lung Opacity, Normal.

Neither dataset alone gives you all three classes you need. So you merged them.

### How you mapped labels

| Source class | Your class |
|---|---|
| Kermany Normal | NORMAL |
| COVID-19 DB Normal | NORMAL |
| Kermany Pneumonia | PNEUMONIA |
| COVID-19 DB Viral Pneumonia | PNEUMONIA |
| COVID-19 DB Lung Opacity | PNEUMONIA |
| COVID-19 DB COVID | COVID |

**Why lump Viral Pneumonia and Lung Opacity into PNEUMONIA?**
Clinically, both cause similar radiological patterns (consolidation, infiltrates) and the clinical action is similar — treat as pneumonia. Keeping them separate would fragment the PNEUMONIA class unnecessarily and confuse the model.

### Deduplication

After merging, the same image might appear in both datasets. You used **hash-based deduplication**: compute a pixel hash (MD5 or similar) of each image and discard exact duplicates. This prevents data leakage — the same image appearing in both train and test.

**Result: 27,005 unique images** — Normal: 11,767 | Pneumonia: 11,622 | COVID: 3,616.

### Why COVID is only 13.4%

The COVID-19 dataset has far fewer confirmed COVID scans than Normal/Pneumonia. This is called **class imbalance** — a model trained naively will learn to mostly predict Normal/Pneumonia because that's what it sees 86% of the time. You fix this with Focal Loss + class weights (Section 5).

---

## 3. Train/Val/Test Split

### What stratified splitting means

A simple random split might, by chance, put most COVID images in training and very few in test. **Stratified split** preserves the class ratio across all three sets.

```
Total: 27,005 images | ~43.6% Normal | ~43.0% Pneumonia | ~13.4% COVID

Train (80%): 21,604 | Normal: 9,413 | Pneumonia: 9,298 | COVID: 2,893
Val  (10%):  2,700  | Normal: 1,177 | Pneumonia: 1,162 | COVID:   361
Test (10%):  2,701  | Normal: 1,177 | Pneumonia: 1,162 | COVID:   362
```

You used `sklearn.model_selection.train_test_split` with `stratify=labels`.

### Why a separate validation set?

- **Train set**: model learns from this.
- **Val set**: you monitor this during training to detect overfitting and trigger early stopping. The model never learns from val — it only *sees* it for evaluation.
- **Test set**: touched exactly once, at the very end. This is your honest estimate of real-world performance. If you used test set to tune hyperparameters, the number would be optimistically biased.

---

## 4. Preprocessing and Augmentation

### Resize to 224×224

All three pretrained models (DenseNet121, ResNet50, EfficientNetB0) were trained on ImageNet with 224×224 input. You must match this.

### ImageNet normalisation

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

These are the pixel mean and standard deviation of the ImageNet dataset (per RGB channel). Subtracting the mean and dividing by std puts your input in the same distribution the pretrained weights were trained on. If you skip this, transfer learning works much worse.

**Conceptually:** the pretrained model's filters were tuned expecting inputs in a certain range. Normalising your X-rays into that range means the pretrained features activate correctly from the start.

### Why X-rays normalised with colour image stats?

X-rays are grayscale, but they're stored as 3-channel images (R=G=B). The ImageNet stats still work because they represent the expected activation range, not colour information specifically.

### Augmentation (training only)

| Transform | Why |
|---|---|
| Random horizontal flip | A lung radiograph is diagnostically symmetric — flipping is valid |
| ±15° rotation | Patients aren't always perfectly upright in the scanner |
| Random affine translation | Slight positional shifts; teaches position invariance |

**Key rule:** augmentation is applied **only during training**, never at val or test time. You want consistent, reproducible evaluation — not randomly transformed test images.

### torchvision ImageFolder

You structured your processed dataset as:
```
data/processed/train/NORMAL/
data/processed/train/PNEUMONIA/
data/processed/train/COVID/
data/processed/val/...
data/processed/test/...
```

`torchvision.datasets.ImageFolder` automatically reads the folder names as class labels and assigns integer indices. Your mapping was `{'COVID': 0, 'NORMAL': 1, 'PNEUMONIA': 2}` (alphabetical).

---

## 5. Handling Class Imbalance: Focal Loss + Class Weights

This is one of the most important parts of your project. Without it, COVID recall would be terrible.

### Inverse-frequency class weights

```python
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=[0,1,2], y=labels)
# Result: COVID=2.489, NORMAL=0.765, PNEUMONIA=0.775
```

**How it works:** `w_c = N / (C × N_c)` where N=total images, C=3 classes, N_c=count of class c.
- COVID has fewer images → higher weight → each COVID sample contributes 2.5× more to the loss.
- This makes the model pay more attention to COVID mistakes.

### Focal Loss

Standard cross-entropy loss:
```
L_CE = -log(p_t)
```
where p_t is the predicted probability for the correct class.

**Problem with cross-entropy on imbalanced data:** easy examples (confidently correct predictions on common classes) still contribute a lot to the total loss, dominating training. Hard examples (rare COVID images, ambiguous cases) get drowned out.

**Focal Loss solution:**
```
L_FL = -α_t × (1 - p_t)^γ × log(p_t)
```

- `(1 - p_t)^γ` is the **modulating factor**. When the model is confident (p_t ≈ 1), this factor ≈ 0 — easy examples are down-weighted. When the model is wrong (p_t ≈ 0), this factor ≈ 1 — hard examples keep their full weight.
- `γ = 2` in your implementation (standard value from the original paper).
- `α = 0.25` is a class-level weight balancing factor.

**Combined effect:** Focal Loss + class weights together ensure the model focuses on hard and rare examples simultaneously.

---

## 6. Transfer Learning: What It Is and How You Did It

### Why transfer learning?

Training a deep CNN from scratch on 27,000 images would likely overfit. ImageNet has 1.2 million images across 1,000 classes. Models pretrained on ImageNet have learned general visual features: edges, textures, shapes, gradients. These generalise well to medical images.

### The three backbones

**DenseNet121**
- Each layer connects to *every subsequent layer* (dense connections).
- Promotes feature reuse — earlier low-level features (edges, textures) remain accessible throughout the network.
- 7.2M parameters. Grad-CAM layer: `features.denseblock4` (the last dense block).

**ResNet50**
- Uses **residual connections**: output = F(x) + x. The network learns the *residual* (what to add to the input) rather than the full mapping.
- Solves the vanishing gradient problem for deep networks.
- 24.0M parameters. Grad-CAM layer: `layer4`.

**EfficientNetB0**
- Scales network depth, width, and input resolution together using a **compound coefficient** found by neural architecture search.
- Most parameter-efficient of the three (4.3M params, best performance).
- Grad-CAM layer: `features[7]` (last MBConv block).

### The custom classification head

You replaced each model's original ImageNet classifier with:
```
BN1d → Dropout(0.4) → Linear(f, 256) → ReLU → BN1d → Dropout(0.4) → Linear(256, 3)
```

- **BatchNorm1d**: normalises the backbone feature vector before the head. Stabilises training, acts as regularisation.
- **Dropout(0.4)**: randomly zeros 40% of neurons during training. Forces the network not to rely on any single feature. Also enables MC Dropout at inference (Section 8).
- **Linear layers**: the actual classification — maps from backbone features (f-dimensional) down to 3 classes.
- `f = 1024` for DenseNet121, `f = 2048` for ResNet50, `f = 1280` for EfficientNetB0.

### Two-phase training strategy

**Phase 1 — Frozen backbone (5 epochs, lr = 1e-3):**
All backbone weights are frozen (`requires_grad = False`). Only the classification head is updated.

*Why:* The head starts with random weights. If you unfreeze the backbone immediately, the large random gradients from the head will corrupt the pretrained backbone weights in the first few updates. Phase 1 brings the head to a reasonable state first.

*Result after Phase 1:* DenseNet 79%, ResNet 80%, EfficientNet 87%.

**Phase 2 — Fine-tuning (up to 20 epochs, lr = 1e-5):**
Unfreeze the last convolutional block + the head. Use a 100× smaller learning rate.

*Why unfreeze only the last block?* The early layers learn general features (edges, textures) that transfer well as-is. The last block learns task-specific features — for ImageNet these are things like "dog ears"; for your task you need "lung infiltrates". Fine-tuning only the last block adapts these high-level features to chest X-rays without disturbing the early useful features.

*Why lr = 1e-5?* The pretrained weights are already in a good region of the loss landscape. A large learning rate would destroy them. A small lr makes tiny adjustments.

### Callbacks

**EarlyStopping (patience=5):** If validation loss doesn't improve for 5 consecutive epochs, stop training. Prevents overfitting to the training set. Triggered for ResNet at Phase 1 epoch 4 and Phase 2 epoch 16; EfficientNet at Phase 2 epoch 7.

**ReduceLROnPlateau (factor=0.5, patience=3):** If val loss plateaus for 3 epochs, halve the learning rate. Helps escape local minima.

**ModelCheckpoint:** Saves the model weights whenever val loss improves. At the end of training, you load the *best* checkpoint, not the last epoch.

### Mixed precision training

```python
from torch.cuda.amp import GradScaler, autocast
```

- **float16** for forward pass (faster computation, less GPU memory).
- **float32** for gradient accumulation and weight updates (numerical stability).
- `GradScaler` scales the loss to prevent float16 underflow during backpropagation.

---

## 7. Ensemble: AUC-Weighted Soft Voting

### What soft voting is

Each model outputs a probability vector `[p_Normal, p_Pneumonia, p_COVID]` (3 numbers summing to 1, from the final softmax). Hard voting would take each model's argmax prediction and vote. **Soft voting** averages the probabilities directly:

```python
ensemble_prob = Σ (w_m × p_m)
```

Soft voting is better because it preserves confidence information. A model that is 99% confident on Normal contributes more than one that is 51% confident.

### AUC-based weights

```python
weights = {
    'DenseNet121':     0.3315,   # val macro AUC = 0.9807
    'ResNet50':        0.3336,   # val macro AUC = 0.9871
    'EfficientNetB0':  0.3348,   # val macro AUC = 0.9906
}
```

Weights are proportional to each model's validation macro ROC-AUC, normalised to sum to 1. EfficientNet gets the highest weight because it was the best individual model.

**Why AUC and not accuracy?** AUC is a threshold-independent metric — it measures how well the model ranks classes across all confidence thresholds, not just at one operating point. It's more informative for soft voting than accuracy.

### Why does the ensemble beat every individual model?

The three models are **architecturally diverse** — they make different mistakes. When one model is wrong, the other two are often right, and their combined probability signal overrides the wrong model. This is the core insight behind ensemble methods.

---

## 8. MC Dropout: Uncertainty Quantification

### What uncertainty quantification is

A standard model gives you a label and a confidence score. But `p = 0.7 for COVID` could mean "definitely COVID, just some inherent ambiguity" (aleatoric) or "I've never seen a case like this, I'm guessing" (epistemic). These are very different clinically.

### How MC Dropout works

**Standard inference:** dropout is turned OFF (model is in eval mode). Outputs are deterministic.

**MC Dropout inference:**
```python
model.train()   # keeps dropout ACTIVE
probs = []
for _ in range(50):
    with torch.no_grad():
        prob = softmax(model(image))
    probs.append(prob)
```

With dropout active, each forward pass randomly zeros different neurons → different outputs. After 50 passes you have a distribution of predictions for the same image.

### What you compute from 50 passes

**Mean probabilities:** `p̄ = mean(probs)` — the final prediction.

**Aleatoric uncertainty (predictive entropy):**
```
H = -Σ p̄_c × log(p̄_c)
```
High entropy = the mean prediction is spread across classes = inherent ambiguity in the image.

**Epistemic uncertainty:** mean standard deviation across the 50 pass predictions.
```
σ = mean(std(probs, axis=0))
```
High σ = the model gives very different answers on different passes = the model is uncertain/unfamiliar with this image.

**Combined score:**
```
combined = 0.6 × normalised_entropy + 0.4 × normalised_σ
```

**Referral decision:** if `combined > 0.35` → flag for radiologist.

### Why this threshold?

0.35 was tuned on the validation set to balance the trade-off: too low → too many referrals (annoying for radiologists); too high → confident wrong predictions slip through (dangerous). At 0.35:
- 44.8% of test cases referred (1,211 images).
- Remaining 55.2% achieve 99.40% accuracy.

---

## 9. Grad-CAM: Visual Explainability

### What Grad-CAM does

Given a class prediction, Grad-CAM produces a heat-map showing *which spatial regions of the image* most influenced that prediction. For a COVID prediction, it should light up the areas with ground-glass opacity.

### How it works (the maths, simply)

1. During the forward pass, **register a hook** to capture the feature maps `A^k` at the last convolutional layer (a 3D tensor: channels × height × width).

2. Compute the class score `y^c` (the logit for the predicted class, before softmax).

3. **Backpropagate** `y^c` to the feature maps to get gradients `∂y^c / ∂A^k`.

4. For each channel k, **global average pool** the gradients:
```
α_k^c = (1/Z) × Σ_{i,j} (∂y^c / ∂A^k_{ij})
```
This gives a scalar "importance weight" for each channel.

5. Take the **weighted sum** of feature maps and apply ReLU:
```
L^c = ReLU(Σ_k α_k^c × A^k)
```
ReLU keeps only features that *positively* activate the class (we don't care about what suppresses the class).

6. **Upsample** the resulting low-resolution map (e.g., 7×7) back to 224×224 using bilinear interpolation and overlay on the original image.

### Your implementation

You used PyTorch hooks:
```python
# Forward hook: captures feature maps during forward pass
handle_fwd = layer.register_forward_hook(save_activation)

# Backward hook: captures gradients during backward pass
handle_bwd = layer.register_full_backward_hook(save_gradient)
```

Target layers per model:
- DenseNet121: `features.denseblock4` — output of the last dense block
- ResNet50: `layer4` — output of the last residual group
- EfficientNetB0: `features[7]` — output of the last MBConv block

---

## 10. Evaluation Metrics — What Each One Means

### Accuracy
```
Accuracy = correct predictions / total predictions
```
Simple but misleading with class imbalance. A model predicting "Normal" always would get 43.6% accuracy on your test set.

### Macro ROC-AUC

**ROC curve:** for each class, plot True Positive Rate vs False Positive Rate as you sweep the decision threshold. Area under this curve = AUC.

**AUC = 1.0** means perfect separation. **AUC = 0.5** means random guessing.

**Macro:** compute AUC separately for each class (one-vs-rest) and average. Treats all classes equally regardless of frequency — crucial for your imbalanced COVID class.

**Your result: 0.9908** — extremely good for a 3-class medical imaging task.

### Macro F1

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Precision:** of all images predicted as COVID, what fraction were actually COVID?
- **Recall:** of all actual COVID images, what fraction did the model catch?
- F1 is their harmonic mean — punishes models that sacrifice one for the other.
- **Macro:** average F1 across all 3 classes equally.

**Your result: 0.9396 macro F1, 0.9465 COVID F1.**

### Confusion Matrix

A 3×3 table where row = true class, column = predicted class. Diagonal = correct. Off-diagonal = errors.

The most clinically dangerous error: COVID predicted as PNEUMONIA (patient gets wrong treatment). The ensemble minimises this specific cell.

### Calibration (Reliability Diagram)

**What it measures:** does `p = 0.8 for COVID` actually mean the model is right 80% of the time?

Plot: x-axis = predicted probability, y-axis = actual fraction correct. A perfectly calibrated model lies on the diagonal. Your ensemble is close to the diagonal → probabilities are trustworthy → the MC Dropout threshold is meaningful.

---

## 11. The Streamlit App — How It Works

```python
@st.cache_resource
def load_models():
    # loads all 3 .pt files once at startup, caches in memory
```

`@st.cache_resource` ensures models are loaded from disk only once — Streamlit reruns the entire script on every user interaction, so without caching you'd reload 35M+ parameters on every button click.

**Single image inference flow:**
1. User uploads image → PIL opens it → `torchvision.transforms` resizes + normalises → tensor
2. Each model: `model.eval()` + `torch.no_grad()` + `autocast()` → softmax probabilities
3. Ensemble: weighted sum of 3 probability vectors → final prediction + bar chart
4. MC Dropout: `model.train()` + 50 forward passes → uncertainty score → referral flag
5. Grad-CAM (if toggled): register hooks → forward + backward → generate heat-map overlay

---

## 12. Key Numbers to Remember

| Metric | Value |
|---|---|
| Total images | 27,005 |
| Test set size | 2,701 |
| Ensemble accuracy | 93.74% |
| Ensemble macro AUC | 0.9908 |
| Ensemble macro F1 | 0.9396 |
| COVID F1 (hardest class) | 0.9465 |
| High-confidence accuracy | 99.40% |
| Referral rate | 44.8% |
| MC Dropout passes | 50 |
| Uncertainty threshold | 0.35 |
| Phase 1 epochs / lr | 5 / 1e-3 |
| Phase 2 epochs / lr | ≤20 / 1e-5 |
| COVID class weight | 2.489× |
| Focal Loss γ | 2 |
| Dropout rate | 0.4 |

---

## 13. Likely Conceptual Questions and Clean Answers

**"What is transfer learning and why does it work?"**
> Pretrained weights encode general visual features (edges, textures, shapes) learned from 1.2M ImageNet images. These features generalise to medical images. Fine-tuning adapts the high-level task-specific layers to X-ray patterns while keeping the general low-level features intact.

**"What is the difference between aleatoric and epistemic uncertainty?"**
> Aleatoric is data uncertainty — inherent ambiguity in the image itself (e.g., borderline findings). It doesn't decrease with more data. Epistemic is model uncertainty — the model hasn't seen enough examples like this. It can decrease with more training data. MC Dropout captures both.

**"Why Focal Loss over weighted cross-entropy alone?"**
> Weighted cross-entropy upweights the rare class but still spends training capacity on easy majority-class examples. Focal Loss additionally down-weights easy examples regardless of class, so the model focuses on hard, informative samples across all classes simultaneously. Together they address both rarity and difficulty.

**"What does AUC-weighted ensemble add over simple averaging?"**
> Simple averaging assumes all models are equally good. AUC-weighting gives more influence to the model with better discriminative performance on the validation set. In practice the weights were similar (0.33 each) because all three models converged to similar AUC, so the benefit was modest — but the principle is sound.

**"How do you know Grad-CAM is actually highlighting the right region?"**
> For pneumonia cases, heat-maps consistently activate on lower-lobe consolidation — consistent with radiological ground truth. For COVID, activation spreads bilaterally — consistent with ground-glass opacity patterns. This qualitative consistency validates the maps, though formal comparison with radiologist annotations would be needed for clinical certification.

**"What is catastrophic forgetting and how does Phase 1 prevent it?"**
> Catastrophic forgetting is when fine-tuning a pretrained network on a new task overwrites previously learned features. If you unfreeze all layers immediately with a high learning rate, the large random gradients from the untrained head destroy the backbone's pretrained weights. Phase 1 stabilises the head first at a high lr; Phase 2 then fine-tunes the backbone at a 100× lower lr, preserving most of the pretrained features.
