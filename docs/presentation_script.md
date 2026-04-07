# Presentation Script
## Automated Pneumonia and COVID-19 Detection from Chest X-Ray Images
**DS 216o: Applied AI in Healthcare — Rajib Roy (SR 24459)**

**Total time: 7 min presentation + 2 min Q&A**
> Keep a timer visible. Slides 1–9 fit in ~6:15, leaving ~45 sec for the live app demo before handoff to Q&A.

---

## Timing Overview

| Slide | Title | Time | Duration |
|---|---|---|---|
| 1 | Title | 0:00 | 0:15 |
| 2 | Problem Statement | 0:15 | 0:45 |
| 3 | Methodology | 1:00 | 1:15 |
| 4 | Dataset | 2:15 | 0:40 |
| 5 | Experiments | 2:55 | 0:40 |
| 6 | Results | 3:35 | 0:55 |
| 7 | SOTA Comparison | 4:30 | 0:45 |
| 8 | Summary & Future Work | 5:15 | 0:35 |
| 9 | Visual Analysis | 5:50 | 0:25 |
| — | **Live App Demo** | 6:15 | 0:45 |
| — | Q&A | 7:00 | 2:00 |

---

## Slide-by-Slide Script

---

### SLIDE 1 — Title `[0:00–0:15]`

> *Click to open. Pause one beat.*

"Good morning. My project is on automated detection of pneumonia and COVID-19 from chest X-ray images using deep learning — with a focus on making predictions not just accurate, but trustworthy and explainable."

---

### SLIDE 2 — Problem Statement `[0:15–1:00]`

"Chest X-rays are the frontline diagnostic tool for respiratory disease, but manual reading is slow and highly variable between radiologists. The COVID-19 pandemic made this even more critical — we needed fast, scalable screening.

My primary objective is a three-class classifier: Normal, Pneumonia, and COVID-19. But beyond classification, there are two things that matter for clinical use: the model needs to know *when it doesn't know* — that's the uncertainty piece — and it needs to *show its reasoning* — that's Grad-CAM.

The core challenge is class imbalance: COVID accounts for only 13% of the data, so naive training simply ignores it."

---

### SLIDE 3 — Methodology `[1:00–2:15]`

"I trained three ImageNet-pretrained models — DenseNet121, ResNet50, and EfficientNetB0 — each with a shared classification head using batch normalisation and dropout.

Training uses a two-phase strategy. Phase one freezes the backbone and trains only the head for five epochs to stabilise initialisation. Phase two unfreezes the last convolutional block and fine-tunes jointly at a much lower learning rate to avoid catastrophic forgetting.

For the loss, I use Focal Loss — which down-weights easy examples so the model focuses on hard, misclassified ones — combined with inverse-frequency class weights that give the COVID class a 2.5× boost.

The three models are combined through AUC-weighted soft voting into an ensemble. On top of that, MC Dropout runs 50 stochastic forward passes at inference to produce an uncertainty score. If the combined score exceeds 0.35, the prediction is automatically flagged for radiologist review. Finally, Grad-CAM generates a saliency heat-map from the last convolutional block of each model."

---

### SLIDE 4 — Dataset `[2:15–2:55]`

"The dataset combines two public Kaggle sources: the Kermany chest X-ray dataset and the COVID-19 Radiography Database. After label unification and hash-based duplicate removal I end up with 27,005 unique images across three classes, split 80-10-10 stratified by class.

You can see the three sample X-rays here — Normal on the left with clear lungs, bacterial Pneumonia in the centre showing lower-lobe consolidation, and COVID-19 on the right with bilateral ground-glass opacity. These are the same images I'll use in the live demo."

---

### SLIDE 5 — Experiments `[2:55–3:35]`

"All training was done locally on an RTX 3060 with mixed precision. Phase one final validation accuracy was around 80% for DenseNet and ResNet, and 87% for EfficientNet — which already shows EfficientNet converging faster even with a frozen backbone.

After Phase two fine-tuning, DenseNet reaches 90%, ResNet 92%, and EfficientNet 94%. The training curves show this clearly — loss drops steadily through both phases, and early stopping correctly fires for ResNet and EfficientNet before they overfit."

---

### SLIDE 6 — Results `[3:35–4:30]`

"On the 2,701 test images, the ensemble achieves 93.7% accuracy, 0.99 macro AUC, and 0.94 macro F1. The most important number for me is COVID F1 at 0.947 — up nearly 8 points from DenseNet alone — because misclassifying COVID as Pneumonia is the most dangerous confusion clinically.

The confusion matrices confirm this: the ensemble substantially reduces that specific off-diagonal error compared to individual models.

The uncertainty mechanism is equally important: 44.8% of cases are flagged for review, but the remaining 55% that the model *is* confident about achieve 99.4% accuracy — so it degrades gracefully rather than failing silently."

---

### SLIDE 7 — Comparison with SOTA `[4:30–5:15]`

"Comparing with published work — our ensemble is competitive. COVID-Net gets 92.7% on 14,000 images; we get 93.7% on 27,000. Narin et al. report 98% but on only 4,200 images, which is a very small test bed.

The key differentiator isn't just accuracy. We are the only method in this comparison that combines uncertainty quantification *and* visual explainability. Every other method gives you a label — we give you a label, a confidence score, a referral flag, and a heat-map showing exactly where the model is looking."

---

### SLIDE 8 — Summary & Future Work `[5:15–5:50]`

"To summarise: 27,000 images, three models, AUC-weighted ensemble, Focal Loss for imbalance, MC Dropout for uncertainty, Grad-CAM for explainability, deployed as a Streamlit app.

The main limitation is the 44% referral rate — which is high for real deployment and something I'd address next with temperature scaling. Other future directions include self-supervised pretraining on unlabelled X-rays and integrating a vision-language model for automated report generation."

---

### SLIDE 9 — Visual Analysis `[5:50–6:15]`

"These four plots capture the full evaluation picture. Training curves confirm no overfitting. The ROC curves show AUC above 0.99 for Normal and Pneumonia. The calibration diagrams show the ensemble is well-calibrated — predicted probabilities match empirical frequencies closely. And the uncertainty plot shows correct predictions cluster at low uncertainty while wrong ones cluster high — validating the referral threshold."

---

### LIVE APP DEMO `[6:15–7:00]`

> *Switch to browser with Streamlit app already open. Have `test_images/` folder ready in the file picker.*

**Script:**
"Let me quickly show this running live."

1. *Upload* `IM-0023-0001.jpeg` (Normal)
   — "Normal X-ray — model predicts Normal with high confidence, low uncertainty, no referral."

2. *Upload* `person1946_bacteria_4875.jpeg` (Pneumonia)
   — "Pneumonia case — confident prediction, and if I toggle Grad-CAM you can see the model activating on the lower-lobe consolidation region."

3. *Upload* `x-ray-image-2b_full.jpg` (COVID-19)
   — "COVID case — bilateral ground-glass opacity, the heat-map spreads across both lungs."

> *If time is tight, skip image 3 and go straight to Q&A.*

---

---

## Q&A Preparation `[7:00–9:00]`

---

**Q: Why not use a single larger model instead of an ensemble?**

> "Ensembles give you diversity — each architecture captures different features. DenseNet reuses features through dense connections, ResNet learns residuals, EfficientNet scales compound-wise. The AUC-weighted voting combines their complementary strengths, and empirically we gain ~0.7% accuracy and ~8 points on COVID F1 over the best single model."

---

**Q: The 44% referral rate seems too high for clinical use.**

> "Agreed — it is high, and I acknowledge that as a limitation. The threshold of 0.35 was chosen conservatively for safety. Temperature scaling or isotonic regression post-hoc can recalibrate the uncertainty scores and bring the referral rate down without sacrificing the coverage guarantee on hard cases."

---

**Q: How does Grad-CAM help clinically if it just highlights the obvious area?**

> "For confident correct predictions it often does highlight the obvious region — which is actually a sanity check that the model isn't using spurious features. The value is greatest on borderline cases: if the model is uncertain *and* the heat-map doesn't activate on lung parenchyma, that's a strong signal to flag for review. It's a debugging tool as much as an explanation tool."

---

**Q: Wouldn't data from only two Kaggle sources introduce bias?**

> "Yes, that's a real concern — different scanners, acquisition protocols, and patient demographics. Hash-based deduplication reduces cross-dataset leakage, and the stratified split ensures class balance. Prospective validation on an independent hospital dataset would be the necessary next step before any real deployment."

---

**Q: What prevents this from being deployed in a hospital today?**

> "Three things: no prospective clinical validation, no DICOM/PACS integration, and no regulatory clearance — this would need FDA 510(k) or CE marking. The current system is research-grade. The uncertainty referral mechanism is a step toward safety, but clinical deployment needs a formal trial."

---

**Q: Could this work without a GPU?**

> "Yes — the Streamlit app runs on CPU, just slower. Inference per image takes a few seconds on CPU versus milliseconds on GPU. Training from scratch definitely needs a GPU."

---

> **General tip for Q&A:** If you don't know the answer, say *"That's a great direction — I haven't explored that yet, but my hypothesis would be..."* — never bluff on medical AI claims.
