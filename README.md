# Fruit Freshness and Spoilage Classification using Deep Learning

Fresh fruit is still checked for quality mostly by eye — slow, tiring and not always
consistent. A rotten fruit packed with good fruit can spoil the whole batch. This project
builds an image classifier that gives the same judgement to every photo, in a fraction of
a second: a photo goes in, and the model answers GOOD or ROTTEN.

The question is deliberately simple. The project does **not** grade ripeness and does
not identify the fruit — it answers the one question a shopkeeper, packing worker or
person at home most often needs answered.

## Dataset

**Source:** [Fruit Ripeness: Unripe, Ripe, and Rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)
by leftin (Kaggle), CC BY-SA 4.0. **19,956 images** were used, in the training and test
folders supplied with the dataset.

> Naming note: on Kaggle the middle class is called **"ripe"**; the project notebook
> calls it **"fresh"**. It is the same class.

The nine original folders were grouped into two classes:

| Class | Contains |
|-------|----------|
| **GOOD** | Fresh + unripe (apple, banana, orange) — the fruit is usable |
| **ROTTEN** | Rotten (apple, banana, orange) — the fruit is spoiled |

| Split | GOOD | ROTTEN | Total |
|-------|------|--------|-------|
| Train | 10,056 | 6,161 | 16,217 (81.3%) |
| Test | 2,205 | 1,534 | 3,739 (18.7%) |
| **Total** | **12,261** | **7,695** | **19,956** |

Per-folder counts (train / test): fresh apples 1,693 / 395, fresh banana 1,581 / 381,
fresh oranges 1,466 / 388, unripe apple 1,934 / 371, unripe banana 2,097 / 400,
unripe orange 1,285 / 270, rotten apples 2,342 / 601, rotten banana 2,224 / 530,
rotten oranges 1,595 / 403.

## Methodology

```
Input photo → Resize → Model-specific preprocessing → Frozen ImageNet backbone
→ Global Average Pooling → Dense 256 (ReLU) → Dropout 0.4 → Sigmoid
→ Probability → Threshold 0.5 → GOOD / ROTTEN
```

**Training setup (identical for all ten models):**

| Setting | Value |
|---------|-------|
| Framework | TensorFlow 2.21.0 on GPU (Google Colab) |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Epochs | Max 20, early stopping on validation loss (patience 3, best weights restored) — models actually ran 5–15 epochs |
| Metrics | Accuracy, precision, recall, AUC |
| Decision threshold | 0.5 (≥ 0.5 = ROTTEN, < 0.5 = GOOD) |
| Data augmentation | None used |
| Seed | 42 |

All ten models share the same head: a **frozen** ImageNet backbone, global average
pooling, Dense 256 + ReLU, Dropout 0.4, and a single sigmoid output. Only the small
head is trained — for example, ResNet50 has 24,112,513 parameters in total but only
524,801 trainable ones. Input size is 224×224 for most models, 240×240 for
EfficientNetB1, and 299×299 for InceptionV3 and Xception.

## Results

Evaluated on the held-out test set of **3,739 images**. Models ranked by test accuracy:

| Model | Accuracy | Precision | Recall | F1-score | AUC | Training time |
|-------|----------|-----------|--------|----------|-----|---------------|
| **MobileNetV3Large** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 17.08 min |
| EfficientNetB1 | 0.9989 | 0.9987 | 0.9987 | 0.9987 | 1.0000 | 17.21 min |
| DenseNet169 | 0.9987 | 0.9974 | 0.9993 | 0.9984 | 1.0000 | 16.82 min |
| ResNet50 | 0.9984 | 0.9980 | 0.9980 | 0.9980 | 0.9997 | 13.69 min |
| EfficientNetB0 | 0.9984 | 0.9974 | 0.9987 | 0.9980 | 0.9998 | 16.18 min |
| Xception | 0.9963 | 0.9974 | 0.9935 | 0.9954 | 0.9997 | 39.41 min |
| DenseNet121 | 0.9960 | 0.9941 | 0.9961 | 0.9951 | 0.9997 | 16.08 min |
| MobileNetV2 | 0.9952 | 0.9903 | 0.9980 | 0.9942 | 0.9999 | 8.07 min |
| ResNet101 | 0.9949 | 0.9954 | 0.9922 | 0.9938 | 0.9995 | 11.40 min |
| InceptionV3 | 0.9947 | 0.9941 | 0.9928 | 0.9935 | 0.9999 | 21.24 min |

*Note on rounding: AUC values are rounded to four decimals. Only MobileNetV3Large has
an exact AUC of 1.0; DenseNet169 (0.99998) and EfficientNetB1 (0.999996) round to 1.0000.*

- **Best confusion matrix (MobileNetV3Large):** 2,205 GOOD correct, 0 wrong;
  1,534 ROTTEN correct, 0 wrong.
- **Sample predictions:** on six selected test images, MobileNetV3Large was correct
  6/6 at 100.00% confidence; across all ten models, **60/60 predictions correct**.
- **Single-image check:** on a fresh green apple, all ten models predicted GOOD —
  eight at 100%, InceptionV3 at 99.97%, Xception at 99.99%.
- **Average precision:** 1.0000 for MobileNetV3Large; above 0.9994 for every model
  (lowest: ResNet101 at 0.9994).
- **Training time:** ~177 minutes in total. Fastest: MobileNetV2 (8.07 min).
  Slowest: Xception (39.41 min).

**Practical pick:** MobileNetV3Large for accuracy, MobileNetV2 when speed or a light
model matters. Xception is the worst trade-off here — slowest and only sixth most accurate.

## How to Run

1. Open the notebook in Google Colab:
   [AIDL PROJECT.ipynb](https://colab.research.google.com/drive/1M2gKKhLd7YPKXWxRkMkFBW-WqW7MzWeK)
2. Mount Google Drive — the notebook reads the dataset from `dataset/dataset`
   (train / test folders) and writes all results to `dataset/results`.
3. Run all cells on GPU. Training the ten models takes about 3 hours in total.
4. Results are saved automatically: `model_comparison.csv`, comparison graphs,
   one confusion matrix per model, and the visual prediction outputs in `results/`.

To test one new photo, use the single-image test cell at the end of the notebook — it
loads an image, runs all ten models on it, and reports each prediction with confidence.

## Project Structure

```
.
├── AIDL PROJECT.ipynb                          # Executed notebook (Google Colab)
├── dataset/
│   ├── dataset/train/  dataset/test/          # 9 fruit folders, grouped to GOOD / ROTTEN
│   └── results/
│       ├── model_comparison.csv              # All 10 models: metrics + training time
│       ├── comparison_graphs/                # Training curves, ROC, PR curves, rankings
│       ├── confusion_matrices/               # One confusion matrix per model
│       └── visual/                           # 6-image prediction outputs
├── Fruit-Freshness-Spoilage-Classification-13-Slides.pptx   # Presentation (13 slides)
├── Fruit-Freshness-Spoilage-Classification-Report.docx/pdf  # Course project report (22 pages)
└── Rehearsal-Run-Through.md                  # ~12.5 min rehearsal sheet + viva Q&A
```

## Limitations (stated honestly)

- Only three fruits (apple, banana, orange) and two classes — fresh and unripe are
  grouped as GOOD, so the model does **not** grade ripeness.
- **No separate validation set:** the test set was used as the validation set during
  training, so the reported scores are not from a fully untouched test set.
- Photos are clean, single-fruit images; cluttered photos, fruit in a pile, or poor
  lighting are outside the training distribution.
- No baseline-versus-improved experiment exists in the notebook — the comparison is
  the ten pretrained models against each other.
- A test accuracy of 100% on one clean test set is not a guarantee of 100% on new
  photos; it should be re-checked on photographs taken outside the dataset.

## Future Scope

- Three-class classification (fresh / unripe / rotten) so unripe becomes its own class
- More fruits beyond apple, banana and orange
- A proper validation split, so the test set is used only once
- Testing on new photos taken outside the dataset
- Data augmentation during training
- A simple phone/web app: upload a photo, get GOOD / ROTTEN with confidence

## References

1. Fruit Ripeness dataset: leftin/fruit-ripeness-unripe-ripe-and-rotten.
   [kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten](https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten)
   — CC BY-SA 4.0.
2. Project notebook: AIDL PROJECT.ipynb — Fruit Condition Classification (GOOD vs ROTTEN),
   executed in Google Colab.
3. He, K. et al. "Deep Residual Learning for Image Recognition." Proc. IEEE Conference on
   Computer Vision and Pattern Recognition (CVPR), 2016. — ResNet50, ResNet101.
4. Huang, G. et al. "Densely Connected Convolutional Networks." Proc. IEEE Conference on
   Computer Vision and Pattern Recognition (CVPR), 2017. — DenseNet121, DenseNet169.
5. Tan, M. & Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural
   Networks." Proc. International Conference on Machine Learning (ICML), 2019. —
   EfficientNetB0, EfficientNetB1.
6. Sandler, M. et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." Proc.
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. — MobileNetV2.
7. Howard, A. et al. "Searching for MobileNetV3." Proc. IEEE/CVF International Conference
   on Computer Vision (ICCV), 2019. — MobileNetV3Large.
8. Szegedy, C. et al. "Rethinking the Inception Architecture for Computer Vision." Proc.
   IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. — InceptionV3.
   Chollet, F. "Xception: Deep Learning with Depthwise Separable Convolutions." Proc. IEEE
   Conference on Computer Vision and Pattern Recognition (CVPR), 2017. — Xception.
9. TensorFlow Contributors. TensorFlow 2.21.0 and Keras Applications documentation.
   [tensorflow.org](https://www.tensorflow.org)
