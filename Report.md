# Report

## Model Training Experiments

### Experiment 1: Number of Epochs

Obiettivo: stimare il numero di epoche utile senza sprecare tempo e senza overfitting. Grazie all’**early stopping**, entrambe le reti si stabilizzano entro \~10–12 epoche.

**Esempio (CNN\_Simple — tuo log):**

```
Epoch 1: train loss 0.452 acc 0.755 f1 0.743 | val loss 0.150 acc 0.958 f1 0.958
Epoch 4: train loss 0.096 acc 0.965 f1 0.965 | val loss 0.133 acc 0.988 f1 0.987
Epoch 8: train loss 0.062 acc 0.974 f1 0.974 | val loss 0.214 acc 0.979 f1 0.979
Early stopping triggered.
```

**Conclusione:** oltre \~10 epoche non si osservano miglioramenti significativi.

---

### Experiment 2: Network Architecture

Confronto tra **CNN\_Simple** (costruita manualmente) e **ResNet18** (transfer learning, pretrained ImageNet). Stesse trasformazioni e stessi iperparametri di base.

**Val (estratto ResNet18 — tuo log):**

```
Epoch 1: val loss 0.272  acc 0.958  f1 0.958
Epoch 7: val loss 0.091  acc 0.979  f1 0.979
Epoch 11: val loss 0.085  acc 0.988  f1 0.987
Epoch 15: val loss 0.074  acc 0.979  f1 0.979
```

**Test set (metriche finali):**

| Model       | Accuracy | Precision | Recall | F1     |
| ----------- | -------- | --------- | ------ | ------ |
| CNN\_Simple | 0.9667   | 0.9688    | 0.9667 | 0.9666 |
| ResNet18    | 0.9917   | 0.9917    | 0.9917 | 0.9917 |

**Note qualitative:** ResNet18 mostra **convergenza più rapida** e **qualità superiore**; la CNN\_Simple è **didatticamente utile** e già competitiva.

---

### Experiment 3: Learning Rate

Non è stata eseguita una ricerca sistematica del *learning rate* per ragioni di tempo. Con LR=**0.0005** si ottiene convergenza stabile. Possibili sviluppi: confronto **0.001** e **0.0001** su 5–10 epoche con early stopping. Nota: anche senza tuning esteso, le prestazioni sono rimaste stabili.

---

## Dataset & Preprocessing

* Dataset: **PlantVillage**.
* Classi usate: `Tomato_healthy` e `Tomato_Late_blight` (binaria).
* Split bilanciato (seed fisso): **train/val/test = 600/120/120** per classe (tot. 1680 img). Vedi `plots/split_bars.png`.
* Trasformazioni (train): resize 256 → crop 224 → toTensor → normalize (ImageNet) + flip orizzontale + *color jitter* leggero.
* Trasformazioni (val/test): resize 256 → crop 224 → toTensor → normalize.
* Nota: il dataset non è incluso nel repo; gli split possono essere ricreati lanciando `src.data_prep`.

## Training Setup

* Optimizer: **Adam** (lr=0.0005, weight\_decay=1e-4).
* Loss: **CrossEntropyLoss** (2 classi).
* Early stopping su **val loss**, salvataggio **best/last**.
* Metriche: **Accuracy**, **Precision**, **Recall**, **F1 macro**.
* Log: **TensorBoard** (`runs/`)con curve Loss/Acc/F1/Precision/Recall/Support di validazione. Nel tab **HParams** vengono mostrati i risultati riepilogativi del best model (Accuracy/Precision/Recall/F1 su validation).

## Results & Analysis

* **Confusion matrices:** `plots/cm_cnn_simple.png`, `plots/cm_resnet18.png`.
* **Trend di training:** da TensorBoard (Loss/Acc/F1 per CNN e ResNet18).
* **Commento:** entrambi i modelli performano molto bene; ResNet18 è superiore su tutte le metriche. Pochi falsi positivi/negativi nelle CM. I grafici mostrano che ResNet18 converge più rapidamente e con maggiore stabilità rispetto a CNN\_Simple. Nessun segno evidente di overfitting grazie a early stopping e augmentation leggera.

## Conclusions

* Sono stati confrontati un **modello manuale** e un **modello con transfer learning**.
* I risultati sono riproducibili (seed, config, salvataggi).
* Possibili estensioni: fine-tuning parziale di ResNet18, aumento graduale del budget per classe, test con ResNet50.
* Il logging avanzato in TensorBoard (Scalars + HParams) permette di confrontare facilmente CNN e ResNet18 e di verificare le metriche chiave di validazione a ogni run.