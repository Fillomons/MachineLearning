# MachineLearning

# README

## Start-up

Per iniziare crea un nuovo *environment* (anche venv) ed installa i pacchetti dal file **requirements.txt**:

```bash
pip install -r requirements.txt
```

In alternativa, se preferisci **conda**, crea l’ambiente a partire dal file `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate progetto-ai
```

Poi prepara il dataset (PlantVillage – binaria: *Tomato\_healthy* vs *Tomato\_Late\_blight*):

1. Scarica PlantVillage (Kaggle o mirror GitHub) e metti le cartelle in:

   ```
   data/raw/PlantVillage/Tomato_healthy/
   data/raw/PlantVillage/Tomato_Late_blight/
   ```
2. Crea subset e split bilanciati (seed fisso) lanciando:

   ```bash
   python -m src.data_prep
   ```

   **Output atteso:**

   ```
   class                 train  val  test  total
   Tomato_healthy         600   120   120    840
   Tomato_Late_blight     600   120   120    840
   ```

Infine entra nel progetto ed avvia il training/evaluation:

* **CNN semplice (baseline fatta a mano)**

  ```bash
  # configs/config.json → "model": {"name": "cnn_simple", "finetune": false}
  python -m src.train
  python -m src.eval
  ```
* **ResNet18 (transfer learning)**

  ```bash
  # configs/config.json → "model": {"name": "resnet18", "finetune": false}
  python -m src.train
  python -m src.eval
  ```

Per visualizzare i grafici di addestramento (Loss/Acc/F1):

```bash
tensorboard --logdir runs
# si apre http://localhost:6006
```

---

## Context

Lo scopo della rete è classificare immagini di foglie di pomodoro in **due classi**: **sana** (*Tomato\_healthy*) e **affetta da late blight** (*Tomato\_Late\_blight*). Il dataset utilizzato è **PlantVillage**; per rispettare i vincoli di tempo e calcolo, è stato creato un **subset bilanciato** con split **train/val/test = 600/120/120** per classe (totale 1680 immagini). Il dataset non è incluso nel repo ma può essere rigenerato lanciando lo script `src.data_prep`.

Le reti considerate:

* **CNN\_Simple**: rete *from scratch* (conv → ReLU → maxpool ×4, classifier con AdaptiveAvgPool2d + Linear) — modello base costruito manualmente.
* **ResNet18 (pretrained ImageNet)**: **transfer learning** con ultimo layer sostituito (2 classi), training del classificatore; stessi pre-processamenti.

Trasformazioni: resize 256, crop 224, normalizzazione ImageNet, in train anche flip orizzontale e *color jitter* leggero.

Ottimizzatore: Adam, **early stopping** su *val loss*, salvataggio **best/last**, metriche: Accuracy, Precision, Recall, **F1 macro**.

---

## Experiments

### Exp1: Epochs

Early stopping → stabilizzazione entro \~10–12 epoche.

CNN (log):

```
Epoch 1: train loss 0.452 acc 0.755 f1 0.743 | val loss 0.150 acc 0.958 f1 0.958
...
Epoch 8: train loss 0.062 acc 0.974 f1 0.974 | val loss 0.214 acc 0.979 f1 0.979
Early stopping triggered.
```

### Exp2: Architecture

Confronto cnn\_simple vs resnet18.

ResNet18 (val, log estratto):

```
Epoch 1: val loss 0.272 acc 0.958 f1 0.958
Epoch 7: val loss 0.091 acc 0.979 f1 0.979
Epoch 11: val loss 0.085 acc 0.988 f1 0.987
Epoch 15: val loss 0.074 acc 0.979 f1 0.979
```

**Test set:**

| Model       | Accuracy | Precision | Recall | F1    |
| ----------- | -------- | --------- | ------ | ----- |
| cnn\_simple | 0.967    | 0.969     | 0.967  | 0.967 |
| resnet18    | 0.992    | 0.992     | 0.992  | 0.992 |

**Osservazioni:**

* ResNet18 converge più rapidamente e raggiunge metriche superiori grazie al *transfer learning*.
* La CNN\_Simple, pur essendo compatta, fornisce una **baseline solida** con ottime prestazioni.

### Exp3: Learning Rate

Non è stata eseguita una ricerca sistematica del *learning rate* per ragioni di tempo. Con LR=0.0005 si ottiene convergenza stabile. Possibili sviluppi: confronto 0.001 e 0.0001.

---

## Results

* **Confusion matrices:**

  * `plots/cm_cnn_simple.png`
  * `plots/cm_resnet18.png`
* **Grafici di training:** da TensorBoard (Loss/Acc/F1 per entrambi i modelli)
* **Distribuzione dataset:** `plots/split_bars.png`

---

## How to run (riassunto)

```bash
# 1) Preparazione dataset
python -m src.data_prep

# 2) CNN baseline
# configs/config.json → "model": {"name": "cnn_simple", "finetune": false}
python -m src.train
python -m src.eval

# 3) ResNet18 (transfer learning)
# configs/config.json → "model": {"name": "resnet18", "finetune": false}
python -m src.train
python -m src.eval

# 4) TensorBoard (grafici)
tensorboard --logdir runs
```

---

## Notes

* Seed fisso per riproducibilità.
* Early stopping per prevenire overfitting.
* Salvataggio **best/last** in `models/`.
* Tutto eseguibile **su CPU** con i budget correnti (600/120/120 per classe).




