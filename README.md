# FraudGuard - Fraud Detection System

**Vergleich regelbasierter und ML-basierter Ansätze**

## Projektziel

Systematischer Vergleich zweier Ansätze zur Betrugserkennung bei Kreditkartentransaktionen mit interaktivem Dashboard.

## Setup

### 1. Virtual Environment erstellen

```bash
python3 -m venv venv

# Aktivieren
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Kaggle API Setup (für Dataset Download)

1. Gehe zu [kaggle.com/account](https://www.kaggle.com/account)
2. Scroll zu "API" → Click "Create New API Token"
3. `kaggle.json` wird heruntergeladen
4. Verschiebe nach:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`

### 4. Dataset herunterladen

```bash
# Im Projekt-Ordner
kaggle datasets download -d kartik2112/fraud-detection
unzip fraud-detection.zip -d data/raw/
```

## Projektstruktur

```
fraudguard-project/
├── data/
│   ├── raw/                    # Original Dataset
│   └── processed/              # Verarbeitete Daten
├── models/                     # Trainierte Modelle (.pkl)
├── src/                        # Source Code
│   ├── rules.py               # Rule Engine
│   ├── features.py            # Feature Engineering
│   └── utils.py               # Hilfsfunktionen
├── notebooks/                  # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_rule_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── app.py                      # Streamlit Dashboard
├── requirements.txt
└── README.md
```

## Entwicklungsablauf

### Woche 1: Setup & Rules
- [x] Projekt-Setup
- [ ] Dataset Download & EDA
- [ ] Rule Engine implementieren

### Woche 2: ML Training
- [ ] Feature Engineering
- [ ] XGBoost Training
- [ ] Evaluation & Vergleich

### Woche 3: Dashboard
- [ ] Streamlit App Setup
- [ ] 4 Tabs implementieren

### Woche 4: Finalisierung
- [ ] Dokumentation
- [ ] Präsentation

## Quick Start

```bash
# Notebook starten
jupyter lab

# Dashboard starten (später)
streamlit run app.py
```

## Technologie

- **Python** 3.9+
- **ML:** XGBoost, Scikit-learn
- **Dashboard:** Streamlit
- **Visualisierung:** Plotly, Matplotlib

## Dataset

**Synthetic Fraud Detection** (Kaggle)
- 1.3M Transaktionen
- 0.6% Fraud Rate
- Features: Zeit, Betrag, Geo, Kunde, Händler

## Kontakt

Julian - [Email]
