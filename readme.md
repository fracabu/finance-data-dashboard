# Advanced Finance Analytics Dashboard

## Overview
The **Advanced Finance Analytics Dashboard** is an interactive Streamlit application designed to analyze financial data, provide visual insights, and perform technical and risk analytics. The dashboard supports file uploads, data filtering, technical indicators, risk assessment, portfolio analysis, and API integrations.

## Features

### 1. Main Dashboard
- Upload financial datasets in **CSV**, **Excel**, or **JSON** format.
- Automatically parse and process data, including date handling.
- Filter data by date range and symbols.
- Display key performance indicators (KPIs) such as:
  - Total rows
  - Total volume
  - Volatility
  - Sharpe ratio
  - Average daily return
- Export filtered datasets in CSV, Excel, or JSON formats.
- Enhanced visualizations:
  - Candlestick charts
  - Simple Moving Average (SMA) and Exponential Moving Average (EMA) overlays

### 2. Advanced Analytics
- Generate a **Correlation Heatmap** for selected assets.
- Display statistical summaries (mean, median, etc.).
- Perform **Principal Component Analysis (PCA)** for dimensionality reduction.
- **K-Means Clustering** for grouping assets based on price and volume.
- **Linear Regression Analysis** for price trends and future predictions.

### 3. Technical Analysis
- Calculate and visualize:
  - **Relative Strength Index (RSI)**
  - **Moving Average Convergence Divergence (MACD)**
  - **Bollinger Bands**
- Set customizable parameters for each indicator.

### 4. Risk Analytics
- Key risk metrics:
  - Value at Risk (VaR)
  - Expected Shortfall
  - Maximum Drawdown
  - Kurtosis and skewness of returns
- Analyze historical drawdowns.
- Visualize return distributions.

### 5. Portfolio Analysis
- Correlation matrix for portfolio assets.
- Annual returns and volatility for selected assets.
- Cumulative return plots.
- Risk-return scatter plots for profiling.

### 6. Settings
- Customize:
  - Chart themes and dimensions.
  - Anomaly detection thresholds.
  - Data handling methods (e.g., outlier detection, missing value imputation).
- Save preferences for future sessions.
# 📊 Financial Analytics Dashboard - Premium Version 
![image](https://github.com/user-attachments/assets/b78d82b3-c35a-408c-a7f2-d86491260559)


## 🌟 [English]

### 🎯 Overview
A comprehensive financial analytics platform built with Streamlit, offering advanced data analysis capabilities, technical indicators, and portfolio management tools.

### 7. API Integration
- Authenticate using API keys for secured endpoints.
- Available endpoints:
  - Market Summary
  - Technical Indicators
  - Historical Data
  - Generate Synthetic Data
- Fetch data dynamically from APIs with options for auto-refresh and export.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

### 🚀 Features

#### 💼 Data Management
- 📁 Multi-format support (CSV, Excel, JSON)
- ⚡ Real-time data filtering
- 📅 Custom date range selection
- 🔄 Multi-symbol support

#### 📈 Analytics Features
- **📊 Main Dashboard**
  * Real-time visualization
  * Candlestick charts
  * Volume analysis
  * Moving averages

- **🔬 Advanced Analytics**
  * Correlation heatmaps
  * PCA Analysis
  * K-means clustering
  * Linear Regression
  * Statistical summaries

- **📉 Technical Analysis**
  * RSI
  * MACD
  * Bollinger Bands
  * Custom indicators

- **⚠️ Risk Analytics**
  * VaR calculations
  * Maximum Drawdown
  * Returns distribution
  * Risk metrics

### 🛠️ Technical Requirements
```txt
# Core packages
streamlit>=1.31.0
pandas>=2.2.0
numpy>=1.26.0

# Data visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
```
[View full requirements.txt](requirements.txt)

---

## 🌟 [Italiano]

### 🎯 Panoramica
Una piattaforma completa di analisi finanziaria costruita con Streamlit, che offre capacità avanzate di analisi dati, indicatori tecnici e strumenti di gestione del portafoglio.
![image](https://github.com/user-attachments/assets/cb2dda45-fce2-4692-90df-13854bf57e5b)

### ⚡ Avvio Rapido
```bash
# Clona il repository
git clone [url-repository]

# Installa le dipendenze
pip install -r requirements.txt

# Avvia l'applicazione
streamlit run main.py
```

## Usage
1. Launch the app and upload your financial dataset.
2. Explore the tabs for analysis:
   - Use the **Main Dashboard** to get a quick overview and export data.
   - Dive into **Advanced Analytics** for clustering, PCA, and statistical summaries.
   - Use **Technical Analysis** to compute and visualize indicators.
   - Assess portfolio risk and performance under the **Risk Analytics** and **Portfolio** tabs.
3. Integrate API data to fetch live financial insights.

## Built With
- **Python**: Core programming language.
- **Streamlit**: Web application framework.
- **Plotly**: Interactive data visualizations.
- **Scikit-learn**: Machine learning algorithms.
- **TA-Lib**: Technical analysis indicators.

## Contributors
Created with ❤️ using Streamlit.

Ecco il **README** in italiano:

---

# Advanced Finance Analytics Dashboard

## Panoramica
**Advanced Finance Analytics Dashboard** è un'applicazione interattiva sviluppata con Streamlit, progettata per analizzare dati finanziari, fornire approfondimenti visivi e offrire strumenti avanzati di analisi tecnica e gestione del rischio. La dashboard supporta il caricamento di file, il filtraggio dei dati, l'analisi di indicatori tecnici, la valutazione del rischio, l'analisi del portafoglio e l'integrazione con API.

## Funzionalità

### 1. Dashboard Principale
- Caricamento di dataset finanziari in formato **CSV**, **Excel** o **JSON**.
- Parsing e processamento automatico dei dati, inclusa la gestione delle date.
- Filtraggio dei dati per intervallo di date e simboli.
- Visualizzazione dei KPI principali:
  - Numero totale di righe
  - Volume totale
  - Volatilità
  - Indice di Sharpe
  - Rendimento giornaliero medio
- Esportazione dei dataset filtrati in formato CSV, Excel o JSON.
- Visualizzazioni avanzate:
  - Grafici Candlestick
  - Overlay di Medie Mobili Semplici (SMA) e Medie Mobili Esponenziali (EMA)

### 2. Analisi Avanzata
- Generazione di una **Mappa di Correlazione** per gli asset selezionati.
- Visualizzazione di riepiloghi statistici (media, mediana, ecc.).
- **Analisi delle Componenti Principali (PCA)** per ridurre le dimensioni dei dati.
- **Clustering K-Means** per raggruppare gli asset in base a prezzo e volume.
- **Analisi di Regressione Lineare** per individuare trend di prezzo e previsioni future.

### 3. Analisi Tecnica
- Calcolo e visualizzazione di indicatori tecnici:
  - **Relative Strength Index (RSI)**
  - **Moving Average Convergence Divergence (MACD)**
  - **Bande di Bollinger**
- Parametri personalizzabili per ciascun indicatore.

### 4. Analisi del Rischio
- Metriche di rischio principali:
  - Value at Risk (VaR)
  - Expected Shortfall
  - Maximum Drawdown
  - Curtosi e asimmetria dei rendimenti
- Analisi dei drawdown storici.
- Visualizzazione della distribuzione dei rendimenti.

### 5. Analisi del Portafoglio
- Matrice di correlazione per gli asset del portafoglio.
- Ritorni annuali e volatilità degli asset selezionati.
- Grafici dei rendimenti cumulativi.
- Diagrammi a dispersione rischio-rendimento per il profiling.

### 6. Impostazioni
- Personalizzazione:
  - Tema e dimensioni dei grafici.
  - Soglie per il rilevamento di anomalie.
  - Metodi di gestione dei dati (es. rilevamento degli outlier, gestione dei valori mancanti).
- Salvataggio delle preferenze per sessioni future.

### 7. Integrazione API
- Autenticazione tramite chiavi API per endpoint protetti.
- Endpoint disponibili:
  - Sommario del Mercato
  - Indicatori Tecnici
  - Dati Storici
  - Generazione di Dati Sintetici
- Recupero dinamico dei dati tramite API con opzioni di aggiornamento automatico ed esportazione.

## Installazione

1. Clona il repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Installa le dipendenze richieste:
   ```bash
   pip install -r requirements.txt
   ```

3. Crea un file `.env` con le seguenti variabili di ambiente:
   ```
   API_KEY=<la_tua_chiave_api>
   API_SECRET=<il_tuo_segreto_api>
   FLASK_DEBUG=<True/False>
   PORT=<numero_porta>
   ```

4. Avvia l'applicazione:
   ```bash
   streamlit run main.py
   ```

## Utilizzo
1. Avvia l'app e carica il tuo dataset finanziario.
2. Esplora i tab per effettuare analisi:
   - Usa la **Dashboard Principale** per ottenere una panoramica rapida ed esportare i dati.
   - Esplora l'**Analisi Avanzata** per clustering, PCA e riepiloghi statistici.
   - Usa l'**Analisi Tecnica** per calcolare e visualizzare indicatori tecnici.
   - Valuta il rischio e le performance del portafoglio nei tab **Analisi del Rischio** e **Portafoglio**.
3. Integra dati API per ottenere approfondimenti finanziari in tempo reale.

## Tecnologie Utilizzate
- **Python**: Linguaggio di programmazione principale.
- **Streamlit**: Framework per applicazioni web.
- **Plotly**: Visualizzazioni dati interattive.
- **Scikit-learn**: Algoritmi di machine learning.
- **TA-Lib**: Indicatori di analisi tecnica.

## Contributori
Creato con ❤️ utilizzando Streamlit.

