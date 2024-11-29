import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import time
import requests
import io
from scipy import stats
from sklearn.decomposition import PCA
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv
import os
from sklearn.linear_model import LinearRegression

# Caricamento delle variabili d'ambiente
load_dotenv()

# Recupera le variabili d'ambiente
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", default=False)
PORT = int(os.getenv("PORT", 5000))
RATE_LIMIT = os.getenv("RATE_LIMIT", default=None)
LOG_LEVEL = os.getenv("LOG_LEVEL", default="INFO")
LOG_FILE = os.getenv("LOG_FILE", default="api.log")

warnings.filterwarnings("ignore")

# Inizializzazione dello state per la gestione dei dati
if 'data' not in st.session_state:
    st.session_state.data = None
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = None

# Configurazione della pagina
st.set_page_config(page_title="Advanced Finance Analytics Dashboard", layout="wide")

# Funzioni di analisi tecnica
def calculate_rsi(prices, period=14):
    rsi_indicator = RSIIndicator(close=prices, window=period)
    return rsi_indicator.rsi()

def calculate_macd(prices):
    macd = MACD(close=prices)
    return macd.macd(), macd.macd_signal()

def calculate_bollinger_bands(prices, period=20):
    indicator_bb = BollingerBands(close=prices, window=period)
    return indicator_bb.bollinger_hband(), indicator_bb.bollinger_lband()

# Funzioni di analisi finanziaria
def calculate_financial_metrics(data):
    metrics = {}
    if "Close" in data.columns:
        data["Daily Returns"] = data["Close"].pct_change()
        metrics["Volatility"] = data["Daily Returns"].std() * np.sqrt(252)
        metrics["Average Daily Return"] = data["Daily Returns"].mean()
        metrics["Sharpe Ratio"] = metrics["Average Daily Return"] / metrics["Volatility"]
        data["SMA_20"] = data["Close"].rolling(window=20).mean()
        data["EMA_20"] = data["Close"].ewm(span=20).mean()
    return metrics

def detect_anomalies(data):
    if "Close" not in data.columns:
        return None
    daily_returns = data["Close"].pct_change()
    mean = daily_returns.mean()
    std = daily_returns.std()
    anomalies = data[abs(daily_returns - mean) > 2 * std]
    return anomalies

def perform_asset_clustering(data, n_clusters=3):
    if "Close" in data.columns and "Volume" in data.columns:
        metrics = data.groupby("Symbol").agg({
            "Close": "mean",
            "Volume": "sum"
        }).reset_index()
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(metrics[["Close", "Volume"]])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        metrics["Cluster"] = kmeans.fit_predict(scaled_features)
        
        return metrics
    return None

def calculate_risk_metrics(data):
    metrics = {}
    if "Close" in data.columns:
        returns = data["Close"].pct_change().dropna()
        metrics["Value at Risk (95%)"] = np.percentile(returns, 5)
        metrics["Expected Shortfall"] = returns[returns <= metrics["Value at Risk (95%)"]].mean()
        peak = data["Close"].expanding(min_periods=1).max()
        drawdown = (data["Close"] - peak) / peak
        metrics["Maximum Drawdown"] = drawdown.min()
        metrics["Kurtosis"] = stats.kurtosis(returns)
        metrics["Skewness"] = stats.skew(returns)
    return metrics

@st.cache_data
def load_data(file):
    try:
        if file.type == "text/csv" or file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" or file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        elif file.type == "application/json" or file.name.endswith(".json"):
            df = pd.read_json(file)
        else:
            st.error("Please upload only CSV, Excel or JSON files")
            return None
            
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def filter_data(data, filters):
    filtered_data = data.copy()
    for column, filter_value in filters.items():
        if column == "Date" and len(filter_value) == 2:
            start_date, end_date = filter_value
            filtered_data = filtered_data[
                (filtered_data[column] >= pd.to_datetime(start_date)) &
                (filtered_data[column] <= pd.to_datetime(end_date))
            ]
        elif filter_value:
            filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
    return filtered_data

# Funzioni di visualizzazione
def create_financial_visualizations(data, container):
    if "Close" in data.columns:
        # Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"]
        )])
        fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
        container.plotly_chart(fig, use_container_width=True)

        # SMA & EMA Overlay
        fig_sma_ema = go.Figure()
        fig_sma_ema.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", mode="lines"))
        fig_sma_ema.add_trace(go.Scatter(x=data["Date"], y=data["SMA_20"], name="SMA (20)", mode="lines"))
        fig_sma_ema.add_trace(go.Scatter(x=data["Date"], y=data["EMA_20"], name="EMA (20)", mode="lines"))
        fig_sma_ema.update_layout(title="SMA & EMA Analysis", xaxis_title="Date", yaxis_title="Price")
        container.plotly_chart(fig_sma_ema, use_container_width=True)

def create_correlation_heatmap(data, container):
    if "Close" in data.columns:
        pivot_data = data.pivot_table(index="Date", columns="Symbol", values="Close").fillna(0)
        correlation_matrix = pivot_data.corr()
        fig_corr = px.imshow(correlation_matrix, title="Correlation Heatmap")
        container.plotly_chart(fig_corr, use_container_width=True)

def calculate_kpi(data):
    kpis = {}
    kpis["Total Rows"] = len(data)
    if "Close" in data.columns:
        kpis["Total Close"] = data["Close"].sum()
    if "Volume" in data.columns:
        kpis["Total Volume"] = data["Volume"].sum()
    return kpis

# Sidebar per il tema
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"])

# Tema CSS
if theme == "Light":
    st.markdown("""
    <style>
    :root {
        --bg-primary: #f9fbfd;
        --bg-secondary: #ffffff;
        --text-primary: #1b2a4e;
        --text-secondary: #6e84a3;
        --accent: #2c7be5;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent: #38bdf8;
    }
    </style>
    """, unsafe_allow_html=True)

# Titolo della dashboard
st.title("Advanced Finance Analytics Dashboard")

# Creazione dei tab
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Main Dashboard",
    "🔍 Advanced Analytics",
    "📈 Technical Analysis",
    "⚠️ Risk Analytics",
    "💼 Portfolio",
    "⚙️ Settings",
    "🌐 API Integration"
])

# Tab 1: Main Dashboard
with tab1:
    st.subheader("Upload Financial Data for Analysis")
    uploaded_file = st.file_uploader(
        "Upload a file (CSV, Excel or JSON)",
        type=["csv", "xlsx", "json"],
        accept_multiple_files=False
    )

    if uploaded_file:
        # Load data and store in session state
        st.session_state.data = load_data(uploaded_file)
        if st.session_state.data is not None and not st.session_state.data.empty:
            st.success("File loaded successfully!")
            st.dataframe(st.session_state.data.head())

            # Create a copy for filters
            st.session_state.filtered_data = st.session_state.data.copy()

            # Handle Date column
            if "Date" in st.session_state.filtered_data.columns:
                try:
                    st.session_state.filtered_data["Date"] = pd.to_datetime(
                        st.session_state.filtered_data["Date"])
                    st.write("Date column processed successfully.")
                except Exception as e:
                    st.error(f"Error processing 'Date' column: {e}")

            # Filters in sidebar
            st.sidebar.header("Filters")
            filters = {}

            if "Date" in st.session_state.filtered_data.columns:
                min_date = st.session_state.filtered_data["Date"].min()
                max_date = st.session_state.filtered_data["Date"].max()
                
                if pd.notnull(min_date) and pd.notnull(max_date):
                    filters["Date"] = st.sidebar.date_input(
                        "Select Date Range",
                        [min_date.date(), max_date.date()],
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                    if len(filters["Date"]) == 2:
                        start_date, end_date = filters["Date"]
                        st.session_state.filtered_data = st.session_state.filtered_data[
                            (st.session_state.filtered_data["Date"].dt.date >= start_date) &
                            (st.session_state.filtered_data["Date"].dt.date <= end_date)
                        ]

            if "Symbol" in st.session_state.filtered_data.columns:
                all_symbols = st.session_state.filtered_data["Symbol"].unique()
                filters["Symbol"] = st.sidebar.multiselect(
                    "Select Symbols",
                    options=all_symbols,
                    default=all_symbols
                )
                if filters["Symbol"]:
                    st.session_state.filtered_data = st.session_state.filtered_data[
                        st.session_state.filtered_data["Symbol"].isin(filters["Symbol"])
                    ]

            # Display filtered data and KPIs
            if not st.session_state.filtered_data.empty:
                st.write("Filtered Data:")
                st.dataframe(st.session_state.filtered_data)

                # KPI Section
                st.header("Key Metrics")
                kpis = calculate_kpi(st.session_state.filtered_data)
                col1, col2 = st.columns(2)
                col1.metric("Total Rows", kpis.get("Total Rows", 0))
                col2.metric("Total Volume", f"{kpis.get('Total Volume', 0):,.0f}")

                # Advanced Metrics
                advanced_metrics = calculate_financial_metrics(st.session_state.filtered_data)
                st.header("Advanced Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Volatility", f"{advanced_metrics.get('Volatility', 0):.2%}")
                col2.metric("Sharpe Ratio", f"{advanced_metrics.get('Sharpe Ratio', 0):.2f}")
                col3.metric("Avg Daily Return", f"{advanced_metrics.get('Average Daily Return', 0):.2%}")

                # Create visualizations
                st.header("Enhanced Visualizations")
                create_financial_visualizations(st.session_state.filtered_data, st)

                # Export Section
                st.markdown("---")
                st.header("Export Data")
                col1, col2, col3 = st.columns(3)
                
                # CSV Export
                csv = st.session_state.filtered_data.to_csv(index=False).encode('utf-8')
                col1.download_button(
                    "Download CSV",
                    csv,
                    "financial_data.csv",
                    "text/csv",
                    key='download-csv'
                )

                # Excel Export
                excel_buffer = io.BytesIO()
                st.session_state.filtered_data.to_excel(excel_buffer, index=False)
                excel_data = excel_buffer.getvalue()
                col2.download_button(
                    "Download Excel",
                    excel_data,
                    "financial_data.xlsx",
                    "application/vnd.ms-excel",
                    key='download-excel'
                )

                # JSON Export
                json_data = st.session_state.filtered_data.to_json(orient="records")
                col3.download_button(
                    "Download JSON",
                    json_data,
                    "financial_data.json",
                    "application/json",
                    key='download-json'
                )
            # Tab 2: Advanced Analytics
with tab2:
    if st.session_state.data is not None and not st.session_state.data.empty:
        st.subheader("Advanced Analytics")
        
        # Correlation Heatmap
        if "Symbol" in st.session_state.data.columns:
            st.write("### Correlation Heatmap")
            symbols_for_heatmap = st.multiselect(
                "Select Symbols for Correlation Analysis",
                options=st.session_state.data["Symbol"].unique(),
                default=st.session_state.data["Symbol"].unique(),
                key="heatmap_symbols"
            )
            filtered_data_for_heatmap = st.session_state.data[st.session_state.data["Symbol"].isin(symbols_for_heatmap)]
            if not filtered_data_for_heatmap.empty:
                create_correlation_heatmap(filtered_data_for_heatmap, st)

        # Statistical Analysis
        st.markdown("---")
        st.write("### Statistical Summary")
        if "Symbol" in st.session_state.data.columns:
            selected_symbols = st.multiselect(
                "Select Symbols for Statistical Analysis",
                options=st.session_state.data["Symbol"].unique(),
                default=st.session_state.data["Symbol"].unique(),
                key="stats_symbols"
            )
            stats_data = st.session_state.data[st.session_state.data["Symbol"].isin(selected_symbols)]
            if not stats_data.empty:
                st.write(stats_data.describe())

        # PCA Analysis
        st.markdown("---")
        st.write("### Principal Component Analysis (PCA)")
        try:
            if "Close" in st.session_state.data.columns:
                pivot_data = st.session_state.data.pivot_table(
                    index="Date",
                    columns="Symbol",
                    values="Close"
                ).fillna(0)

                if pivot_data.shape[1] >= 2:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pivot_data)

                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(scaled_data)
                    pca_df = pd.DataFrame(
                        pca_result,
                        columns=["Principal Component 1", "Principal Component 2"]
                    )
                    pca_df["Date"] = pivot_data.index

                    fig_pca = px.scatter(
                        pca_df,
                        x="Principal Component 1",
                        y="Principal Component 2",
                        hover_data=["Date"],
                        title="PCA Scatter Plot"
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                else:
                    st.warning("PCA requires at least two symbols. Please select more data.")
        except Exception as e:
            st.error(f"An error occurred during PCA: {e}")

        # K-Means Clustering
        st.markdown("---")
        st.write("### K-Means Clustering")
        if "Close" in st.session_state.data.columns and "Volume" in st.session_state.data.columns:
            num_clusters = st.slider(
                "Select Number of Clusters",
                min_value=2,
                max_value=6,
                value=3,
                key="kmeans_clusters"
            )
            try:
                cluster_data = perform_asset_clustering(st.session_state.data, n_clusters=num_clusters)
                if cluster_data is not None:
                    fig_cluster = px.scatter(
                        cluster_data,
                        x="Close",
                        y="Volume",
                        color="Cluster",
                        hover_data=["Symbol"],
                        title=f"Asset Clustering (K={num_clusters})"
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
            except Exception as e:
                st.error(f"Clustering error: {e}")

        # Linear Regression Analysis
        st.markdown("---")
        st.write("### Linear Regression Analysis")
        if "Close" in st.session_state.data.columns and "Symbol" in st.session_state.data.columns:
            selected_symbol_lr = st.selectbox(
                "Select Symbol for Linear Regression",
                options=st.session_state.data["Symbol"].unique(),
                key="lr_symbol"
            )
            
            lr_data = st.session_state.data[st.session_state.data["Symbol"] == selected_symbol_lr].copy()
            if not lr_data.empty:
                # Prepara i dati per la regressione
                lr_data = lr_data.sort_values("Date")
                lr_data["Days"] = (lr_data["Date"] - lr_data["Date"].min()).dt.days
                
                # Fit del modello
                lr = LinearRegression()
                X = lr_data[["Days"]]
                y = lr_data["Close"]
                lr.fit(X, y)
                
                # Predizioni
                lr_data["Predicted"] = lr.predict(X)
                
                # Calcola R-squared e altri metri
                r2_score = lr.score(X, y)
                
                # Mostra metriche
                col1, col2 = st.columns(2)
                col1.metric("R-squared", f"{r2_score:.4f}")
                col2.metric("Slope", f"{lr.coef_[0]:.4f}")
                
                # Plot
                fig_lr = go.Figure()
                
                # Dati reali
                fig_lr.add_trace(
                    go.Scatter(
                        x=lr_data["Date"],
                        y=lr_data["Close"],
                        mode="markers+lines",
                        name="Actual Price",
                        line=dict(color="blue")
                    )
                )
                
                # Linea di regressione
                fig_lr.add_trace(
                    go.Scatter(
                        x=lr_data["Date"],
                        y=lr_data["Predicted"],
                        mode="lines",
                        name="Regression Line",
                        line=dict(color="red")
                    )
                )
                
                fig_lr.update_layout(
                    title=f"Linear Regression Analysis for {selected_symbol_lr}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    showlegend=True
                )
                
                st.plotly_chart(fig_lr, use_container_width=True)
                
                # Mostra previsioni future
                days_ahead = st.slider(
                    "Predict days ahead",
                    min_value=1,
                    max_value=30,
                    value=7
                )
                
                last_day = lr_data["Days"].max()
                future_days = np.array([[day] for day in range(last_day + 1, last_day + days_ahead + 1)])
                future_predictions = lr.predict(future_days)
                
                st.write("### Future Price Predictions")
                future_dates = pd.date_range(
                    start=lr_data["Date"].max() + pd.Timedelta(days=1),
                    periods=days_ahead
                )
                
                predictions_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                
                st.dataframe(predictions_df)
    else:
        st.info("Please upload data in the Main Dashboard tab first.")

# Tab 3: Technical Analysis
with tab3:
    if st.session_state.data is not None and not st.session_state.data.empty:
        st.subheader("Technical Analysis")
        
        # Technical Indicators Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.number_input("RSI Period", min_value=1, value=14)
        with col2:
            macd_fast = st.number_input("MACD Fast Period", min_value=1, value=12)
        with col3:
            bb_period = st.number_input("Bollinger Band Period", min_value=1, value=20)

        # Symbol Selection for Technical Analysis
        if "Symbol" in st.session_state.data.columns:
            selected_symbol = st.selectbox(
                "Select Symbol for Technical Analysis",
                options=st.session_state.data["Symbol"].unique()
            )
            symbol_data = st.session_state.data[st.session_state.data["Symbol"] == selected_symbol].copy()
        else:
            symbol_data = st.session_state.data.copy()

        if "Close" in symbol_data.columns:
            # Calculate indicators
            symbol_data["RSI"] = calculate_rsi(symbol_data["Close"], rsi_period)
            symbol_data["MACD"], symbol_data["Signal"] = calculate_macd(symbol_data["Close"])
            symbol_data["BB_Upper"], symbol_data["BB_Lower"] = calculate_bollinger_bands(symbol_data["Close"], bb_period)

            # RSI Plot
            st.write("### Relative Strength Index (RSI)")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=symbol_data["Date"], y=symbol_data["RSI"], name="RSI"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI Indicator")
            st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD Plot
            st.write("### Moving Average Convergence Divergence (MACD)")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=symbol_data["Date"], y=symbol_data["MACD"], name="MACD"))
            fig_macd.add_trace(go.Scatter(x=symbol_data["Date"], y=symbol_data["Signal"], name="Signal"))
            fig_macd.update_layout(title="MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

            # Bollinger Bands
            st.write("### Bollinger Bands")
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=symbol_data["Date"], y=symbol_data["Close"], name="Price"))
            fig_bb.add_trace(go.Scatter(x=symbol_data["Date"], y=symbol_data["BB_Upper"], name="Upper Band"))
            fig_bb.add_trace(go.Scatter(x=symbol_data["Date"], y=symbol_data["BB_Lower"], name="Lower Band"))
            fig_bb.update_layout(title="Bollinger Bands")
            st.plotly_chart(fig_bb, use_container_width=True)
    else:
        st.info("Please upload data in the Main Dashboard tab first.")
        
        # Tab 4: Risk Analytics
with tab4:
    if st.session_state.data is not None and not st.session_state.data.empty:
        st.subheader("Risk Analysis")
        
        # Symbol Selection for Risk Analysis
        if "Symbol" in st.session_state.data.columns:
            selected_symbol_risk = st.selectbox(
                "Select Symbol for Risk Analysis",
                options=st.session_state.data["Symbol"].unique(),
                key="risk_symbol"
            )
            risk_data = st.session_state.data[st.session_state.data["Symbol"] == selected_symbol_risk].copy()
        else:
            risk_data = st.session_state.data.copy()

        risk_metrics = calculate_risk_metrics(risk_data)

        # Risk Metrics Display
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Value at Risk (95%)",
            f"{risk_metrics.get('Value at Risk (95%)', 0):.2%}"
        )
        col2.metric(
            "Expected Shortfall",
            f"{risk_metrics.get('Expected Shortfall', 0):.2%}"
        )
        col3.metric(
            "Maximum Drawdown",
            f"{risk_metrics.get('Maximum Drawdown', 0):.2%}"
        )

        col1, col2 = st.columns(2)
        col1.metric(
            "Kurtosis",
            f"{risk_metrics.get('Kurtosis', 0):.2f}"
        )
        col2.metric(
            "Skewness",
            f"{risk_metrics.get('Skewness', 0):.2f}"
        )

        # Drawdown Analysis
        st.markdown("---")
        st.write("### Drawdown Analysis")
        if "Close" in risk_data.columns:
            peak = risk_data["Close"].expanding(min_periods=1).max()
            drawdown = (risk_data["Close"] - peak) / peak

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=risk_data["Date"],
                y=drawdown,
                fill="tozeroy",
                name="Drawdown"
            ))
            fig_dd.update_layout(
                title="Historical Drawdown",
                yaxis_title="Drawdown (%)",
                xaxis_title="Date"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # Returns Distribution
        st.markdown("---")
        st.write("### Returns Distribution")
        if "Close" in risk_data.columns:
            returns = risk_data["Close"].pct_change().dropna()
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name="Returns Distribution"
            ))
            fig_dist.update_layout(
                title="Returns Distribution",
                xaxis_title="Daily Returns",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# Tab 5: Portfolio
with tab5:
    if st.session_state.data is not None and not st.session_state.data.empty:
        st.subheader("Portfolio Analysis")

        if "Symbol" in st.session_state.data.columns:
            # Clean data for portfolio analysis
            clean_data = st.session_state.data.drop_duplicates(subset=["Date", "Symbol"]).copy()
            clean_data["Date"] = pd.to_datetime(clean_data["Date"])

            # Portfolio Asset Selection
            selected_assets = st.multiselect(
                "Select Assets for Portfolio Analysis",
                options=clean_data["Symbol"].unique(),
                default=clean_data["Symbol"].unique()
            )

            if selected_assets:
                portfolio_data = clean_data[clean_data["Symbol"].isin(selected_assets)]

                # Calculate portfolio metrics
                pivot_data = portfolio_data.pivot(
                    index="Date",
                    columns="Symbol",
                    values="Close"
                )
                returns_df = pivot_data.pct_change()
                correlation = returns_df.corr()
                annual_returns = (1 + returns_df.mean()) ** 252 - 1
                annual_vol = returns_df.std() * np.sqrt(252)

                # Correlation Matrix
                st.write("### Asset Correlation")
                fig_corr = px.imshow(
                    correlation,
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

                # Returns and Volatility
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Annual Returns")
                    st.dataframe(annual_returns.round(4))
                with col2:
                    st.write("### Annual Volatility")
                    st.dataframe(annual_vol.round(4))

                # Cumulative Returns
                st.write("### Cumulative Returns")
                cum_returns = (1 + returns_df).cumprod()
                fig_cum = go.Figure()
                for col in cum_returns.columns:
                    fig_cum.add_trace(
                        go.Scatter(x=cum_returns.index, y=cum_returns[col], name=col)
                    )
                fig_cum.update_layout(
                    title="Cumulative Returns by Asset",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return"
                )
                st.plotly_chart(fig_cum, use_container_width=True)

                # Risk-Return Scatter Plot
                st.write("### Risk-Return Profile")
                risk_return_df = pd.DataFrame({
                    'Return': annual_returns,
                    'Risk': annual_vol,
                    'Symbol': annual_returns.index
                })
                fig_scatter = px.scatter(
                    risk_return_df,
                    x='Risk',
                    y='Return',
                    text='Symbol',
                    title="Risk-Return Profile"
                )
                fig_scatter.update_traces(textposition='top center')
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Please select at least one asset for portfolio analysis.")
        else:
            st.warning("The data doesn't contain multiple symbols for portfolio analysis.")
    else:
        st.info("Please upload data in the Main Dashboard tab first.")
        
        # Tab 6: Settings
with tab6:
    st.subheader("Dashboard Settings")

    # Chart Settings
    st.write("### Chart Settings")
    col1, col2 = st.columns(2)
    with col1:
        chart_theme = st.selectbox(
            "Chart Theme",
            ["plotly", "plotly_white", "plotly_dark"],
            key="chart_theme"
        )
    with col2:
        chart_height = st.number_input(
            "Chart Height (px)",
            min_value=300,
            max_value=1000,
            value=500,
            step=50
        )

    # Analysis Settings
    st.write("### Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        anomaly_threshold = st.slider(
            "Anomaly Detection Threshold",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1
        )
    with col2:
        rolling_window = st.slider(
            "Rolling Window Size",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )

    # Data Processing Settings
    st.write("### Data Processing Settings")
    col1, col2 = st.columns(2)
    with col1:
        handle_missing = st.selectbox(
            "Handle Missing Values",
            ["Drop", "Forward Fill", "Backward Fill", "Linear Interpolation"]
        )
    with col2:
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["Z-Score", "IQR", "None"]
        )

    # Export Settings
    st.write("### Export Settings")
    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox(
            "Default Export Format",
            ["CSV", "Excel", "JSON"]
        )
    with col2:
        date_format = st.selectbox(
            "Date Format",
            ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"]
        )

    # Save Settings Button
    if st.button("Save Settings"):
        # Store settings in session state
        st.session_state.settings = {
            "chart_theme": chart_theme,
            "chart_height": chart_height,
            "anomaly_threshold": anomaly_threshold,
            "rolling_window": rolling_window,
            "handle_missing": handle_missing,
            "outlier_method": outlier_method,
            "export_format": export_format,
            "date_format": date_format
        }
        st.success("Settings saved successfully!")

# Tab 7: API Integration
with tab7:
    st.header("API Data Integration")

    # API Authentication
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    st.subheader("API Authentication")
    api_key_input = st.text_input("API Key", type="password", help="Enter your API key")
    api_secret_input = st.text_input("API Secret", type="password", help="Enter your API secret")

    if st.button("Authenticate"):
        if api_key_input == API_KEY and api_secret_input == API_SECRET:
            st.session_state.authenticated = True
            st.success("Authentication successful!")
        else:
            st.session_state.authenticated = False
            st.error("Authentication failed. Please check your API Key and Secret.")

    # API Integration Features
    if st.session_state.authenticated:
        st.subheader("Available Endpoints")
        endpoints = {
            "Market Summary": "http://localhost:5000/market-summary",
            "Technical Indicators": "http://localhost:5000/technical-indicators/",
            "Historical Data": "http://localhost:5000/historical-data/",
            "Generate Data": "http://localhost:5000/generate-finance-data",
        }

        # Endpoint Selection
        selected_endpoint = st.selectbox("Select Endpoint", list(endpoints.keys()))

        # Dynamic Parameter Inputs
        params = {}
        if selected_endpoint in ["Technical Indicators", "Historical Data"]:
            symbol = st.selectbox(
                "Select Symbol",
                ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
            )
            params["symbol"] = symbol

        if selected_endpoint == "Historical Data":
            period = st.select_slider(
                "Select Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y"]
            )
            params["period"] = period

        if selected_endpoint == "Generate Data":
            num_records = st.slider(
                "Number of Records",
                min_value=10,
                max_value=1000,
                value=100
            )
            params["num_records"] = num_records

        # Auto-Refresh Settings
        st.subheader("Auto-Refresh Settings")
        enable_auto_refresh = st.checkbox("Enable Auto-Refresh")
        if enable_auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=300,
                value=60
            )

        # Fetch Data Button
        if st.button("Fetch Data from API"):
            with st.spinner("Fetching data from API..."):
                try:
                    # Construct URL with parameters
                    base_url = endpoints[selected_endpoint]
                    if selected_endpoint in ["Technical Indicators", "Historical Data"]:
                        url = f"{base_url}?symbol={params['symbol']}"
                        if "period" in params:
                            url += f"&period={params['period']}"
                    elif selected_endpoint == "Generate Data":
                        url = f"{base_url}?num_records={params['num_records']}"
                    else:
                        url = base_url

                    headers = {
                        "X-API-Key": api_key_input,
                        "X-API-Secret": api_secret_input
                    }

                    # Make API Request
                    response = requests.get(url, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Data fetched successfully!")
                        
                        # Convert to DataFrame
                        if "data" in data:
                            df = pd.DataFrame(data["data"])
                        else:
                            df = pd.DataFrame(data)
                        
                        # Display data
                        st.write("### API Response Data")
                        st.dataframe(df)
                        
                        # Export options
                        st.write("### Export Options")
                        col1, col2, col3 = st.columns(3)
                        
                        # CSV Export
                        csv = df.to_csv(index=False).encode('utf-8')
                        col1.download_button(
                            "Download CSV",
                            csv,
                            "api_data.csv",
                            "text/csv"
                        )
                        
                        # Excel Export
                        buffer = io.BytesIO()
                        df.to_excel(buffer, index=False)
                        col2.download_button(
                            "Download Excel",
                            buffer.getvalue(),
                            "api_data.xlsx",
                            "application/vnd.ms-excel"
                        )
                        
                        # JSON Export
                        col3.download_button(
                            "Download JSON",
                            df.to_json(orient="records"),
                            "api_data.json",
                            "application/json"
                        )
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")

        # Auto-refresh logic
        if enable_auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()

# Footer
st.markdown("""
---
Created with ❤️ using Streamlit | Last updated: {}
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))