import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import json

# Configurazione pagina
st.set_page_config(page_title="Finance API Dashboard", layout="wide")

# Stile CSS personalizzato
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Titolo principale
st.title("Finance API Dashboard")

# Creazione tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Data", "Technical Indicators", "Historical Data", "API Settings"])

# Tab Market Data
with tab1:
    st.header("Market Summary")
    if st.button("Fetch Market Data"):
        try:
            response = requests.get("http://localhost:5000/market-summary")
            if response.status_code == 200:
                data = response.json()
                
                # Creazione della tabella dei dati
                df = pd.DataFrame([
                    {
                        "Symbol": symbol,
                        "Price": details["current_price"],
                        "Change %": details["change_percent"],
                        "Volume": details["volume"]
                    }
                    for symbol, details in data.items()
                ])
                
                # Visualizzazione con colori condizionali
                st.dataframe(
                    df.style.format({
                        "Price": "${:.2f}",
                        "Change %": "{:.2f}%",
                        "Volume": "{:,.0f}"
                    }).background_gradient(subset=["Change %"], cmap="RdYlGn")
                )
                
                # Grafico delle variazioni percentuali
                fig = go.Figure(data=[
                    go.Bar(
                        x=df["Symbol"],
                        y=df["Change %"],
                        marker_color=['red' if x < 0 else 'green' for x in df["Change %"]]
                    )
                ])
                fig.update_layout(title="Daily Price Changes (%)")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")

# Tab Technical Indicators
with tab2:
    st.header("Technical Analysis")
    symbol = st.selectbox("Select Symbol", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
    
    if st.button("Get Technical Indicators"):
        try:
            response = requests.get(f"http://localhost:5000/technical-indicators/{symbol}")
            if response.status_code == 200:
                data = response.json()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("RSI", f"{data['RSI']:.2f}")
                col2.metric("MACD", f"{data['MACD']:.2f}")
                col3.metric("Current Price", f"${data['current_price']:.2f}")
                
                # Bande di Bollinger
                st.subheader("Bollinger Bands")
                bb_data = pd.DataFrame({
                    "Upper": [data["BB_upper"]],
                    "Lower": [data["BB_lower"]],
                    "Price": [data["current_price"]]
                })
                st.dataframe(bb_data.style.format("${:.2f}"))
                
        except Exception as e:
            st.error(f"Error fetching technical indicators: {str(e)}")

# Tab Historical Data
with tab3:
    st.header("Historical Data")
    symbol = st.selectbox("Select Symbol", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"], key="hist_symbol")
    period = st.select_slider("Select Period", options=["1d", "5d", "1mo", "3mo", "6mo", "1y"])
    
    if st.button("Get Historical Data"):
        try:
            response = requests.get(f"http://localhost:5000/historical-data/{symbol}?period={period}")
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data["data"])
                
                # Candlestick chart
                fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                )])
                fig.update_layout(title=f"{symbol} Price History")
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure(data=[go.Bar(x=df['Date'], y=df['Volume'])])
                fig_volume.update_layout(title=f"{symbol} Trading Volume")
                st.plotly_chart(fig_volume, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error fetching historical data: {str(e)}")

# Tab API Settings
with tab4:
    st.header("API Configuration")
    api_url = st.text_input("API Base URL", "http://localhost:5000")
    st.info("Available endpoints:\n- /market-summary\n- /technical-indicators/{symbol}\n- /historical-data/{symbol}")

# Footer
st.markdown("---")
st.markdown("Data provided by Finance API | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))