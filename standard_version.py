import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurazione della pagina
st.set_page_config(page_title="Sales Dashboard", layout="wide")

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
st.title("Sales Analytics Dashboard")

# Funzioni per metriche base
def calculate_metrics(data):
    metrics = {}
    
    if "Sales" in data.columns:
        # Metriche di vendita
        metrics['total_sales'] = data['Sales'].sum()
        metrics['avg_daily_sales'] = data.groupby('Date')['Sales'].sum().mean()
        
    if "Profit" in data.columns:
        # Metriche di profitto
        metrics['total_profit'] = data['Profit'].sum()
        metrics['profit_margin'] = (data['Profit'].sum() / data['Sales'].sum()) * 100
        
    if "Product" in data.columns:
        # Performance prodotti
        product_sales = data.groupby('Product')['Sales'].sum()
        metrics['top_products'] = product_sales.nlargest(5).index.tolist()
        
    return metrics

def create_visualizations(data, container):
    # 1. Overview Dashboard
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sales Trend', 'Sales by Region', 'Top Products', 'Monthly Sales')
    )
    
    # Sales Trend
    daily_sales = data.groupby('Date')['Sales'].sum().reset_index()
    fig1.add_trace(
        go.Scatter(x=daily_sales['Date'], y=daily_sales['Sales'], 
                  name='Daily Sales', mode='lines'),
        row=1, col=1
    )
    
    # Regional Performance
    regional_sales = data.groupby('Region')['Sales'].sum().reset_index()
    fig1.add_trace(
        go.Bar(x=regional_sales['Region'], y=regional_sales['Sales'], 
               name='Regional Sales'),
        row=1, col=2
    )
    
    # Top Products
    product_sales = data.groupby('Product')['Sales'].sum().nlargest(10).reset_index()
    fig1.add_trace(
        go.Bar(x=product_sales['Product'], y=product_sales['Sales'], 
               name='Product Sales'),
        row=2, col=1
    )
    
    # Monthly Sales
    data['Month'] = data['Date'].dt.strftime('%Y-%m')
    monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
    fig1.add_trace(
        go.Scatter(x=monthly_sales['Month'], y=monthly_sales['Sales'],
                  name='Monthly Sales', mode='lines+markers'),
        row=2, col=2
    )
    
    fig1.update_layout(height=800, showlegend=True)
    container.plotly_chart(fig1, use_container_width=True)

# Funzione per filtrare i dati
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

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "⚙️ Settings"])

# Tab 1: Main Dashboard
with tab1:
    st.subheader("Upload Data for Analysis")
    uploaded_file = st.file_uploader("Upload a file (CSV, Excel)", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Load data
        file_type = uploaded_file.name.split(".")[-1]
        if file_type == "csv":
            data = pd.read_csv(uploaded_file)
        elif file_type in ["xlsx", "xls"]:
            data = pd.read_excel(uploaded_file)
        
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

        # Preview
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Filters
        st.sidebar.header("Filters")
        filters = {}

        if "Date" in data.columns:
            min_date, max_date = data["Date"].min(), data["Date"].max()
            filters["Date"] = st.sidebar.date_input("Date Range", [min_date, max_date])

        if "Product" in data.columns:
            products = data["Product"].unique()
            filters["Product"] = st.sidebar.multiselect("Products", options=products, default=products)

        if "Region" in data.columns:
            regions = data["Region"].unique()
            filters["Region"] = st.sidebar.multiselect("Regions", options=regions, default=regions)

        # Apply filters
        filtered_data = filter_data(data, filters)

        # Key Metrics
        st.header("Key Metrics")
        metrics = calculate_metrics(filtered_data)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"${metrics.get('total_sales', 0):,.2f}")
        col2.metric("Total Profit", f"${metrics.get('total_profit', 0):,.2f}")
        col3.metric("Profit Margin", f"{metrics.get('profit_margin', 0):,.1f}%")
        col4.metric("Avg Daily Sales", f"${metrics.get('avg_daily_sales', 0):,.2f}")

        # Visualizations
        st.header("Sales Analytics")
        create_visualizations(filtered_data, st)

        # Export
        st.subheader("Export Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download CSV", 
                             filtered_data.to_csv(index=False).encode('utf-8'),
                             "sales_data.csv", 
                             "text/csv")
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                filtered_data.to_excel(writer, sheet_name='Sales Data', index=False)
            st.download_button("Download Excel",
                             buffer.getvalue(),
                             "sales_data.xlsx",
                             "application/vnd.ms-excel")

# Tab 2: Analytics
with tab2:
    if 'data' in locals():
        st.subheader("Sales Analysis")
        
        # Monthly Trends
        st.write("### Monthly Trends")
        # Creiamo monthly_sales qui
        data['Month'] = data['Date'].dt.strftime('%Y-%m')
        monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
        monthly_fig = px.line(monthly_sales, 
                            x='Month', 
                            y='Sales',
                            title='Monthly Sales Trend')
        st.plotly_chart(monthly_fig, use_container_width=True)

# Tab 3: Settings
with tab3:
    st.subheader("Dashboard Settings")
    
    # Visual Settings
    st.write("### Visual Settings")
    chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark"])
    
    # Export Settings
    st.write("### Export Settings")
    export_format = st.radio("Default Export Format", ["CSV", "Excel"])
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# Footer
st.markdown(
    """
---
Created with ❤️ using Streamlit | Last updated: {}
""".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
)
