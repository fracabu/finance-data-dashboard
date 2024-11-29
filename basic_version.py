import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime

# Configurazione della pagina
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Titolo della dashboard
st.title("Sales Dashboard")


def calculate_basic_metrics(data):
    metrics = {}

    if "Sales" in data.columns:
        metrics["total_sales"] = data["Sales"].sum()
        metrics["avg_sales"] = data["Sales"].mean()

    if "Profit" in data.columns:
        metrics["total_profit"] = data["Profit"].sum()

    return metrics


def create_basic_visualizations(data):
    # Sales Trend
    st.subheader("Sales Trend")
    daily_sales = data.groupby("Date")["Sales"].sum().reset_index()
    fig_sales = px.line(daily_sales, x="Date", y="Sales", title="Daily Sales")
    st.plotly_chart(fig_sales, use_container_width=True)

    # Regional Performance
    st.subheader("Sales by Region")
    regional_sales = data.groupby("Region")["Sales"].sum().reset_index()
    fig_regions = px.bar(regional_sales, x="Region", y="Sales", title="Sales by Region")
    st.plotly_chart(fig_regions, use_container_width=True)

    # Top Products
    st.subheader("Top Products")
    product_sales = data.groupby("Product")["Sales"].sum().nlargest(5).reset_index()
    fig_products = px.bar(product_sales, x="Product", y="Sales", title="Top 5 Products")
    st.plotly_chart(fig_products, use_container_width=True)


# Main
st.subheader("Upload Data")
uploaded_file = st.file_uploader("Upload a file (CSV, Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"])

        # Data Preview
        st.write("Data Preview:")
        st.dataframe(data.head())

        # Base Filters
        st.sidebar.header("Filters")

        # Date Filter
        if "Date" in data.columns:
            min_date = data["Date"].min()
            max_date = data["Date"].max()
            date_range = st.sidebar.date_input(
                "Select Date Range", [min_date, max_date]
            )
            if len(date_range) == 2:
                data = data[
                    (data["Date"] >= pd.to_datetime(date_range[0]))
                    & (data["Date"] <= pd.to_datetime(date_range[1]))
                ]

        # Key Metrics
        st.header("Key Metrics")
        metrics = calculate_basic_metrics(data)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"${metrics['total_sales']:,.2f}")
        col2.metric("Total Profit", f"${metrics['total_profit']:,.2f}")
        col3.metric("Average Sales", f"${metrics['avg_sales']:,.2f}")

        # Basic Visualizations
        create_basic_visualizations(data)

        # Export
        st.subheader("Export Data")
        st.download_button(
            "Download CSV",
            data.to_csv(index=False).encode("utf-8"),
            "sales_data.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown(
    """
---
Created with ❤️ using Streamlit | Last updated: {}
""".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
)
