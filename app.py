
import pandas as pd
import streamlit as st
import plotly.express as px
from ai_assistant import answer_question
from anomaly_detection import detect_anomalies

# Configure dashboard
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Agent")
st.write("Analyze sales data using Python and AI tools.")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload your sales CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()

    # Validate required columns
    required = {"date", "product", "category",
                "quantity", "price"}

    if not required.issubset(df.columns):
        st.error(
            "CSV must contain: date, product, "
            "category, quantity, price"
        )
        st.stop()

    # Clean data
    df = df.drop_duplicates().copy()
    df["quantity"] = pd.to_numeric(
        df["quantity"], errors="coerce"
    )
    df["price"] = pd.to_numeric(
        df["price"], errors="coerce"
    )
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df = df.dropna(
        subset=["date", "product", "quantity", "price"]
    )

    if df.empty:
        st.warning("No valid sales records found.")
        st.stop()

    # Calculate revenue
    df["revenue"] = df["quantity"] * df["price"]

    total_revenue = df["revenue"].sum()
    total_orders = len(df)
    total_quantity = df["quantity"].sum()

    # Dashboard statistics
    st.subheader("Sales Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Revenue",
        f"₹{total_revenue:,.2f}"
    )
    col2.metric("Total Records", total_orders)
    col3.metric(
        "Units Sold",
        f"{total_quantity:,.0f}"
    )

    # Data preview
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Product revenue
    product_sales = (
        df.groupby("product", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    st.subheader("Revenue by Product")

    fig1 = px.bar(
        product_sales,
        x="product",
        y="revenue",
        color="product",
        title="Product Revenue Comparison"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Daily revenue
    daily_sales = (
        df.groupby("date", as_index=False)["revenue"]
        .sum()
        .sort_values("date")
    )

    st.subheader("Daily Revenue Trend")

    fig2 = px.line(
        daily_sales,
        x="date",
        y="revenue",
        markers=True,
        title="Daily Sales Trend"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Category distribution
    category_sales = (
        df.groupby("category", as_index=False)["revenue"]
        .sum()
    )

    st.subheader("Revenue by Category")

    fig3 = px.pie(
        category_sales,
        names="category",
        values="revenue",
        title="Revenue Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Export data
    st.subheader("Download Cleaned Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name="sales_report.csv",
        mime="text/csv"
    )


    # AI Assistant
    st.divider()
    st.subheader("🤖 Ask Your Data")

    question = st.text_input(
        "Ask a question about your dataset",
        placeholder="What is the total revenue?"
    )

    if st.button("Ask Assistant"):
        if question.strip():
            answer = answer_question(df, question)
            st.info(answer)
        else:
            st.warning("Please enter a question.")

    # Machine Learning Anomaly Detection
    st.divider()
    st.subheader("🧠 ML Anomaly Detection")

    st.write(
        "Detect unusual sales transactions "
        "using the Isolation Forest algorithm."
    )

    if st.button("Detect Anomalies"):
        try:
            results = detect_anomalies(df)

            anomaly_count = (
                results["anomaly"] == "Anomaly"
            ).sum()

            st.metric(
                "Anomalies Detected",
                int(anomaly_count)
            )

            st.dataframe(
                results,
                width="stretch"
            )
            

            st.download_button(
                "Download Anomaly Report",
                data=results.to_csv(index=False),
                file_name="anomaly_report.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(str(error))


else:
    st.info("Upload sample_sales.csv to begin.")
