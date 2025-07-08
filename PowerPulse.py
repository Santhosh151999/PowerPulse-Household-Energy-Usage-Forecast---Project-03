# PowerPulse Streamlit App with Optimized Performance and 3 Pages
import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestRegressor

# -------------------- Set Page Config (MUST be at the top) --------------------
st.set_page_config(page_title="PowerPulse Dashboard", layout="wide")

# -------------------- Caching DB Load --------------------
@st.cache_data(show_spinner=False)
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Dark2020@",
        database="PowerPulse"
    )
    df = pd.read_sql("SELECT * FROM energy_data", conn)
    conn.close()
    return df

df = load_data()

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("🔘 Navigation")
page = st.sidebar.radio("Go to", ["📋 Project Summary", "📊 Energy Dashboard", "🤖 Predict Energy Usage"])

# -------------------- 📋 Project Summary --------------------
if page == "📋 Project Summary":
    st.title("📋 PowerPulse: Project Summary")

    st.markdown("""
    ## 🧠 Problem Statement
    This project builds a predictive model to forecast power usage, enabling better decisions for both consumers and energy providers.
    """)

    st.subheader("🧭 Business Use Cases")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🏠 Energy Management for Households")
        st.success("📈 Demand Forecasting for Providers")
        st.warning("🚨 Anomaly Detection")
    with col2:
        st.info("⚙️ Smart Grid Integration")
        st.success("🌱 Environmental Impact")

    st.subheader("🔧 Project Approach")
    st.markdown("""
    - EDA: Trends, outliers
    - Preprocessing: Feature engineering
    - Modeling: Linear & Random Forest (best)
    - Evaluation: RMSE, MAE, R²
    """)

    st.subheader("✅ Model Comparison")
    st.table(pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest (Best)"],
        "RMSE": [0.0403, 0.028],
        "MAE": [0.0258, 0.015],
        "R² Score": [0.9986, 0.9993]
    }))

    st.subheader("💻 Tech Stack")
    st.markdown("Python • Pandas • Scikit-learn • Streamlit • MySQL • Plotly")

# -------------------- 📊 Energy Dashboard --------------------
elif page == "📊 Energy Dashboard":
    st.title("📊 PowerPulse: Household Energy Usage Dashboard")

    st.sidebar.header("🔍 Filters")
    selected_month = st.sidebar.selectbox("Select Month", sorted(df['month'].unique()))
    filtered_df = df[df['month'] == selected_month]

    st.subheader(f"📅 Monthly Summary: Month {selected_month}")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔌 Active Power", f"{filtered_df['Global_active_power'].sum():.2f} kW")
    col2.metric("⚡ Reactive Power", f"{filtered_df['Global_reactive_power'].sum():.2f} kW")
    col3.metric("🔋 Avg Voltage", f"{filtered_df['Voltage'].mean():.2f} V")

    # Hourly Usage
    st.subheader("📈 Hourly Energy Usage")
    hourly_avg = filtered_df.groupby('hour')['Global_active_power'].mean().reset_index()
    st.plotly_chart(px.line(hourly_avg, x='hour', y='Global_active_power', markers=True), use_container_width=True)

    # Daily Usage
    st.subheader("📊 Daily Usage Trend")
    daily_usage = filtered_df.groupby('day')['Global_active_power'].sum().reset_index()
    st.plotly_chart(px.line(daily_usage, x='day', y='Global_active_power', markers=True), use_container_width=True)

    # Weekend vs Weekday
    st.subheader("📆 Weekday vs Weekend")
    weekend = filtered_df.groupby('is_weekend')['Global_active_power'].mean().reset_index()
    weekend['Type'] = weekend['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    st.plotly_chart(px.bar(weekend, x='Type', y='Global_active_power', color='Type'), use_container_width=True)

    # Outliers
    st.subheader("🚨 Anomaly Detection")
    st.plotly_chart(px.box(filtered_df, y='Global_active_power'), use_container_width=True)

    # Top Usage
    st.subheader("📌 Top 5 Usage Points")
    top5 = filtered_df.sort_values(by='Global_active_power', ascending=False).head(5)
    st.dataframe(top5[['day', 'month', 'hour', 'Global_active_power']])

    # Pie Chart
    st.subheader("🌱 Energy by Appliance")
    pie_df = pd.DataFrame({
        "Category": ["Kitchen", "Laundry", "AC & Heater"],
        "Energy": [
            filtered_df['Sub_metering_1'].sum(),
            filtered_df['Sub_metering_2'].sum(),
            filtered_df['Sub_metering_3'].sum()
        ]
    })
    st.plotly_chart(px.pie(pie_df, names="Category", values="Energy"), use_container_width=True)

    # Heatmap
    st.subheader("🧭 Heatmap (Hour vs Weekday)")
    heatmap = filtered_df.groupby(['weekday', 'hour'])['Global_active_power'].mean().unstack()
    st.dataframe(heatmap.style.background_gradient(cmap='YlOrRd'))

# -------------------- 🤖 Model Prediction Page --------------------
elif page == "🤖 Predict Energy Usage":
    st.title("🤖 Predict Household Energy Consumption")

    st.markdown("Enter feature values to predict Global Active Power usage")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            Voltage = st.number_input("Voltage (V)", value=240.0)
            Global_intensity = st.number_input("Global Intensity (A)", value=5.0)
            Sub_metering_1 = st.number_input("Kitchen Meter (W-h)", value=1.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            is_weekend = st.selectbox("Weekend?", [0, 1])
        with col2:
            Global_reactive_power = st.number_input("Reactive Power (kW)", value=0.1)
            Sub_metering_2 = st.number_input("Laundry Meter (W-h)", value=1.0)
            Sub_metering_3 = st.number_input("AC/Heater Meter (W-h)", value=6.0)
            weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6, 2)

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            'Global_reactive_power': Global_reactive_power,
            'Voltage': Voltage,
            'Global_intensity': Global_intensity,
            'Sub_metering_1': Sub_metering_1,
            'Sub_metering_2': Sub_metering_2,
            'Sub_metering_3': Sub_metering_3,
            'hour': hour,
            'weekday': weekday,
            'is_weekend': is_weekend
        }])

        # Load Model
        model = joblib.load("/Users/santhoshms/Desktop/Python/PowerPulse/powerpulse_model.pkl")
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Global Active Power: {prediction:.4f} kW")

        

st.markdown("---")
st.caption("🚀 PowerPulse • Streamlit Dashboard • Santhosh")
