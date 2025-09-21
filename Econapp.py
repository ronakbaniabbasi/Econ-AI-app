import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# =====================
# 1. DATA PREPARATION
# =====================
countries = ['USA', 'China', 'UK', 'Russia', 'UAE', 'Iran', 'Germany', 'India', 'Brazil']
np.random.seed(42)

data = {
    'Year': np.random.choice(range(2010, 2023), 1000),
    'Country': np.random.choice(countries, 1000),
    # Traditional factors
    'Tech_Investment': np.random.uniform(0.5, 5, 1000),
    'Education_Spending': np.random.uniform(1, 8, 1000),
    'Environmental_Protection': np.random.uniform(30, 95, 1000),
    # Modern factors
    'Urbanization_Rate': np.random.uniform(30, 95, 1000),
    'Trade_Openness': np.random.uniform(10, 150, 1000),  # Globalization proxy
    'Individualism_Index': np.random.uniform(10, 90, 1000),  # Hofstede scale
    # Outcomes
    'GDP_Growth': np.random.normal(3, 1.5, 1000)
}

df = pd.DataFrame(data)

# Add realistic relationships
df['GDP_Growth'] = (
    df['GDP_Growth']
    + 0.12 * df['Tech_Investment']
    + 0.08 * df['Education_Spending']
    + 0.05 * df['Environmental_Protection']
    + 0.07 * df['Urbanization_Rate']
    + 0.04 * df['Trade_Openness']
    + 0.03 * df['Individualism_Index']
)

# Country-specific adjustments
adjustments = {
    'USA': {'Individualism_Index': +15, 'Tech_Investment': +2},
    'UAE': {'Environmental_Protection': -10, 'Trade_Openness': +40},
    'Iran': {'Trade_Openness': -30, 'Tech_Investment': -1},
    'Russia': {'Individualism_Index': -10, 'Education_Spending': -0.5}
}

for country, adj in adjustments.items():
    mask = df['Country'] == country
    for col, val in adj.items():
        df.loc[mask, col] += val

# =====================
# 2. MODEL TRAINING
# =====================
X = pd.get_dummies(df.drop(['GDP_Growth', 'Country'], axis=1))
y = df['GDP_Growth']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# =====================
# 3. STREAMLIT APP
# =====================
st.set_page_config(page_title="EconAI Simulator", layout="wide")
st.title("🌍 Global Economic Prosperity Simulator")
st.caption("Done by Ronak Baniabbasi | May 5, 2025")

# Sidebar inputs
st.sidebar.header("Simulation Settings")
country = st.sidebar.selectbox("Select Country:", countries, index=1)
years = st.sidebar.slider("Projection Years:", 1, 10, 5)

st.sidebar.markdown("### Traditional Factors")
tech = st.sidebar.slider("Technology Investment Change (%)", -10, 50, 0)
edu = st.sidebar.slider("Education Spending Change (%)", -10, 30, 0)
env = st.sidebar.slider("Environmental Protection Change (%)", -20, 40, 0)

st.sidebar.markdown("### Modern Factors")
urban = st.sidebar.slider("Urbanization Rate Change (%)", -5, 25, 0)
global_ = st.sidebar.slider("Globalization (Trade Openness) Change (%)", -30, 40, 0)
individual = st.sidebar.slider("Individualism Index Change (%)", -10, 20, 0)

# =====================
# 4. SIMULATION
# =====================
scale = 1 + (years / 10)
inputs = {
    'Tech_Investment': tech * scale,
    'Education_Spending': edu * scale,
    'Environmental_Protection': env * scale,
    'Urbanization_Rate': urban * scale,
    'Trade_Openness': global_ * scale,
    'Individualism_Index': individual * scale,
    'Year': 2023 + years
}

input_df = pd.DataFrame([inputs])
input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
gdp_change = model.predict(input_df)[0] - df[df['Country'] == country]['GDP_Growth'].mean()

# =====================
# 5. OUTPUT
# =====================
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📈 Projected GDP Change in {years} Years for {country}:")
    st.markdown(
        f"<h2 style='color:{'green' if gdp_change > 0 else 'red'}'>{gdp_change:.2f}%</h2>",
        unsafe_allow_html=True
    )

with col2:
    st.subheader("🔎 Factor Changes")
    st.write(f"- Technology: {tech}%")
    st.write(f"- Education: {edu}%")
    st.write(f"- Environment: {env}%")
    st.write(f"- Urbanization: {urban}%")
    st.write(f"- Globalization: {global_}%")
    st.write(f"- Individualism: {individual}%")

# Feature importance
importance = pd.DataFrame({'Factor': X.columns, 'Importance': model.feature_importances_})
importance_fig = px.bar(importance, x='Factor', y='Importance', title='Factor Importance for GDP Growth')
st.plotly_chart(importance_fig, use_container_width=True)

# Country comparison
country_comparison = px.line_polar(
    df.groupby('Country').mean().reset_index(),
    r='GDP_Growth',
    theta='Country',
    title='GDP Growth Comparison'
)
st.plotly_chart(country_comparison, use_container_width=True)

# Trade-offs
tradeoff_fig = px.scatter(
    df,
    x='Individualism_Index',
    y='Environmental_Protection',
    color='GDP_Growth',
    hover_name='Country',
    title='Individualism vs Environmental Protection Trade-offs'
)
st.plotly_chart(tradeoff_fig, use_container_width=True)

