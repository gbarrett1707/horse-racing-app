import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Horse Ability Viewer", layout="centered")

st.title("📊 Horse Ability & Trend Score Viewer")

# Define the data path to load automatically
DATA_PATH = os.path.join(os.path.dirname(__file__), "horse_ability_trend_scores_combined.csv")

# Cache loading function
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# Button to refresh data
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()

# Try loading the data
try:
    df = load_data()
    st.success("✅ Data loaded successfully.")

    # Search / Filter section
    with st.expander("🔍 Search / Filter"):
        horse_name = st.text_input("Filter by horse name:")
        if horse_name:
            filtered = df[df['Horse'].str.contains(horse_name, case=False)]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    st.divider()

    # Racecard builder
    st.subheader("📋 Racecard Builder")

    selected_horses = st.multiselect("Select horses to compare:", df['Horse'].unique())

    if selected_horses:
        racecard_df = df[df['Horse'].isin(selected_horses)].copy()
        racecard_df = racecard_df[['Horse', 'RaceType', 'CurrentAbility']]

        total_ability = racecard_df['CurrentAbility'].sum()
        racecard_df['WinChance (%)'] = (racecard_df['CurrentAbility'] / total_ability * 100).round(2)

        racecard_df = racecard_df.sort_values("WinChance (%)", ascending=False).reset_index(drop=True)

        st.write("### 🏇 Racecard Comparison")
        st.dataframe(racecard_df, use_container_width=True)

except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
