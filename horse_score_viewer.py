import streamlit as st
import pandas as pd

# Page setup
st.set_page_config(page_title="Horse Ability Viewer", layout="centered")
st.title("📊 Horse Ability & Trend Score Viewer")

# Path to bundled CSV (ensure it's committed to the repo and under 100MB)
DATA_FILE = "horse_ability_trend_scores_combined.csv"

# Load data from file (cached for performance)
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

# Load and display
try:
    df = load_data()
    st.success("✅ Data loaded successfully from file.")

    # Filter/search section
    with st.expander("🔍 Search / Filter"):
        horse_name = st.text_input("Filter by horse name:")
        if horse_name:
            filtered = df[df['Horse'].str.contains(horse_name, case=False)]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)

    st.divider()

    # Racecard Builder
    st.subheader("📋 Racecard Builder")
    selected_horses = st.multiselect("Select horses to compare:", df['Horse'].unique())

    if selected_horses:
        racecard_df = df[df['Horse'].isin(selected_horses)].copy()
        racecard_df = racecard_df[['Horse', 'RaceType', 'CurrentAbility']]
        total = racecard_df['CurrentAbility'].sum()
        racecard_df['WinChance (%)'] = (racecard_df['CurrentAbility'] / total * 100).round(2)
        racecard_df = racecard_df.sort_values("WinChance (%)", ascending=False).reset_index(drop=True)

        st.write("### 🏇 Racecard Comparison")
        st.dataframe(racecard_df, use_container_width=True)

except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
