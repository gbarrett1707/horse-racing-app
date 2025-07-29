import streamlit as st
import pandas as pd

st.set_page_config(layout="centered", page_title="Horse Viewer", page_icon="📊")
st.title("📊 Horse Ability & Trend Score Viewer")

uploaded_file = st.file_uploader("Upload combined ability scores CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and loaded!")

        # Display full table
        st.dataframe(df)

        # 🔍 Filtering section
        with st.expander("🔍 Search / Filter"):
            horse_name = st.text_input("Filter by horse name:")
            if horse_name:
                filtered = df[df['Horse'].str.contains(horse_name, case=False)]
                st.write(filtered)

        # 📋 Racecard Builder
        st.subheader("📋 Racecard Builder")
        selected_horses = st.multiselect("Select horses to compare:", df['Horse'].unique())

        if selected_horses:
            racecard_df = df[df['Horse'].isin(selected_horses)].copy()
            racecard_df = racecard_df[['Horse', 'RaceType', 'CurrentAbility']]

            total_ability = racecard_df['CurrentAbility'].sum()
            racecard_df['WinChance (%)'] = (racecard_df['CurrentAbility'] / total_ability * 100).round(2)

            st.write("### 🏇 Racecard Comparison")
            st.dataframe(racecard_df.reset_index(drop=True))

    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
else:
    st.info("📂 Upload your combined score file to begin.")
