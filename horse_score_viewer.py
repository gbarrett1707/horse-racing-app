import streamlit as st
import pandas as pd

st.title("📊 Horse Ability & Trend Score Viewer")

uploaded_file = st.file_uploader("Upload combined ability scores CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and loaded!")
        st.dataframe(df)

        with st.expander("🔍 Search / Filter"):
            horse_name = st.text_input("Filter by horse name:")
            if horse_name:
                filtered = df[df['Horse'].str.contains(horse_name, case=False)]
                st.write(filtered)

    except Exception as e:
        st.error(f"❌ Failed to load CSV: {e}")
else:
    st.info("📂 Upload your combined score file to begin.")
