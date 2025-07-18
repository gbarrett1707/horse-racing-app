import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime

import streamlit as st

# Upload section
st.title("🏇 Horse Ability & Trend Analyzer")
uploaded_file = st.file_uploader("Upload your racing data CSV", type=["csv"])

# Define dtypes to avoid warnings
dtype_overrides = {
    'Id': str, 'Class': str, 'Prize': str, 'Ran': str, 'Yards': str, 'Limit': str,
    'Seconds': str, 'TotalBtn': str, 'CardNo': str, 'Draw': str, 'Sp': str,
    'Age': str, 'Stone': str, 'Lbs': str, 'WeightLBS': str, 'Allow': str, 'OR': str
}

# Lbs per length conversion (Flat)
def lbs_per_length(distance_f):
    if distance_f < 6: return 3.5
    elif distance_f < 9: return 2.5
    elif distance_f < 12: return 2.0
    elif distance_f < 16: return 1.5
    else: return 1.0

# Lbs per length conversion (Jumps)
def lbs_per_length_jumps(distance_f):
    if distance_f < 16: return 1.0
    elif distance_f < 20: return 0.9
    else: return 0.75

# Simplified WFA scale
WFA_ALLOWANCE = {
    'Flat': {'3': 8},
    'Jumps': {'4': 10}
}

# Convert yards to furlongs
def get_distance_f(row):
    try:
        return float(row['Yards']) / 220
    except:
        return np.nan

# Adjust weight by adding WFA + allowance
def adjust_weight(row):
    try:
        weight = float(row['WeightLBS'])
        age = str(int(float(row['Age'])))
        race_type = row['RaceCode']
        wfa = WFA_ALLOWANCE.get(race_type, {}).get(age, 0)
        allow = float(row['Allow']) if pd.notna(row['Allow']) else 0
        return weight + wfa + allow
    except:
        return np.nan

# Performance rating vs winner
def performance_rating(row, winner_rating, winner_weight, lbs_per_len):
    try:
        dist_btn = float(row.get('TotalBtn', 0))
    except:
        dist_btn = 0
    wt_diff = row['AdjWeight'] - winner_weight
    return winner_rating - (dist_btn * lbs_per_len) + wt_diff

# Trend label logic
def trend_slope(values):
    if len(values) < 2:
        return 0, "Stable"
    slope = (values[-1] - values[0]) / (len(values) - 1)
    if slope >= 1.0:
        return slope, "Improving"
    elif slope <= -1.0:
        return slope, "Declining"
    else:
        return slope, "Stable"

# Main logic
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, dtype=dtype_overrides, low_memory=False)
        st.success("✅ Data uploaded and loaded successfully!")

        if 'Type' not in df.columns:
            st.error("❌ The uploaded file is missing the required column: 'Type'.")
            st.stop()

        # ✅ Normalize and correctly map flat vs jumps
        df['Type'] = df['Type'].str.strip().str.lower()
        df['RaceCode'] = df['Type'].map(lambda x: 'Flat' if x in ['f', 'a', 'af', 'fa'] else 'Jumps')

        df['DistanceF'] = df.apply(get_distance_f, axis=1)
        df['AdjWeight'] = df.apply(adjust_weight, axis=1)

        df['FPos'] = pd.to_numeric(df['FPos'], errors='coerce')
        df = df.dropna(subset=['FPos'])
        df = df.sort_values(['Id', 'FPos'])

        ratings = []
        for race_id, race_group in df.groupby('Id'):
            race_type = race_group['RaceCode'].iloc[0]
            dist_f = race_group['DistanceF'].mean()
            lbs_len = lbs_per_length(dist_f) if race_type == 'Flat' else lbs_per_length_jumps(dist_f)

            race_group = race_group.sort_values('FPos')
            winner = race_group.iloc[0]

            try:
                winner_rating = float(winner['OR']) if pd.notna(winner['OR']) else 80
            except:
                winner_rating = 80

            winner_weight = winner['AdjWeight']

            for _, row in race_group.iterrows():
                perf = performance_rating(row, winner_rating, winner_weight, lbs_len)
                ratings.append({
                    'Horse': row['HorseName'],
                    'RaceDate': row['RaceDate'],
                    'Rating': perf,
                    'RaceType': race_type
                })

        perf_df = pd.DataFrame(ratings)
        perf_df['RaceDate'] = pd.to_datetime(perf_df['RaceDate'], format="%d/%m/%y %H:%M:%S", errors='coerce')
        perf_df = perf_df.dropna(subset=['RaceDate'])

        result = []
        for (horse, code), group in perf_df.groupby(['Horse', 'RaceType']):
            group = group.sort_values('RaceDate')
            ratings = list(group['Rating'])

            if len(ratings) == 0:
                continue

            weights = [i+1 for i in range(len(ratings[-3:]))]
            current = np.average(ratings[-3:], weights=weights)
            slope, label = trend_slope(ratings[-5:])

            result.append({
                'Horse': horse,
                'RaceType': code,
                'CurrentAbility': round(current, 2),
                'TrendScore': round(slope, 2),
                'TrendCategory': label
            })

        final_scores = pd.DataFrame(result)
        final_scores.sort_values(['RaceType', 'CurrentAbility'], ascending=[True, False], inplace=True)

        # ✅ Save safely to temp folder
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, "horse_ability_trend_scores.csv")
        final_scores.to_csv(output_path, index=False)

        # 🌟 Let user download file
        with open(output_path, "rb") as f:
            st.download_button(
                label="📅 Download Final Ability Scores CSV",
                data=f,
                file_name="horse_ability_trend_scores.csv",
                mime="text/csv"
            )

        st.success("✅ Processing complete! Your results are ready to download.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("📂 Please upload a CSV file to get started.")
