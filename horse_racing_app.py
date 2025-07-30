import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Horse Racing Predictor & Racecard Builder", page_icon="🏎", layout="wide")

st.title("🏇 Horse Racing Predictor & Racecard Builder")
st.write("Explore horse racing data, view derived performance metrics, and predict win probabilities for each horse. Select a race from the sidebar to get started.")

@st.cache_data
def load_data():
    file_name = "racing_data upload.csv.gz"
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found. Please ensure it exists in the working directory.")
        return None
    df = pd.read_csv(file_name)
    if 'RaceDate' not in df.columns:
        st.error("Required data column 'RaceDate' not found. Please check your input file.")
        return None
    if 'RaceTime' not in df.columns:
        st.warning("Column 'RaceTime' not found. Some features may be limited.")
    df['RaceDate'] = pd.to_datetime(df['RaceDate'], errors='coerce')
    if 'RaceTime' in df.columns:
        df['RaceTime'] = pd.to_datetime(df['RaceTime'], errors='coerce')
        df['RaceTimeOnly'] = df['RaceTime'].dt.time
    return df

@st.cache_data
def compute_trainer_jockey_stats(df):
    if df is None:
        return None, None
    trainer_stats_df, jockey_stats_df = None, None
    if 'Trainer' in df.columns:
        trainer_group = df.groupby('Trainer')
        trainer_stats_df = pd.DataFrame({
            'Runs': trainer_group.size(),
            'Wins': trainer_group['FPos'].apply(lambda x: (pd.to_numeric(x, errors='coerce') == 1).sum()),
            'Places': trainer_group['FPos'].apply(lambda x: (pd.to_numeric(x, errors='coerce') <= 3).sum())
        })
        trainer_stats_df['WinPct'] = trainer_stats_df['Wins'] / trainer_stats_df['Runs'] * 100
        trainer_stats_df['PlacePct'] = trainer_stats_df['Places'] / trainer_stats_df['Runs'] * 100
    if 'Jockey' in df.columns:
        jockey_group = df.groupby('Jockey')
        jockey_stats_df = pd.DataFrame({
            'Rides': jockey_group.size(),
            'Wins': jockey_group['FPos'].apply(lambda x: (pd.to_numeric(x, errors='coerce') == 1).sum()),
            'Places': jockey_group['FPos'].apply(lambda x: (pd.to_numeric(x, errors='coerce') <= 3).sum())
        })
        jockey_stats_df['WinPct'] = jockey_stats_df['Wins'] / jockey_stats_df['Rides'] * 100
        jockey_stats_df['PlacePct'] = jockey_stats_df['Places'] / jockey_stats_df['Rides'] * 100
    return trainer_stats_df, jockey_stats_df

def compute_pace_and_ability(df, horses, race_key=None):
    pace_scores, ability_scores = {}, {}
    if df is None:
        return pace_scores, ability_scores
    for horse in horses:
        horse_hist = df[df['HorseName'] == horse].copy()
        if horse_hist.empty:
            pace_scores[horse] = 0.0
            ability_scores[horse] = 0.0
            continue
        horse_hist.sort_values(['RaceDate', 'RaceTime'], inplace=True)
        if race_key:
            if 'RaceTimeOnly' in horse_hist.columns:
                horse_hist = horse_hist[~(
                    (horse_hist['Course'] == race_key[0]) &
                    (horse_hist['RaceDate'] == race_key[1]) &
                    (horse_hist['RaceTimeOnly'] == race_key[2])
                )]
        if horse_hist.empty:
            pace_scores[horse] = 0.0
            ability_scores[horse] = 0.0
        else:
            last_speed = horse_hist['SpeedRating'].iloc[-1] if 'SpeedRating' in horse_hist.columns else 0.0
            recent_speeds = horse_hist['SpeedRating'].tail(3) if 'SpeedRating' in horse_hist.columns else pd.Series([0.0])
            avg_recent = recent_speeds.mean()
            pace_scores[horse] = round(last_speed, 1) if pd.notna(last_speed) else 0.0
            ability_scores[horse] = round((last_speed + avg_recent) / 2, 1) if pd.notna(avg_recent) else 0.0
    return pace_scores, ability_scores

def predict_win_probabilities(race_df):
    if race_df.empty:
        return []
    try:
        model = joblib.load("win_model.pkl")
        features = ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']
        X = race_df[features].fillna(0.0).astype(float)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)
    except:
        df = race_df.copy()
        for col in ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']:
            if col not in df:
                df[col] = 0.0
            else:
                max_val = df[col].max()
                df[col] = df[col] / max_val if max_val > 0 else 0.0
        df['Score'] = df[['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']].sum(axis=1)
        total = df['Score'].sum()
        return (df['Score'] / total).values if total > 0 else np.repeat(1.0 / len(df), len(df))

with st.spinner("Loading data..."):
    df = load_data()

if df is not None:
    trainer_stats, jockey_stats = compute_trainer_jockey_stats(df)
    st.sidebar.header("Select Race")
    if 'RaceDate' in df.columns:
        available_dates = sorted(df['RaceDate'].dt.date.unique())
        selected_date = st.sidebar.date_input("Race Date", available_dates[-1] if available_dates else None)
        date_filtered = df[df['RaceDate'].dt.date == selected_date]
        courses = sorted(date_filtered['Course'].unique())
        selected_course = st.sidebar.selectbox("Course", courses)
        course_filtered = date_filtered[date_filtered['Course'] == selected_course]
        times = course_filtered[['RaceTime', 'Race']].drop_duplicates()
        times['label'] = times.apply(lambda x: f"{x['RaceTime'].strftime('%H:%M')} - {x['Race']}", axis=1)
        selected_label = st.sidebar.selectbox("Race", times['label'].tolist())
        selected_row = times[times['label'] == selected_label].iloc[0]
        selected_time = selected_row['RaceTime']
        race_df = df[(df['Course'] == selected_course) &
                     (df['RaceDate'].dt.date == selected_date) &
                     (df['RaceTime'] == selected_time)]
        if not race_df.empty:
            horses = race_df['HorseName'].unique()
            race_key = (selected_course, pd.to_datetime(selected_date), selected_time.time())
            pace_scores, ability_scores = compute_pace_and_ability(df, horses, race_key=race_key)
            race_df['PaceScore'] = race_df['HorseName'].map(pace_scores)
            race_df['SuperAbilityScore'] = race_df['HorseName'].map(ability_scores)
            if trainer_stats is not None:
                race_df['TrainerWinPct'] = race_df['Trainer'].map(trainer_stats['WinPct'])
            if jockey_stats is not None:
                race_df['JockeyWinPct'] = race_df['Jockey'].map(jockey_stats['WinPct'])
            race_df['PredictedWinProb'] = predict_win_probabilities(race_df) * 100
            st.subheader(f"Race: {selected_course} — {selected_label}")
            st.dataframe(race_df[['HorseName', 'Trainer', 'Jockey', 'PaceScore', 'SuperAbilityScore',
                                 'TrainerWinPct', 'JockeyWinPct', 'PredictedWinProb']])
            fig = px.bar(race_df, x='HorseName', y='PredictedWinProb', text='PredictedWinProb',
                         title="Predicted Win Probability by Horse")
            fig.update_layout(xaxis_title="Horse", yaxis_title="Win Probability (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No race data found for the selected course and time.")
    else:
        st.error("'RaceDate' column is required but not found in dataset.")
else:
    st.warning("No data loaded. Please upload a valid racing data file.")
