import streamlit as st
import pandas as pd
import numpy as np
import joblib  # for loading a pre-trained model if available
import plotly.express as px
import os

# Set page configuration for mobile-friendly layout
st.set_page_config(page_title="Horse Racing Predictor & Racecard Builder", page_icon="🏎", layout="wide")

# App title and description
st.title("🏎️ Horse Racing Predictor & Racecard Builder")
st.write("Explore horse racing data, view derived performance metrics, and predict win probabilities for each horse. Select a race from the sidebar to get started.")

@st.cache_data
def load_data():
    file_name = "racing_data_upload.csv.gz"
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found. Please ensure it exists in the working directory.")
        return None
    df = pd.read_csv(file_name)

    # Check and convert RaceDate and RaceTime
    if 'RaceDate' in df.columns:
        df['RaceDate'] = pd.to_datetime(df['RaceDate'], errors='coerce')
    if 'RaceTime' in df.columns:
        df['RaceTime'] = pd.to_datetime(df['RaceTime'], errors='coerce')
        df['RaceTimeOnly'] = df['RaceTime'].dt.time

    # Compute SpeedRating if possible
    if 'TimeTaken' in df.columns and 'DistanceRun' in df.columns:
        df['SpeedRating'] = df['DistanceRun'] / df['TimeTaken']
    else:
        df['SpeedRating'] = 0.0  # default if columns are missing

    return df

@st.cache_data
def compute_trainer_jockey_stats(df):
    trainer_stats_df = None
    jockey_stats_df = None
    if df is None:
        return None, None

    if 'Trainer' in df.columns:
        trainer_group = df.groupby('Trainer')
        total_runs = trainer_group.size()
        wins = trainer_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') == 1).sum())
        places = trainer_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') <= 3).sum())
        trainer_stats_df = pd.DataFrame({'Runs': total_runs, 'Wins': wins, 'Places': places})
        trainer_stats_df['WinPct'] = (trainer_stats_df['Wins'] / trainer_stats_df['Runs'] * 100).round(1)
        trainer_stats_df['PlacePct'] = (trainer_stats_df['Places'] / trainer_stats_df['Runs'] * 100).round(1)

    if 'Jockey' in df.columns:
        jockey_group = df.groupby('Jockey')
        total_rides = jockey_group.size()
        wins_j = jockey_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') == 1).sum())
        places_j = jockey_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') <= 3).sum())
        jockey_stats_df = pd.DataFrame({'Rides': total_rides, 'Wins': wins_j, 'Places': places_j})
        jockey_stats_df['WinPct'] = (jockey_stats_df['Wins'] / jockey_stats_df['Rides'] * 100).round(1)
        jockey_stats_df['PlacePct'] = (jockey_stats_df['Places'] / jockey_stats_df['Rides'] * 100).round(1)

    return trainer_stats_df, jockey_stats_df

def compute_pace_and_ability(df, horses, race_key=None):
    pace_scores = {}
    ability_scores = {}
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
                horse_hist = horse_hist[~((horse_hist['Course'] == race_key[0]) &
                                          (horse_hist['RaceDate'] == race_key[1]) &
                                          (horse_hist['RaceTimeOnly'] == race_key[2]))]
            else:
                horse_hist = horse_hist[~((horse_hist['Course'] == race_key[0]) &
                                          (horse_hist['RaceDate'] == race_key[1]) &
                                          (horse_hist['RaceTime'] == race_key[2]))]

        last_speed = horse_hist['SpeedRating'].iloc[-1] if not horse_hist.empty else 0.0
        recent_speeds = horse_hist['SpeedRating'].tail(3)
        avg_recent = recent_speeds.mean() if not recent_speeds.empty else 0.0
        super_score = (last_speed + avg_recent) / 2

        pace_scores[horse] = round(last_speed, 1) if pd.notna(last_speed) else 0.0
        ability_scores[horse] = round(super_score, 1) if pd.notna(super_score) else 0.0

    return pace_scores, ability_scores

def predict_win_probabilities(race_df):
    if race_df.empty:
        return []

    try:
        model = joblib.load("win_model.pkl")
    except:
        model = None

    if model:
        feature_cols = ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']
        X = race_df[feature_cols].fillna(0.0).astype(float)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            return probs[:, 1]
        else:
            return model.predict(X)
    else:
        df = race_df.copy()
        for col in ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']:
            df[col] = df[col] if col in df.columns else 0.0
        df['Score'] = df[['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']].mean(axis=1)
        total_score = df['Score'].sum()
        return (df['Score'] / total_score).values if total_score > 0 else np.full(len(df), 1.0 / len(df))

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df is not None:
    st.write("Preview of loaded data:")
    st.write(df.head())
    
    # This is where race selector, visuals, and racecard logic would continue
    # Add more components below to complete the interface
else:
    st.stop()
