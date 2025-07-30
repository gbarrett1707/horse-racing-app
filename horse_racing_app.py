import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="Horse Racing Predictor & Racecard Builder", page_icon="🏎️", layout="wide")

# Title
st.title("🏎️ Horse Racing Predictor & Racecard Builder")
st.write("Explore horse racing data, view derived performance metrics, and predict win probabilities for each horse. Select a race from the sidebar to get started.")

@st.cache_data
def load_data():
    possible_files = [
        "racing_data_upload.csv.gz",
        "racing_data upload.csv.gz"
    ]
    for file_name in possible_files:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            if 'RaceDate' in df.columns:
                try:
                    df['RaceDate'] = pd.to_datetime(df['RaceDate'], errors='coerce')
                except:
                    pass
            if 'RaceTime' in df.columns:
                try:
                    df['RaceTime'] = pd.to_datetime(df['RaceTime'], errors='coerce')
                except:
                    pass
            if 'RaceTime' in df.columns and df['RaceTime'].dtype == 'datetime64[ns]':
                df['RaceTimeOnly'] = df['RaceTime'].dt.time
            return df
    st.error("File 'racing_data_upload.csv.gz' or 'racing_data upload.csv.gz' not found. Please ensure it exists in the working directory.")
    return None

@st.cache_data
def compute_trainer_jockey_stats(df):
    trainer_stats_df, jockey_stats_df = None, None
    if df is None:
        return None, None
    data = df.copy()
    if 'Trainer' in data.columns:
        trainer_group = data.groupby('Trainer')
        total_runs = trainer_group.size()
        wins = trainer_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') == 1).sum())
        places = trainer_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') <= 3).sum())
        trainer_stats_df = pd.DataFrame({'Runs': total_runs, 'Wins': wins, 'Places': places})
        trainer_stats_df['WinPct'] = (trainer_stats_df['Wins'] / trainer_stats_df['Runs'] * 100).round(1)
        trainer_stats_df['PlacePct'] = (trainer_stats_df['Places'] / trainer_stats_df['Runs'] * 100).round(1)
    if 'Jockey' in data.columns:
        jockey_group = data.groupby('Jockey')
        total_rides = jockey_group.size()
        wins_j = jockey_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') == 1).sum())
        places_j = jockey_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') <= 3).sum())
        jockey_stats_df = pd.DataFrame({'Rides': total_rides, 'Wins': wins_j, 'Places': places_j})
        jockey_stats_df['WinPct'] = (jockey_stats_df['Wins'] / jockey_stats_df['Rides'] * 100).round(1)
        jockey_stats_df['PlacePct'] = (jockey_stats_df['Places'] / jockey_stats_df['Rides'] * 100).round(1)
    return trainer_stats_df, jockey_stats_df

def predict_win_probabilities(race_df):
    if race_df.empty:
        return []
    model = None
    try:
        model = joblib.load("win_model.pkl")
    except:
        model = None
    if model:
        feature_cols = ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']
        X = race_df[feature_cols].fillna(0.0).astype(float)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            win_probs = probs[:, 1]
        else:
            win_probs = model.predict(X)
        return win_probs
    else:
        df = race_df.copy()
        for col in ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']:
            if col not in df.columns:
                df[col] = 0.0
        for col in ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']:
            max_val = df[col].max()
            if pd.notna(max_val) and max_val > 0:
                df[col] = df[col] / max_val
            else:
                df[col] = 0.0
        df['Score'] = (df['SpeedRating'] + df['PaceScore'] + df['SuperAbilityScore'] + df['TrainerWinPct'] + df['JockeyWinPct']) / 5.0
        total_score = df['Score'].sum()
        if total_score == 0:
            probs = np.array([1.0 / len(df)] * len(df))
        else:
            probs = df['Score'] / total_score
        return probs.values

with st.spinner("Loading data..."):
    df = load_data()

if df is not None:
    trainer_stats, jockey_stats = compute_trainer_jockey_stats(df)

    st.sidebar.header("Select Race")
    selected_date = None
    if 'RaceDate' in df.columns:
        available_dates = sorted(df['RaceDate'].dropna().dt.date.unique())
        if available_dates:
            default_date = available_dates[-1]
            selected_date = st.sidebar.date_input("Race Date", value=default_date,
                                                  min_value=min(available_dates), max_value=max(available_dates))
    if selected_date:
        date_mask = df['RaceDate'].dt.date == selected_date
        day_data = df[date_mask]
        courses = sorted(day_data['Course'].dropna().unique())
        if courses:
            selected_course = st.sidebar.selectbox("Course", courses)
            course_mask = (day_data['Course'] == selected_course)
            course_data = day_data[course_mask]
            races = []
            for time_val, race_name in course_data[['RaceTime', 'Race']].dropna().drop_duplicates().values:
                try:
                    t_str = pd.to_datetime(str(time_val)).strftime("%H:%M")
                except:
                    t_str = str(time_val)[-8:] if ":" in str(time_val) else str(time_val)
                label = f"{t_str} - {race_name}" if t_str else race_name
                races.append((time_val, label))
            races = sorted(races, key=lambda x: x[0])
            race_labels = [lbl for _, lbl in races]
            selected_label = st.sidebar.selectbox("Race", race_labels) if race_labels else None
            race_df = pd.DataFrame()
            if selected_label:
                for time_val, lbl in races:
                    if lbl == selected_label:
                        selected_time = time_val
                        break
                race_mask = (df['Course'] == selected_course) & (df['RaceDate'].dt.date == selected_date)
                if 'RaceTimeOnly' in df.columns and pd.notna(selected_time):
                    race_mask &= (df['RaceTimeOnly'] == (selected_time if not isinstance(selected_time, pd.Timestamp) else selected_time.time()))
                else:
                    race_mask &= (df['RaceTime'] == selected_time)
                race_df = df[race_mask].copy()
        else:
            race_df = pd.DataFrame()
    else:
        race_df = pd.DataFrame()

    if not race_df.empty:
        horses = race_df['HorseName'].unique()
        pace_scores, ability_scores = {}, {}
        for horse in horses:
            horse_hist = df[df['HorseName'] == horse].copy()
            if horse_hist.empty:
                pace_scores[horse] = 0.0
                ability_scores[horse] = 0.0
                continue
            horse_hist.sort_values(['RaceDate', 'RaceTime'], inplace=True)
            recent_speeds = horse_hist['SpeedRating'].tail(3)
            last_speed = recent_speeds.iloc[-1] if not recent_speeds.empty else 0.0
            avg_recent = recent_speeds.mean()
            super_score = (last_speed + avg_recent) / 2
            pace_scores[horse] = round(last_speed, 1) if pd.notna(last_speed) else 0.0
            ability_scores[horse] = round(super_score, 1) if pd.notna(super_score) else 0.0

        race_df['PaceScore'] = race_df['HorseName'].map(pace_scores)
        race_df['SuperAbilityScore'] = race_df['HorseName'].map(ability_scores)
        if trainer_stats is not None:
            race_df['TrainerWinPct'] = race_df['Trainer'].map(trainer_stats['WinPct'])
        if jockey_stats is not None:
            race_df['JockeyWinPct'] = race_df['Jockey'].map(jockey_stats['WinPct'])

        win_probs = predict_win_probabilities(race_df)
        race_df['PredictedWinProb'] = (win_probs * 100).round(1)

        display_cols = ['HorseName', 'Trainer', 'Jockey', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct', 'PredictedWinProb']
        display_df = race_df[display_cols].copy()
        display_df.rename(columns={
            'HorseName': 'Horse',
            'PaceScore': 'Pace',
            'SuperAbilityScore': 'Ability',
            'TrainerWinPct': 'Trainer Win%',
            'JockeyWinPct': 'Jockey Win%',
            'PredictedWinProb': 'Win %'
        }, inplace=True)

        st.subheader(f"Race: {selected_course} - {selected_label}")
        st.dataframe(display_df, height=400)

        st.markdown("**Win Probabilities**")
        prob_chart_data = display_df[['Horse', 'Win %']]
        fig_bar = px.bar(prob_chart_data, x='Horse', y='Win %', text='Win %')
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_yaxes(range=[0, 100])
        fig_bar.update_layout(xaxis_title=None, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning("Please select a valid race to view racecard predictions.")
