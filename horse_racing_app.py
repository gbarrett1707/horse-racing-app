import streamlit as st
import pandas as pd
import numpy as np
import joblib  # for loading a pre-trained model if available
import plotly.express as px
import os

# Set page configuration for mobile-friendly layout
st.set_page_config(page_title="Horse Racing Predictor & Racecard Builder", page_icon="🏎", layout="wide")

# App title and description
st.title("🏍 Horse Racing Predictor & Racecard Builder")
st.write("Explore horse racing data, view derived performance metrics, and predict win probabilities for each horse. Select a race from the sidebar to get started.")

@st.cache_data
def load_data():
    """Load processed racing data from combined trend scores CSV file."""
    file_name = "horse_ability_trend_scores_combined.csv"
    if not os.path.exists(file_name):
        st.error(f"File '{file_name}' not found. Please ensure it exists in the working directory.")
        return None
    df = pd.read_csv(file_name)
    if 'RaceDate' in df.columns:
        try:
            df['RaceDate'] = pd.to_datetime(df['RaceDate'])
        except:
            pass
    if 'RaceTime' in df.columns:
        try:
            df['RaceTime'] = pd.to_datetime(df['RaceTime'])
        except:
            pass
    if 'RaceTime' in df.columns and df['RaceTime'].dtype == 'datetime64[ns]':
        df['RaceTimeOnly'] = df['RaceTime'].dt.time
    return df

@st.cache_data
def compute_trainer_jockey_stats(df):
    trainer_stats_df = None
    jockey_stats_df = None
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
        if horse_hist.empty:
            pace_scores[horse] = 0.0
            ability_scores[horse] = 0.0
        else:
            last_speed = horse_hist['SpeedRating'].iloc[-1]
            recent_speeds = horse_hist['SpeedRating'].tail(3)
            avg_recent = recent_speeds.mean()
            super_score = (last_speed + avg_recent) / 2
            pace_scores[horse] = round(last_speed, 1) if pd.notna(last_speed) else 0.0
            ability_scores[horse] = round(super_score, 1) if pd.notna(super_score) else 0.0
    return pace_scores, ability_scores

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

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# If data is loaded, continue with race selection and prediction
if df is not None:
    # Compute stats
    trainer_stats, jockey_stats = compute_trainer_jockey_stats(df)

    # User selects race date and course
    st.sidebar.header("Select Race")
    available_dates = sorted(df['RaceDate'].dt.date.unique())
    selected_date = st.sidebar.date_input("Race Date", value=available_dates[-1], min_value=available_dates[0], max_value=available_dates[-1])
    selected_day_df = df[df['RaceDate'].dt.date == selected_date]

    available_courses = sorted(selected_day_df['Course'].unique())
    selected_course = st.sidebar.selectbox("Course", available_courses)
    course_day_df = selected_day_df[selected_day_df['Course'] == selected_course]

    available_times = course_day_df['RaceTimeOnly'].dropna().unique() if 'RaceTimeOnly' in course_day_df else course_day_df['RaceTime'].dropna().unique()
    selected_time = st.sidebar.selectbox("Race Time", sorted(available_times))

    race_key = (selected_course, pd.to_datetime(selected_date), selected_time)
    race_df = course_day_df[(course_day_df['RaceTimeOnly'] == selected_time) if 'RaceTimeOnly' in course_day_df else (course_day_df['RaceTime'] == selected_time)].copy()

    if not race_df.empty:
        horses = race_df['HorseName'].unique()
        pace_scores, ability_scores = compute_pace_and_ability(df, horses, race_key=race_key)

        race_df['PaceScore'] = race_df['HorseName'].map(pace_scores)
        race_df['SuperAbilityScore'] = race_df['HorseName'].map(ability_scores)
        race_df['TrainerWinPct'] = race_df['Trainer'].map(trainer_stats['WinPct']) if trainer_stats is not None else 0.0
        race_df['JockeyWinPct'] = race_df['Jockey'].map(jockey_stats['WinPct']) if jockey_stats is not None else 0.0

        win_probs = predict_win_probabilities(race_df)
        race_df['PredictedWinProb'] = (win_probs * 100).round(1)

        st.subheader(f"Race: {selected_course} — {selected_time}")
        st.dataframe(race_df[['HorseName', 'Trainer', 'Jockey', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct', 'PredictedWinProb']])

        st.markdown("**Win Probabilities**")
        fig_bar = px.bar(race_df, x='HorseName', y='PredictedWinProb', text='PredictedWinProb')
        fig_bar.update_layout(xaxis_title="Horse", yaxis_title="Win Probability (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

        if 'Sp' in race_df.columns:
            race_df['ImpliedProb'] = 100 / (race_df['Sp'].astype(float) + 1)
            scatter_data = race_df[['HorseName', 'PredictedWinProb', 'ImpliedProb']]
            fig_scatter = px.scatter(scatter_data, x='PredictedWinProb', y='ImpliedProb', text='HorseName')
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(xaxis_title="Predicted Win %", yaxis_title="Implied Win %")
            st.plotly_chart(fig_scatter, use_container_width=True)

        selected_horse = st.selectbox("Select Horse for Form Graph", race_df['HorseName'].unique())
        form_df = df[df['HorseName'] == selected_horse].sort_values('RaceDate')
        fig_form = px.line(form_df, x='RaceDate', y='SpeedRating', title=f"{selected_horse} - Speed Rating Trend")
        st.plotly_chart(fig_form, use_container_width=True)
