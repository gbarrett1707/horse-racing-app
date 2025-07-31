import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(page_title="Horse Racing Racecard & Predictor", page_icon="🐎", layout="wide")

st.title("🏇 Horse Racing Racecard Builder & Predictor")
st.write("Build your own custom racecard, get advanced runner stats, and see data-driven win probabilities for your selections.")

# --- Load data ---
@st.cache_data
def load_data():
    # Always load latest file
    df = pd.read_csv("racing_data_upload.csv.gz", low_memory=False)
    # Dates
    df['RaceDate'] = pd.to_datetime(df['RaceDate'], errors='coerce')
    if 'RaceTime' in df.columns:
        df['RaceTime'] = pd.to_datetime(df['RaceTime'], errors='coerce')
    # Handle column names
    for col in ['Seconds', 'Yards', 'HorseName', 'Trainer', 'Jockey']:
        if col not in df.columns:
            df[col] = np.nan
    # Calculate SpeedRating (Yards per Second), skip if missing
    df['SpeedRating'] = np.where(
        (df['Yards'].notnull()) & (df['Seconds'].notnull()) & (df['Seconds'] > 0),
        df['Yards'] / df['Seconds'],
        np.nan
    )
    # Drop rows missing critical info for ML/stats
    df = df.dropna(subset=['HorseName', 'RaceDate', 'SpeedRating'])
    # Add runner form for each horse (last 5 runs)
    df = df.sort_values(['HorseName', 'RaceDate'])
    df['Last5SpeedRatings'] = (
        df.groupby('HorseName')['SpeedRating']
        .rolling(window=5, min_periods=1).apply(lambda x: list(x), raw=False)
        .reset_index(level=0, drop=True)
    )
    # Add consistency (stddev over last 5 runs)
    df['Consistency'] = (
        df.groupby('HorseName')['SpeedRating']
        .rolling(window=5, min_periods=2).std().reset_index(level=0, drop=True)
    )
    # Trainer/jockey win%
    trainer_win = df.groupby('Trainer').apply(lambda x: (x['FPos']==1).mean()*100 if 'FPos' in x else np.nan)
    jockey_win = df.groupby('Jockey').apply(lambda x: (x['FPos']==1).mean()*100 if 'FPos' in x else np.nan)
    df['TrainerWinPct'] = df['Trainer'].map(trainer_win)
    df['JockeyWinPct'] = df['Jockey'].map(jockey_win)
    # Preferred track for horse/trainer/jockey (most frequent course with best avg speed)
    def preferred_track(group):
        if 'Course' not in group or group['Course'].isnull().all():
            return np.nan
        return group.groupby('Course')['SpeedRating'].mean().idxmax()
    df['HorsePrefTrack'] = df.groupby('HorseName').apply(preferred_track).reindex(df.index)
    df['TrainerPrefTrack'] = df.groupby('Trainer').apply(preferred_track).reindex(df.index)
    df['JockeyPrefTrack'] = df.groupby('Jockey').apply(preferred_track).reindex(df.index)
    return df

df = load_data()

if df is None or df.empty:
    st.error("No data loaded! Please check your source file.")
    st.stop()

# --- Build custom racecard ---
st.subheader("🎯 Build a Custom Racecard")
all_horses = df['HorseName'].unique()
selected_horses = st.multiselect("Select horses to add to your custom racecard:", all_horses)

if not selected_horses:
    st.info("Select at least two horses to build a racecard and see comparisons.")
    st.stop()

custom_race_df = df[df['HorseName'].isin(selected_horses)].copy()

# Always show latest row per horse for current ability, plus last 5 runs
custom_race_df = custom_race_df.sort_values('RaceDate')
latest_rows = (
    custom_race_df.groupby("HorseName", as_index=False)
    .last()
)

# Build stats per runner
def build_runner_stats(row):
    # Last 5 runs
    horse_hist = df[df['HorseName'] == row['HorseName']].sort_values('RaceDate').tail(5)
    last5_speeds = horse_hist['SpeedRating'].tolist()
    fastest_last_run = np.nanmax(last5_speeds) if last5_speeds else np.nan
    most_consistent = np.nanstd(last5_speeds) if len(last5_speeds) > 1 else np.nan
    # Other features
    return pd.Series({
        "Last5SpeedRatings": " → ".join([f"{x:.2f}" for x in last5_speeds]),
        "FastestLastRun": fastest_last_run,
        "Consistency5": most_consistent,
        "AvgAbility": np.nanmean(last5_speeds),
        "TrainerInForm": row['TrainerWinPct'],
        "JockeyInForm": row['JockeyWinPct'],
        "HorsePrefTrack": row['HorsePrefTrack'],
        "TrainerPrefTrack": row['TrainerPrefTrack'],
        "JockeyPrefTrack": row['JockeyPrefTrack'],
    })

runner_stats = latest_rows.apply(build_runner_stats, axis=1)
display_df = pd.concat([latest_rows.reset_index(drop=True), runner_stats], axis=1)

# --- Predict win probabilities (Logistic Regression heuristic) ---
# Build features
features = ['SpeedRating', 'Consistency', 'TrainerWinPct', 'JockeyWinPct', 'FastestLastRun', 'AvgAbility']
for f in features:
    if f not in display_df.columns:
        display_df[f] = np.nan
X = display_df[features].fillna(0).values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Fake binary outcomes (simulate: horse with fastest last run as winner)
y = (display_df['FastestLastRun'] == display_df['FastestLastRun'].max()).astype(int)
try:
    model = LogisticRegression()
    model.fit(X_scaled, y)
    win_probs = model.predict_proba(X_scaled)[:,1]
except Exception:
    win_probs = np.full(len(display_df), 1/len(display_df))

display_df['PredictedWinProb(%)'] = np.round(win_probs * 100, 2)

# --- Visuals ---
st.subheader("📊 Runner Stats and Predictions")
st.dataframe(display_df[
    ['HorseName', 'SpeedRating', 'Last5SpeedRatings', 'FastestLastRun', 'Consistency5',
     'AvgAbility', 'TrainerInForm', 'JockeyInForm', 'HorsePrefTrack', 'TrainerPrefTrack', 'JockeyPrefTrack', 'PredictedWinProb(%)']
].rename(columns={
    'HorseName': 'Horse',
    'SpeedRating': 'Latest Speed',
    'FastestLastRun': 'Fastest Run (Last 5)',
    'Consistency5': 'Consistency (Std Dev)',
    'AvgAbility': 'Avg Ability (Last 5)',
    'TrainerInForm': 'Trainer Win%',
    'JockeyInForm': 'Jockey Win%',
    'HorsePrefTrack': 'Horse Pref Track',
    'TrainerPrefTrack': 'Trainer Pref Track',
    'JockeyPrefTrack': 'Jockey Pref Track',
    'PredictedWinProb(%)': 'Win Probability %'
}), use_container_width=True)

# --- Plots ---
st.markdown("#### Win Probabilities")
fig = px.bar(display_df, x="HorseName", y="PredictedWinProb(%)", text="PredictedWinProb(%)", title="Predicted Win Probability (%)")
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(yaxis=dict(range=[0, 100]))
st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Speed Rating Trends")
for horse in selected_horses:
    history = df[df['HorseName'] == horse].sort_values('RaceDate')
    fig = px.line(history, x='RaceDate', y='SpeedRating', title=f"{horse} Speed Ratings")
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

st.success("Ready. If you want more stats or new visuals, just say the word!")

