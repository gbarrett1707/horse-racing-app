import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import streamlit as st

# Set page configuration for Streamlit
st.set_page_config(page_title="Horse Racing Predictor", page_icon="🐎", layout="wide")

# Title and description in the app
st.title("🐎 Horse Racing Predictor & Racecard Builder")
st.write("Explore horse racing data, view derived performance metrics, and predict win probabilities for each horse. Select a race from the sidebar to get started.")

# Load racing data (combine parts if necessary) with caching
@st.cache_data
def load_data():
    """Load and combine racing data into a single DataFrame, with basic cleaning and feature computation."""
    # Possible file locations (compressed or uncompressed)
    file_candidates = [
        "racing_data_upload.csv.gz",    # combined data compressed
        "racing_data upload.csv.gz",    # alternate naming
        "racing_data_upload.csv"        # uncompressed CSV
    ]
    df = None
    for file in file_candidates:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file, low_memory=False)
                break
            except Exception as e:
                st.error(f"Error reading file '{file}': {e}")
                st.stop()
    if df is None:
        st.error("Data files not found. Please ensure the racing data CSV files are available.")
        st.stop()
    # Verify required columns
    required_cols = ['RaceDate', 'HorseName', 'FPos', 'Ran', 'Seconds', 'TotalBtn', 'Yards', 'WeightLBS', 'Age', 'Class', 'CardNo']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Required data columns {missing} not found in input data. Please check the data format.")
        st.stop()
    # Compute horse finish times and speed ratings
    # If TotalBtn (distance behind winner in lengths) is NaN (e.g., for non-finishers), HorseTime will be NaN
    df['HorseTime'] = np.where(pd.isna(df['TotalBtn']), np.nan, df['Seconds'] + 0.2 * df['TotalBtn'])
    df['SpeedRating'] = df['Yards'] / df['HorseTime']
    df['SpeedRating'] = df['SpeedRating'].fillna(0.0)  # Non-finishers get 0 speed rating
    # Clean and convert RaceDate and RaceTime for sorting
    df['RaceDate_dt'] = pd.to_datetime(df['RaceDate'], dayfirst=True, errors='coerce')
    # Some RaceDate entries might fail to parse if format is unexpected
    if df['RaceDate_dt'].isna().all():
        st.error("RaceDate column could not be parsed. Please check date format in data.")
        st.stop()
    # RaceTime might include an Excel base date (e.g., '12/30/99') – strip it to get actual time
    df['RaceTime'] = df['RaceTime'].astype(str).str[-8:]
    # Convert RaceTime to seconds from midnight for sorting
    def time_to_seconds(t):
        try:
            h, m, s = t.split(':')
            return int(h) * 3600 + int(m) * 60 + int(s)
        except:
            return 0
    df['RaceTime_sec'] = df['RaceTime'].apply(time_to_seconds)
    # Sort data chronologically by race date and time (and by Id to group same races)
    df.sort_values(['RaceDate_dt', 'RaceTime_sec', 'Id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Cache the loaded data
racing_df = load_data()

# Train or load the machine learning model (cache result to avoid repeat training)
@st.cache_resource
def get_trained_model(data):
    """Train a logistic regression model to predict race winners. Returns a trained model (pipeline)."""
    # If a pre-trained model file exists, load it to save time
    model_path = "horse_win_model.joblib"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            # If loading fails, proceed to train a new model
            st.warning(f"Could not load saved model: {e}. Retraining the model.")
    # Prepare training feature matrix X and target vector y
    X_features = []
    y_target = []
    # Initialize horse form ratings and speed history containers
    form_rating = {}   # current trend/form rating for each horse
    speed_history = {} # list of recent speed ratings for each horse
    alpha = 0.7        # smoothing factor for trend score (higher -> more weight on past form)
    # Iterate through races in chronological order to build features
    for race_id, race_data in data.groupby('Id'):
        # For each horse in this race, record features before the race
        for _, row in race_data.iterrows():
            horse = row['HorseName']
            # Initialize horse tracking if first encounter
            if horse not in form_rating:
                form_rating[horse] = 50.0   # baseline form score
                speed_history[horse] = []
            # Feature 1: current trend/form score before this race
            trend_score = form_rating[horse]
            # Feature 2: average of last 3 speed ratings before this race
            last3_speeds = speed_history[horse][-3:]
            if last3_speeds:
                # Treat NaN speeds as 0 in the average (e.g., if horse did not finish in one of those races)
                clean_speeds = [0.0 if pd.isna(sp) else sp for sp in last3_speeds]
                avg_speed = float(np.mean(clean_speeds))
            else:
                avg_speed = 0.0
            # Feature 3: horse's age
            age = float(row['Age']) if not pd.isna(row['Age']) else 0.0
            # Feature 4: horse's carried weight in lbs
            weight = float(row['WeightLBS']) if not pd.isna(row['WeightLBS']) else 0.0
            # Feature 5: race class (numeric, lower is higher class)
            race_class = float(row['Class']) if not pd.isna(row['Class']) else 0.0
            # Feature 6: race distance in yards
            distance = float(row['Yards']) if not pd.isna(row['Yards']) else 0.0
            X_features.append([trend_score, avg_speed, age, weight, race_class, distance])
            # Target: 1 if this horse won the race, 0 otherwise
            won = 1 if (str(row['FPos']).isdigit() and int(row['FPos']) == 1) else 0
            y_target.append(won)
        # After computing features for all horses in the race, update each horse's form rating based on race results
        for _, row in race_data.iterrows():
            horse = row['HorseName']
            ran = int(row['Ran']) if not pd.isna(row['Ran']) else 0
            # Determine finishing position; if not a finisher (e.g., 'PU'), treat as last (position = ran)
            if str(row['FPos']).isdigit():
                pos = int(row['FPos'])
            else:
                pos = ran if ran > 0 else 0
            # Compute performance metric (0 to 1, higher is better) for this race
            perf = 1.0
            if ran > 1:
                perf = (ran - pos) / (ran - 1)  # e.g., winner gets 1.0, last gets 0.0
            # Update trend/form rating using exponential smoothing
            prev_rating = form_rating[horse]
            new_rating = alpha * prev_rating + (1 - alpha) * (perf * 100)
            form_rating[horse] = new_rating
            # Update speed history with this race's speed rating
            sp = float(row['SpeedRating']) if not pd.isna(row['SpeedRating']) else 0.0
            speed_history[horse].append(sp)
    X = np.array(X_features)
    y = np.array(y_target)
    # Train a logistic regression model with standard scaling
    model_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight='balanced'))
    model_pipeline.fit(X, y)
    # Save the model for future use
    try:
        joblib.dump(model_pipeline, model_path)
    except Exception as e:
        st.warning(f"Could not save model to disk: {e}")
    return model_pipeline

# Get trained model (this will load from cache or train if not already cached)
model = get_trained_model(racing_df)

# Prepare sidebar for race selection
st.sidebar.header("Race Selection")
# Create a sorted list of races for selection (format: "Date – Course – RaceName")
race_options = []
race_summary = racing_df.groupby('Id').first()  # one representative row per race
race_summary = race_summary.sort_values(['RaceDate_dt', 'RaceTime_sec'])
for rid, row in race_summary.iterrows():
    date_str = row['RaceDate_dt'].strftime("%d %b %Y")
    course = row['Course']
    race_name = row['Race']
    race_options.append(f"{date_str} – {course} – {race_name} (Id {rid})")
# Sidebar selectbox for races
selected_label = st.sidebar.selectbox("Select a race:", race_options)
# Extract race Id from the selected label (the Id is at the end in parentheses)
try:
    selected_id = int(selected_label.split("Id")[-1].strip().strip(')'))
except:
    # Fallback: parse by finding the last part after the last '–'
    parts = selected_label.split("–")
    selected_id = int(parts[-1].split()[-1]) if parts else None

if not selected_id or selected_id not in racing_df['Id'].values:
    st.error("Invalid race selection.")
    st.stop()

# Filter data for the selected race
race_df = racing_df[racing_df['Id'] == selected_id].copy()
if race_df.empty:
    st.error("No data found for the selected race.")
    st.stop()

# Get race metadata from the first entry
race_info = race_df.iloc[0]
race_date = race_info['RaceDate_dt'].strftime("%d %b %Y")
race_course = race_info['Course']
race_name = race_info['Race']
race_class = race_info['Class'] if not pd.isna(race_info['Class']) else "N/A"
race_age = race_info['AgeLimit'] if isinstance(race_info['AgeLimit'], str) else ""
race_going = race_info['Going'] if isinstance(race_info['Going'], str) else ""
race_distance = race_info['Distance'] if isinstance(race_info['Distance'], str) else f"{int(race_info['Yards'])}y"

# Display race header information
st.subheader(f"**{race_course} – {race_name}**")
st.write(f"**Date:** {race_date}  |  **Class:** {race_class}  |  **Age Limit:** {race_age}  |  **Going:** {race_going}  |  **Distance:** {race_distance}")

# Compute prediction features for each horse in the selected race (using only past data up to this race)
pred_features = []
for _, horse_row in race_df.iterrows():
    horse = horse_row['HorseName']
    # Filter this horse's past races (strictly before the selected race's date/time)
    hist = racing_df[(racing_df['HorseName'] == horse) &
                     ((racing_df['RaceDate_dt'] < race_info['RaceDate_dt']) |
                      ((racing_df['RaceDate_dt'] == race_info['RaceDate_dt']) & 
                       (racing_df['RaceTime_sec'] < race_info['RaceTime_sec'])))]
    # Calculate horse's trend score prior to this race
    curr_rating = 50.0
    past_speeds = []
    if not hist.empty:
        hist = hist.sort_values(['RaceDate_dt', 'RaceTime_sec'])
        for _, past_row in hist.iterrows():
            ran = int(past_row['Ran']) if not pd.isna(past_row['Ran']) else 0
            if str(past_row['FPos']).isdigit():
                pos = int(past_row['FPos'])
            else:
                pos = ran if ran > 0 else 0
            perf = 1.0
            if ran > 1:
                perf = (ran - pos) / (ran - 1)
            curr_rating = alpha * curr_rating + (1 - alpha) * (perf * 100)
            past_speeds.append(float(past_row['SpeedRating']) if not pd.isna(past_row['SpeedRating']) else 0.0)
    # Average of last 3 speeds before this race
    if past_speeds:
        last3 = past_speeds[-3:]
        clean_last3 = [0.0 if pd.isna(x) else x for x in last3]
        avg_speed = float(np.mean(clean_last3))
    else:
        avg_speed = 0.0
    age = float(horse_row['Age']) if not pd.isna(horse_row['Age']) else 0.0
    weight = float(horse_row['WeightLBS']) if not pd.isna(horse_row['WeightLBS']) else 0.0
    race_class_val = float(horse_row['Class']) if not pd.isna(horse_row['Class']) else 0.0
    distance_val = float(horse_row['Yards']) if not pd.isna(horse_row['Yards']) else 0.0
    pred_features.append([curr_rating, avg_speed, age, weight, race_class_val, distance_val])

pred_features = np.array(pred_features)
# Get win probability predictions for each horse in the race
probs = model.predict_proba(pred_features)[:, 1]
# Normalize probabilities so they sum to 1 (for a clearer comparison as a percentage)
if probs.sum() > 0:
    probs = probs / probs.sum()
# Add predictions to the race DataFrame
race_df['PredProb'] = probs
race_df['WinProb(%)'] = (race_df['PredProb'] * 100).round(1)

# Highlight the top pick with a star
if len(race_df) > 0:
    top_idx = race_df['PredProb'].idxmax()
    race_df.at[top_idx, 'HorseName'] = "★ " + str(race_df.at[top_idx, 'HorseName'])

# Rename finishing position column for clarity
race_df.rename(columns={'FPos': 'Result'}, inplace=True)

# Sort horses by their race card number (CardNo) for display
race_df.sort_values('CardNo', inplace=True)

# Select and arrange columns for the output table
display_cols = ['CardNo', 'HorseName', 'Age', 'WeightLBS', 'Jockey', 'Trainer', 'Sp', 'WinProb(%)', 'Result']
race_display = race_df[display_cols].reset_index(drop=True)

# Show the race card table with predictions
st.write("**Racecard and Predictions:**")
st.dataframe(race_display.style.format({"WinProb(%)": "{:.1f}"}), height=400)

# Plot an interactive bar chart of predicted win probabilities
chart_df = race_df[['HorseName', 'PredProb']].copy()
chart_df.sort_values('PredProb', ascending=False, inplace=True)
fig = px.bar(chart_df, x='HorseName', y='PredProb', 
             title="Predicted Win Probability by Horse", 
             labels={'HorseName': 'Horse', 'PredProb': 'Win Probability'})
fig.update_layout(yaxis_tickformat='.0%', yaxis_title='Win Probability', xaxis_title=None)
fig.update_traces(texttemplate='%{y:.1%}', textposition='outside', marker_color='skyblue')
st.plotly_chart(fig, use_container_width=True)
