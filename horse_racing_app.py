import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import logging
from sklearn.linear_model import LogisticRegression

# Configure logging for info and above, include timestamp
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set Streamlit page config for better layout (wide mode for desktop, responsive on mobile)
st.set_page_config(page_title="Horse Racing Analytics", layout="wide")

# Title of the app
st.title("🏇 Horse Racing Analytics App")

# Step 1: File upload section
st.header("1. Upload Racing Data")
st.write("Upload a horse racing CSV file (or use the default dataset) to begin.")

# File uploader (accept CSV or compressed CSV)
uploaded_file = st.file_uploader("Choose a racing data file (CSV or CSV.gz)", type=["csv", "gz", "gzip"])

@st.cache_data(show_spinner=False)
def load_data(file) -> tuple:
    """Load and preprocess racing data from a CSV file (possibly compressed).
    Returns a tuple of (DataFrame, race_summary DataFrame)."""
    try:
        # Read CSV (handle compression automatically if needed)
        if file is None:
            # No file uploaded: use default dataset
            data_path = "racing_data_upload.csv.gz"
            df = pd.read_csv(data_path, low_memory=False)
            logging.info(f"Loaded data file '{data_path}' successfully with shape {df.shape}")
        else:
            # Use uploaded file; reset pointer and read
            file.seek(0)
            df = pd.read_csv(file, low_memory=False)
            logging.info(f"Loaded uploaded data file successfully with shape {df.shape}")
    except Exception as e:
        logging.error(f"Error reading data file: {e}")
        # Re-raise to be caught outside
        raise

    # Basic validation for required columns
    required_cols = {"RaceDate", "RaceTime", "Yards", "Seconds", "FPos", "Trainer", "Jockey", "Id"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        logging.error(f"Data is missing required columns: {missing}")
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # Combine RaceDate and RaceTime into a single datetime, and sort data chronologically
    # Strip any extraneous time or date parts (RaceDate might include '00:00:00', RaceTime might include dummy date)
    date_str = df["RaceDate"].astype(str).str.split().str[0]  # take date portion before any space
    time_str = df["RaceTime"].astype(str).str.split().str[-1]  # take time portion (after last space, handles dummy date prefix)
    df["RaceDateTime"] = pd.to_datetime(date_str + " " + time_str, dayfirst=True, errors="coerce")
    # Log any parse issues
    num_unparsed = df["RaceDateTime"].isna().sum()
    if num_unparsed > 0:
        logging.warning(f"Warning: {num_unparsed} RaceDateTime entries could not be parsed.")
    # Sort by RaceDateTime (ascending chronological order)
    df.sort_values("RaceDateTime", inplace=True, kind="mergesort")
    logging.info("Parsed RaceDate and RaceTime into a combined datetime and sorted the data chronologically.")

    # Convert numeric columns from strings to numeric types (coerce errors to NaN)
    numeric_cols = ["Yards", "Seconds", "TotalBtn", "OR", "WeightLBS", "Ran"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate SpeedRating = Yards / Seconds (distance per second), handle missing or zero values
    df["SpeedRating"] = np.nan
    valid_time_mask = df["Yards"].notna() & df["Seconds"].notna() & (df["Yards"] > 0) & (df["Seconds"] > 0)
    df.loc[valid_time_mask, "SpeedRating"] = df.loc[valid_time_mask, "Yards"] / df.loc[valid_time_mask, "Seconds"]
    # Log how many entries could not have SpeedRating computed
    num_no_speed = len(df) - int(valid_time_mask.sum())
    if num_no_speed > 0:
        logging.warning(f"{num_no_speed} entries have missing or invalid times/distances; SpeedRating not computed for these.")

    # Compute PaceScore: adjust speed rating by adding distance behind to time (approx 0.2 sec per length)
    df["PaceScore"] = np.nan
    length_to_sec = 0.2
    pace_mask = valid_time_mask & df["TotalBtn"].notna()
    df.loc[pace_mask, "PaceScore"] = df.loc[pace_mask, "Yards"] / (df.loc[pace_mask, "Seconds"] + df.loc[pace_mask, "TotalBtn"] * length_to_sec)
    # For entries where PaceScore couldn't be computed (e.g., missing TotalBtn but have SpeedRating), use SpeedRating as fallback
    fallback_mask = df["PaceScore"].isna() & df["SpeedRating"].notna()
    df.loc[fallback_mask, "PaceScore"] = df.loc[fallback_mask, "SpeedRating"]

    # Compute SuperAbilityScore (e.g., use Official Rating OR as a base measure of ability)
    df["OR"].fillna(0, inplace=True)
    df["SuperAbilityScore"] = df["OR"]

    # Calculate Trainer win percentage and Jockey win percentage across the dataset
    # Determine winners for each entry
    df["FPos_num"] = pd.to_numeric(df["FPos"], errors="coerce")
    df["is_winner"] = (df["FPos_num"] == 1).astype(int)
    # Group by Trainer and Jockey to compute total wins and runs
    trainer_wins = df.groupby("Trainer")["is_winner"].sum()
    trainer_runs = df.groupby("Trainer")["is_winner"].count()
    trainer_win_pct = trainer_wins / trainer_runs
    jockey_wins = df.groupby("Jockey")["is_winner"].sum()
    jockey_runs = df.groupby("Jockey")["is_winner"].count()
    jockey_win_pct = jockey_wins / jockey_runs
    # Map these percentages back to each entry
    df["TrainerWinPct"] = df["Trainer"].map(trainer_win_pct).fillna(0.0)
    df["JockeyWinPct"] = df["Jockey"].map(jockey_win_pct).fillna(0.0)

    # Prepare race summary for selection: unique races with date, time, course, etc.
    # We use the RaceDateTime and Id (race ID) and Course/Race name for identification
    race_summary = df[["Id", "Course", "RaceDateTime", "Race", "Distance", "Class", "Going", "Ran"]].drop_duplicates(subset="Id")
    race_summary["RaceDate"] = race_summary["RaceDateTime"].dt.date  # date only
    # Sort races by datetime descending (latest first) for easier selection of recent races
    race_summary.sort_values("RaceDateTime", ascending=False, inplace=True)
    logging.info("Computed performance metrics (PaceScore, SuperAbilityScore, TrainerWinPct, JockeyWinPct) and prepared race summary.")

    return df, race_summary

# Attempt to load data (with caching) inside a spinner for user feedback
try:
    with st.spinner("Loading and processing data..."):
        df, race_summary = load_data(uploaded_file)
except Exception as e:
    # If data loading failed, show an error and stop
    st.error(f"Failed to load data: {e}")
    st.stop()

# Step 2: Race selection section (only show if data loaded successfully)
st.header("2. Select a Race")
# Get unique race dates for selection (latest first)
unique_dates = sorted(race_summary["RaceDate"].unique(), reverse=True)
# If no dates found (df empty), handle gracefully
if not unique_dates:
    st.error("No race data available in the file.")
    st.stop()

# Race date selection
selected_date = st.selectbox("Race Date", unique_dates, format_func=lambda x: x.strftime("%d %b %Y") if hasattr(x, "strftime") else str(x))
# Filter available races for the selected date
races_on_date = race_summary[race_summary["RaceDate"] == selected_date]
# Prepare race options for selection (e.g., "HH:MM - Course - RaceName")
race_options = []
race_id_map = {}
for _, race in races_on_date.sort_values("RaceDateTime").iterrows():
    race_time_str = race["RaceDateTime"].strftime("%H:%M")
    course = race["Course"]
    race_name = str(race["Race"]) if pd.notna(race["Race"]) else ""
    option_label = f"{race_time_str} - {course}"
    if race_name:
        option_label += f" - {race_name}"
    race_options.append(option_label)
    race_id_map[option_label] = race["Id"]
if not race_options:
    st.warning("No races found for the selected date.")
    st.stop()

selected_race_label = st.selectbox("Race (Time - Course - Name)", race_options)
selected_race_id = race_id_map.get(selected_race_label)

# Step 3: Train model (if not already trained for current data) and analyze selected race
# Define features for the model
features = ["SpeedRating", "PaceScore", "SuperAbilityScore", "TrainerWinPct", "JockeyWinPct"]

# Determine a data identifier to cache model per dataset (use file name and size or shape as proxy)
if uploaded_file is None:
    data_id = "default_dataset"
else:
    data_id = f"{uploaded_file.name}_{uploaded_file.size}"

# Train or retrieve the model (avoid retraining if same data already used in session)
if "model" not in st.session_state or st.session_state.get("data_id") != data_id:
    # Train a new model for this dataset
    with st.spinner("Training prediction model..."):
        try:
            X = df[features].fillna(0.0)
            y = df["is_winner"]
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
            model.fit(X, y)
            st.session_state["model"] = model
            st.session_state["data_id"] = data_id
            logging.info(f"Trained win-prediction model on {len(df)} entries.")
        except Exception as ex:
            logging.error(f"Model training failed: {ex}")
            st.error("Model training failed. Please check the data and try again.")
            st.stop()
else:
    model = st.session_state["model"]
    logging.info("Using cached model from previous run for the current dataset.")

# Once model is ready, proceed to race analysis
st.header("3. Race Analysis & Predictions")

# Retrieve data for the selected race
race_df = df[df["Id"] == selected_race_id].copy()
if race_df.empty:
    st.error("No data found for the selected race.")
    st.stop()

# Sort horses by their card number or finishing position for consistent order (if not already sorted)
# The data might already be sorted by finishing position within a race due to earlier sorting.
# We'll sort by FPos_num (with NaN last for non-finishers) just to be sure.
race_df["FPos_num"] = pd.to_numeric(race_df["FPos"], errors="coerce")
race_df.sort_values("FPos_num", inplace=True, na_position="last")

# Predict win probabilities for each horse in the race
X_race = race_df[features].fillna(0.0)
if model:
    # model.predict_proba returns [P(not win), P(win)] for each entry
    probs = model.predict_proba(X_race)[:, 1]
    race_df["PredWinProb"] = probs
else:
    race_df["PredWinProb"] = 0.0  # fallback if model not available

# Display race details (course, time, going, class, etc.) as a header
race_info = race_summary[race_summary["Id"] == selected_race_id].iloc[0]
race_title = f"**{race_info['Course']}**, {race_info['RaceDateTime'].strftime('%d %b %Y %H:%M')} – *{race_info['Race']}*"
race_details = f"Distance: {race_info['Distance']} | Going: {race_info['Going']} | Class: {race_info['Class']} | Runners: {int(race_info['Ran']) if pd.notna(race_info['Ran']) else race_df.shape[0]}"
st.markdown(f"**Selected Race:** {race_title}")
st.markdown(race_details)

# Display a racecard table with horse details and predicted win probability
show_columns = ["HorseName", "Trainer", "Jockey", "SpeedRating", "PaceScore", "SuperAbilityScore", "TrainerWinPct", "JockeyWinPct", "PredWinProb"]
# Format percentages and probabilities for display
display_df = race_df[show_columns].copy()
display_df["TrainerWinPct"] = display_df["TrainerWinPct"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
display_df["JockeyWinPct"] = display_df["JockeyWinPct"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
display_df["PredWinProb"] = display_df["PredWinProb"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
st.subheader("Racecard & Predicted Probabilities")
st.dataframe(display_df, use_container_width=True)

# Create a bar chart for predicted win probabilities
st.subheader("Predicted Win Probabilities by Horse")
# Sort by predicted probability for chart
race_df_chart = race_df.copy()
race_df_chart.sort_values("PredWinProb", ascending=False, inplace=True)
chart_data = race_df_chart[["HorseName", "PredWinProb"]]
prob_chart = alt.Chart(chart_data).mark_bar(color='#1f77b4').encode(
    x=alt.X("HorseName:N", sort=None, axis=alt.Axis(labelAngle=-45)),
    y=alt.Y("PredWinProb:Q", title="Win Probability", axis=alt.Axis(format='%'))
)
st.altair_chart(prob_chart, use_container_width=True)

# Create a bar chart for comparative PaceScore across horses
st.subheader("Comparative Pace Scores")
pace_chart_data = race_df_chart[["HorseName", "PaceScore"]]
pace_chart = alt.Chart(pace_chart_data).mark_bar(color='#ff7f0e').encode(
    x=alt.X("HorseName:N", sort=None, axis=alt.Axis(labelAngle=-45)),
    y=alt.Y("PaceScore:Q", title="Pace Score (higher = faster)")
)
st.altair_chart(pace_chart, use_container_width=True)

# Provide a concluding note or insight
st.write("*(The model above is trained on historical data to estimate each horse's win probability. Predictions are for demonstration purposes and should be interpreted with caution.)*")
