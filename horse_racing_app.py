import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import logging
from sklearn.ensemble import RandomForestClassifier

# Configure logging for debugging and tracing
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page layout to wide for better display (optional)
st.set_page_config(layout="wide", page_title="Horse Racing Analysis")

# Title and description
st.title("Horse Racing Performance Viewer")
st.write(
    "This Streamlit app loads historical horse racing data, computes performance metrics, and "
    "allows you to explore race results with predicted win probabilities for each horse."
)

@st.cache_data
def load_data():
    """Load and preprocess the racing data from a CSV file."""
    data_file = "racing_data_upload.csv.gz"
    try:
        # Read the compressed CSV data
        df = pd.read_csv(data_file, low_memory=False)
        logger.info(f"Loaded data file '{data_file}' successfully with shape {df.shape}")
    except FileNotFoundError as e:
        logger.error(f"Data file '{data_file}' not found. Exception: {e}", exc_info=True)
        st.error(f"Error: Data file '{data_file}' not found. Please ensure the file is present.")
        st.stop()
    except Exception as e:
        logger.error(f"Failed to read data file. Exception: {e}", exc_info=True)
        st.error(f"Error: Could not load the data due to an unexpected error.")
        st.stop()

    # Ensure required columns are present
    required_cols = {"RaceDate", "Course", "Race", "Yards", "Seconds", "HorseName", "FPos"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        st.error(f"Error: The data is missing required columns: {', '.join(missing_cols)}. "
                 "Cannot continue without these columns.")
        st.stop()

    # Parse RaceDate and RaceTime into a proper datetime for sorting
    # Clean RaceDate (remove time if present) and RaceTime (remove any date component)
    date_str = df["RaceDate"].astype(str).str.strip()
    date_str = date_str.str.replace(r"\s*00:00:00", "", regex=True)  # remove " 00:00:00" if present

    time_str = df["RaceTime"].astype(str).str.strip()
    # If there's a date portion in RaceTime (e.g., "12/30/99 12:45:00"), remove it
    time_str = time_str.str.split(" ").str[-1]
    # Ensure time has seconds component for uniform format
    time_str = time_str.apply(lambda t: t + ":00" if t and ":" in t and t.count(":") == 1 else t)

    # Combine date and time strings and parse into datetime
    combined_dt = date_str + " " + time_str
    # Use dayfirst=True to correctly interpret formats like "31-May-25" and "01/01/25"
    df["RaceDateTime"] = pd.to_datetime(combined_dt, dayfirst=True, errors="coerce")

    # If any datetimes failed to parse, log a warning
    if df["RaceDateTime"].isnull().any():
        num_null = df["RaceDateTime"].isnull().sum()
        logger.warning(f"Warning: {num_null} RaceDateTime entries could not be parsed.")
        # Attempt a secondary parse for any unparsed entries (try different format assumptions if needed)
        # For robustness, we can try common alternate formats on the failed ones
        failed_idxs = df[df["RaceDateTime"].isnull()].index
        for idx in failed_idxs:
            raw_date = str(df.at[idx, "RaceDate"]).strip()
            raw_time = str(df.at[idx, "RaceTime"]).strip()
            try:
                # Try swapping dayfirst False if initial failed
                parsed = pd.to_datetime(raw_date + " " + raw_time, dayfirst=False)
                df.at[idx, "RaceDateTime"] = parsed
            except Exception:
                pass  # Leave as NaT if still fails
        if df["RaceDateTime"].isnull().any():
            # If still null, alert the user but continue
            st.warning("Some race dates/times could not be parsed. These entries will be handled as unknown dates.")

    # Sort the DataFrame by RaceDateTime to prepare for grouping and trend calculations
    df.sort_values("RaceDateTime", inplace=True)
    logger.info("Parsed RaceDate and RaceTime into a combined datetime and sorted the data chronologically.")

    # Calculate SpeedRating = Yards / Seconds, handle missing or zero times gracefully
    df["SpeedRating"] = np.nan
    # Only compute if Yards and Seconds are valid positive numbers
    valid_time_mask = df["Seconds"].notna() & (df["Seconds"] > 0) & df["Yards"].notna() & (df["Yards"] > 0)
    df.loc[valid_time_mask, "SpeedRating"] = df.loc[valid_time_mask, "Yards"] / df.loc[valid_time_mask, "Seconds"]
    # Log if any speed ratings could not be computed due to missing data
    num_no_speed = len(df) - valid_time_mask.sum()
    if num_no_speed > 0:
        logger.info(f"SpeedRating could not be calculated for {num_no_speed} entries (missing or zero values in Yards/Seconds).")

    # Compute ability and trend scores for each horse
    ability_list = []
    trend_list = []
    horse_list = []

    # Group by HorseName and calculate ability and trend
    for horse, grp in df.groupby("HorseName"):
        # Ensure the group is sorted by RaceDateTime (df is already sorted globally)
        # Calculate ability as the mean SpeedRating of all completed races (skip NaN speeds)
        # Calculate trend as difference between last race SpeedRating and the horse's previous average SpeedRating
        horse_list.append(horse)
        # Ability score: average of SpeedRating (skip NaN values)
        if grp["SpeedRating"].notna().any():
            ability_val = grp["SpeedRating"].mean(skipna=True)
        else:
            ability_val = np.nan  # no finished races
        # Determine last race's SpeedRating (the last row in this group, which is the most recent race)
        last_speed = grp.iloc[-1]["SpeedRating"]
        if len(grp) <= 1:
            # Only one race (or none finished) -> no prior race to compare, trend is 0
            trend_val = 0.0
        else:
            # Compute the mean SpeedRating of all previous races (exclude the last race's entry)
            prev_mean = grp.iloc[:-1]["SpeedRating"].mean(skipna=True)
            if pd.isna(last_speed):
                # Last race did not have a recorded SpeedRating (e.g., did not finish)
                if not np.isnan(prev_mean):
                    # If there were previous finished races, trend is negative (0 minus previous average)
                    trend_val = 0.0 - prev_mean
                else:
                    # No previous finished races either (all races were DNF)
                    trend_val = 0.0
            else:
                if np.isnan(prev_mean):
                    # Last race was the first finished race (no prior finished races)
                    trend_val = 0.0
                else:
                    trend_val = last_speed - prev_mean
        ability_list.append(ability_val)
        trend_list.append(trend_val)
    # Create a DataFrame for horse stats and merge back to main DataFrame
    horse_stats_df = pd.DataFrame({
        "HorseName": horse_list,
        "AbilityScore": ability_list,
        "TrendScore": trend_list
    })
    # Merge ability and trend into the main DataFrame
    df = df.merge(horse_stats_df, on="HorseName", how="left")

    # Replace any remaining NaN in AbilityScore or TrendScore with 0 (for horses with no completed races)
    df["AbilityScore"].fillna(0.0, inplace=True)
    df["TrendScore"].fillna(0.0, inplace=True)

    logger.info("Computed ability and trend scores for each horse and merged into main DataFrame.")

    # Prepare a summary of unique races for the selectors (one entry per race)
    # We'll use the first occurrence (which is likely the winner or first listed) for each race ID
    race_summary = df.drop_duplicates(subset="Id", keep="first").copy()
    # Extract just the date (without time) for selection use
    race_summary["RaceDate_only"] = race_summary["RaceDateTime"].dt.date
    # Sort race_summary by date and time
    race_summary.sort_values("RaceDateTime", inplace=True)

    return df, race_summary

# Load and process data (cached)
df, race_summary = load_data()

# Prepare options for selectors
# Unique race dates (as Python date objects) sorted, latest first
unique_dates = sorted(race_summary["RaceDate_only"].unique(), reverse=True)
if not unique_dates:
    st.error("No race dates found in the data.")
    st.stop()

# Race Date selector
selected_date = st.selectbox("Select Race Date", unique_dates, index=0,
                             format_func=lambda x: x.strftime("%d %b %Y") if isinstance(x, (pd.Timestamp, np.datetime64)) or hasattr(x, "strftime") else str(x))
# Filter tracks available on the selected date
tracks_on_date = race_summary[race_summary["RaceDate_only"] == selected_date]["Course"].unique()
tracks_on_date.sort()
if len(tracks_on_date) == 0:
    st.warning("No races found for the selected date. Please choose a different date.")
    st.stop()
selected_track = st.selectbox("Select Track", tracks_on_date, index=0)

# Filter races for the selected date and track
races_filtered = race_summary[(race_summary["RaceDate_only"] == selected_date) & (race_summary["Course"] == selected_track)]
if races_filtered.empty:
    st.warning("No races found for the selected date and track.")
    st.stop()

# Create a mapping from Race Id to a display label (time and race name)
race_options = races_filtered.copy()
# Format RaceTime for display (use RaceDateTime or RaceTimeClean if exists)
if "RaceTimeClean" in race_options.columns:
    # If we had stored a cleaned time string in preprocessing (not explicitly in code above, but could reuse RaceDateTime)
    # We can derive time string from RaceDateTime for display
    race_options["DisplayTime"] = race_options["RaceDateTime"].dt.strftime("%H:%M")
else:
    race_options["DisplayTime"] = race_options["RaceDateTime"].dt.strftime("%H:%M")
race_options["DisplayLabel"] = race_options["DisplayTime"] + " - " + race_options["Race"] + " (" + race_options["Ran"].astype(str) + " runners)"
race_id_list = race_options["Id"].tolist()
label_dict = dict(zip(race_options["Id"], race_options["DisplayLabel"]))

selected_race_id = st.selectbox("Select Race", race_id_list, format_func=lambda x: label_dict.get(x, str(x)))

# Now retrieve the data for the selected race
race_df = df[df["Id"] == selected_race_id].copy()
if race_df.empty:
    st.error("Error: No data found for the selected race.")
    st.stop()

# Sort the race entries by finishing position if available (1st, 2nd, etc.)
# FPos might be numeric or strings like 'PU', so we handle sorting carefully:
# We'll create a sort key that puts winners (FPos=1) first, then 2, 3, ... and non-finishers at the end.
def sort_key(pos):
    """
    Generate a sort key for finishing position. Numeric positions come first in order, 
    non-numeric (e.g., 'PU', 'F') come last in original order.
    """
    try:
        # Convert to int if possible (this will work for numeric strings and ints)
        return (0, int(pos))
    except:
        # Non-numeric positions get a large default sort value
        return (1, float('inf'))

race_df["FPos_str"] = race_df["FPos"].astype(str).str.strip()  # ensure string type for FPos
race_df.sort_values(by="FPos_str", key=lambda col: col.map(sort_key), inplace=True)
race_df.drop(columns=["FPos_str"], inplace=True)

# Identify the winner(s) of the race for display
# There could be ties, but we'll assume one winner for simplicity (FPos == 1)
race_df["WinnerFlag"] = race_df["FPos"].astype(str).str.strip().apply(lambda x: True if x == "1" else False)

# Prepare features for prediction
features = ["AbilityScore", "TrendScore", "OR", "Age", "WeightLBS"]
# Ensure the necessary columns for model features exist; if not, we adjust
for col in features:
    if col not in df.columns:
        # Fallback: if any feature column missing, drop it from feature list
        features.remove(col)
        logger.warning(f"Feature '{col}' is missing in data. It will be omitted from the model.")

# Prepare and train the machine learning model (RandomForest) to predict win probability
@st.cache_resource
def train_model(df, feature_cols):
    """Train a RandomForestClassifier on the historical data to predict race winners."""
    # Define target: winner or not
    y = (df["FPos"].astype(str).str.strip() == "1").astype(int)
    # Filter out any entries where required feature data is missing
    X = df[feature_cols].copy()
    # Fill any remaining NaNs in features with 0 (safe fallback for missing numerical data)
    X = X.fillna(0)
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X, y)
    logger.info(f"Trained RandomForest model with {len(feature_cols)} features on {len(X)} samples.")
    return model

model = None
if features:
    try:
        model = train_model(df, features)
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        st.warning("Warning: The machine learning model could not be trained due to an error. "
                   "Predictions will not be available.")
        model = None
else:
    st.warning("Insufficient feature columns for model training. Predictions are disabled.")

# If model is available, compute win probabilities for each horse in the selected race
if model is not None:
    X_race = race_df[features].copy().fillna(0)
    # Get probability of class "1" (winning) for each horse
    win_proba = model.predict_proba(X_race)[:, 1]
    race_df["WinProbability"] = win_proba
    # Convert probability to percentage string for display
    race_df["WinProbabilityPct"] = (race_df["WinProbability"] * 100).map(lambda x: f"{x:.1f}%")
    logger.info("Computed win probabilities for horses in the selected race.")
else:
    race_df["WinProbabilityPct"] = "N/A"

# Display race information and results
race_title = race_options[race_options["Id"] == selected_race_id]["Race"].iloc[0]
race_time_str = race_options[race_options["Id"] == selected_race_id]["DisplayTime"].iloc[0]
st.subheader(f"Race Details: {race_title}")
st.write(f"**Date:** {selected_date.strftime('%d %b %Y')}  |  **Track:** {selected_track}  |  **Time:** {race_time_str}  |  **Runners:** {len(race_df)}")

# Show an interactive table of the race entrants with their details and predicted win probabilities
# Select relevant columns to display
display_cols = ["HorseName", "Age", "WeightLBS", "OR", "AbilityScore", "TrendScore", "WinProbabilityPct", "FPos", "Jockey", "Trainer"]
# Filter out columns not in data (in case some are missing in certain dataset variations)
display_cols = [col for col in display_cols if col in race_df.columns]
# Rename columns for clarity in the UI
col_rename = {
    "HorseName": "Horse",
    "WeightLBS": "Weight (lbs)",
    "OR": "Official Rating",
    "AbilityScore": "Ability Score",
    "TrendScore": "Trend Score",
    "WinProbabilityPct": "Win Probability",
    "FPos": "Finish Position"
}
race_display_df = race_df[display_cols].rename(columns=col_rename)

# Highlight the winner row in the table for clarity (if any)
def highlight_winner(row):
    return ['background-color: gold' if str(row.get("Finish Position", "")).strip() == "1" or row.get("Finish Position", "") == 1 else '' for _ in row]

# Use st.dataframe with styling
st.dataframe(race_display_df.style.apply(highlight_winner, axis=1), use_container_width=True)

# Additionally, display a bar chart of win probabilities if available
if model is not None:
    chart_data = race_df.copy()
    chart_data.sort_values("WinProbability", ascending=True, inplace=True)  # sort by probability for better plotting
    # Use horse names as y-axis and win probability as x-axis for a horizontal bar chart
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("WinProbability", title="Predicted Win Probability", axis=alt.Axis(format='%')),
        y=alt.Y("HorseName", title="Horse", sort=None)  # sort=None to use data order (already sorted by prob)
    )
    text = alt.Chart(chart_data).mark_text(
        align='left',
        baseline='middle',
        dx=3  # adjust text position
    ).encode(
        x=alt.X("WinProbability", aggregate=None),
        y=alt.Y("HorseName", sort=None),
        text=alt.Text("WinProbability", format=".1%")
    )
    st.altair_chart(bar_chart + text, use_container_width=True)
