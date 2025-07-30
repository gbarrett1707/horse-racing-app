import streamlit as st
import pandas as pd
import numpy as np
from fractions import Fraction

# Title and description
st.title("Horse Racing Analysis and Prediction")
st.markdown(
    "This Streamlit app loads historical horse racing data, computes performance metrics, "
    "and allows you to explore race results with predicted win probabilities for each horse."
)

@st.cache_data
def load_data():
    # Attempt to load the compressed CSV data
    file_path = "racing_data_upload.csv.gz"
    try:
        df = pd.read_csv(file_path, compression="gzip", low_memory=False)
    except UnicodeDecodeError:
        # Fallback to an alternative encoding if UTF-8 fails
        df = pd.read_csv(file_path, compression="gzip", low_memory=False, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Failed to load data file: {e}")
        st.stop()

    # Log and handle data shape
    n_entries = df.shape[0]
    n_columns = df.shape[1]
    # Combine RaceDate and RaceTime into a single datetime, handling known format issues
    # RaceDate appears as "DD/MM/YY 00:00:00" and RaceTime as "12/30/99 HH:MM:SS" (base date)
    # Extract date and time components from these strings
    date_str = df["RaceDate"].astype(str).str.split(" ").str[0]       # e.g. "01/01/25"
    time_str = df["RaceTime"].astype(str).str.split(" ").str[-1]      # e.g. "12:45:00"
    # Parse to datetime (dayfirst format since dates are DD/MM/YY)
    df["RaceDateTime"] = pd.to_datetime(date_str + " " + time_str, format="%d/%m/%y %H:%M:%S", errors="coerce")
    # Count parsing failures
    failed_parses = df["RaceDateTime"].isna().sum()
    if failed_parses > 0:
        st.warning(f"Some race dates/times could not be parsed. These entries will be excluded from analysis.")
        # Remove entries with invalid date/time
        df = df[df["RaceDateTime"].notna()].copy()
    # Sort data chronologically by RaceDateTime
    df.sort_values("RaceDateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert numeric columns from strings to proper dtypes
    df["Yards"] = pd.to_numeric(df["Yards"], errors="coerce")
    df["Seconds"] = pd.to_numeric(df["Seconds"], errors="coerce")
    df["OR"] = pd.to_numeric(df["OR"], errors="coerce")
    # Drop or flag entries with missing/invalid time or distance
    valid_mask = (df["Yards"].notna()) & (df["Seconds"].notna()) & (df["Yards"] > 0) & (df["Seconds"] > 0)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        st.warning(f"{invalid_count} entries have missing or invalid times/distances; they will be dropped.")
    df = df[valid_mask].copy()
    # Calculate SpeedRating = Yards / Seconds
    df["SpeedRating"] = df["Yards"] / df["Seconds"]

    # Fill missing Official Ratings (OR) with 0 (assume unrated horses get 0)
    df["OR"].fillna(0, inplace=True)
    # Calculate PaceScore for each horse (based on distance behind winner in lengths)
    # Convert finishing position and beaten lengths to numeric
    df["FPos_num"] = pd.to_numeric(df["FPos"], errors="coerce")
    df["TotalBtn_num"] = pd.to_numeric(df["TotalBtn"], errors="coerce")
    # Compute max distance behind per race (group by Race Id)
    max_behind = df.groupby("Id")["TotalBtn_num"].transform("max")
    # For horses with no recorded total beaten (NaN, e.g. did not finish), use max_behind as their beaten distance
    df["TotalBtn_num"].fillna(max_behind, inplace=True)
    # PaceScore = (max_beaten_distance - horse_beaten_distance) / max_beaten_distance (0 to 1 scale)
    df["PaceScore"] = np.where(max_behind > 0, (max_behind - df["TotalBtn_num"]) / max_behind,
                               np.where(df["FPos_num"] == 1, 1.0, 0.0))
    # Calculate SuperAbilityScore (combining SpeedRating and OR as a composite ability metric)
    df["SuperAbilityScore"] = df["SpeedRating"] * 10 + df["OR"]
    # Compute Trainer and Jockey win percentages
    df["Won"] = (df["FPos_num"] == 1).astype(int)
    trainer_group = df.groupby("Trainer")["Won"]
    trainer_win_pct = (trainer_group.sum() / trainer_group.count()).fillna(0.0)
    jockey_group = df.groupby("Jockey")["Won"]
    jockey_win_pct = (jockey_group.sum() / jockey_group.count()).fillna(0.0)
    # Map win percentages back to each entry (as percentages)
    df["TrainerWinPct"] = df["Trainer"].map(trainer_win_pct) * 100
    df["JockeyWinPct"] = df["Jockey"].map(jockey_win_pct) * 100

    # Prepare a summary of races for selection (one entry per race)
    # Take the first entry of each race (data is sorted chronologically)
    race_summary = df.drop_duplicates(subset=["Id"], keep="first").copy()
    # Extract race date (as Python date object) for selection filtering
    race_summary["RaceDate"] = race_summary["RaceDateTime"].dt.date
    # Also store race time (just time component) and other details for display
    race_summary["RaceTimeOnly"] = race_summary["RaceDateTime"].dt.time

    # Train a simple machine learning model (logistic regression) to predict win probability
    from sklearn.linear_model import LogisticRegression
    features = ["SpeedRating", "PaceScore", "SuperAbilityScore", "TrainerWinPct", "JockeyWinPct"]
    X = df[features].values
    y = df["Won"].values
    # Train logistic regression on the entire dataset (this is for demonstration purposes)
    model = LogisticRegression(solver="sag", max_iter=200, random_state=42)
    # If dataset is very large, the solver 'sag' might not converge fully, but it's fast for approximate results
    try:
        model.fit(X, y)
    except Exception as e:
        # In case of convergence or other issues, try a simpler solver as fallback
        model = LogisticRegression(solver="lbfgs", max_iter=100, random_state=42)
        model.fit(X, y)

    return df, race_summary, model

# Step 1: Load and prepare data
st.header("1. Load and Prepare Data")
with st.spinner("Loading data..."):
    df, race_summary, model = load_data()
st.success(f"Loaded data with {df.shape[0]} entries and {race_summary.shape[0]} races.")

# Step 2: Race selection
st.header("2. Select a Race")
if race_summary.shape[0] == 0:
    st.error("No race data available for selection.")
    st.stop()

# Get unique race dates (most recent first)
unique_dates = sorted(race_summary["RaceDate"].unique(), reverse=True)
selected_date = st.selectbox("Choose Race Date", unique_dates)
# Filter races for the selected date
races_on_date = race_summary[race_summary["RaceDate"] == selected_date]
# Create options list for races (use race Id as the value, but show Course, Time, Race name)
race_options = races_on_date["Id"].tolist()
def format_race_option(race_id):
    info = races_on_date[races_on_date["Id"] == race_id].iloc[0]
    race_time = info["RaceTimeOnly"].strftime("%H:%M")
    return f"{info['Course']} {race_time} – {info['Race']}"
selected_race_id = st.selectbox("Choose Race", race_options, format_func=format_race_option)

# Step 3: Racecard and Predictions
if selected_race_id:
    st.header("3. Racecard and Predictions")
    # Retrieve all horses in the selected race
    race_df = df[df["Id"] == selected_race_id].copy()
    if race_df.empty:
        st.write("No data available for the selected race.")
    else:
        # Get race info for display
        race_info = race_summary[race_summary["Id"] == selected_race_id].iloc[0]
        race_title = f"{race_info['Course']} {race_info['RaceDateTime'].strftime('%d %b %Y %I:%M %p')} – {race_info['Race']}"
        st.subheader(race_title)
        # Display additional race details (Distance, Going, Class, etc.)
        race_distance = race_info["Distance"]
        race_going = race_info["Going"]
        race_class = race_info["Class"] if pd.notna(race_info["Class"]) else "N/A"
        race_prize = race_info["Prize"] if pd.notna(race_info["Prize"]) else 0
        st.markdown(
            f"**Distance:** {race_distance} &nbsp;&nbsp; **Going:** {race_going} &nbsp;&nbsp; "
            f"**Class:** {race_class} &nbsp;&nbsp; **Prize:** £{int(race_prize):,}"
        )

        # Predict win probabilities for each horse in this race
        features = ["SpeedRating", "PaceScore", "SuperAbilityScore", "TrainerWinPct", "JockeyWinPct"]
        X_race = race_df[features].values
        # model.predict_proba returns [prob_not_win, prob_win] for each horse; we take prob_win
        win_probs = model.predict_proba(X_race)[:, 1]
        race_df["PredictedWinProb"] = win_probs
        # Convert probabilities to percentage for display
        race_df["WinProb(%)"] = (race_df["PredictedWinProb"] * 100).round(1)
        # Convert SP (odds) to a friendly string (fractional odds)
        def odds_to_frac(x):
            if pd.isna(x):
                return ""
            try:
                val = float(x)
            except:
                return str(x)
            if val == 1.0:
                return "Evens"
            if 0 < val < 1:
                frac = Fraction(val).limit_denominator(50)
                return f"{frac.numerator}/{frac.denominator}"
            if abs(val - round(val)) < 1e-9:
                return f"{int(round(val))}/1"
            frac = Fraction(val).limit_denominator(50)
            return f"{frac.numerator}/{frac.denominator}"
        race_df["SP"] = race_df["Sp"].apply(odds_to_frac)

        # Prepare dataframe for display: select relevant columns and sort by card number (or as is)
        display_cols = ["CardNo", "HorseName", "Age", "WeightLBS", "OR", "Trainer", "Jockey", "SP", "WinProb(%)"]
        # If advanced metrics should be shown, they can be added to display_cols (SpeedRating, PaceScore, etc.)
        race_df_display = race_df[display_cols].sort_values(by="CardNo")
        # Show the race card as an interactive table
        st.dataframe(race_df_display.reset_index(drop=True), use_container_width=True)

        # Visualization: Bar chart of predicted win probabilities for the horses
        chart_data = race_df_display.copy()
        chart_data.sort_values("WinProb(%)", ascending=False, inplace=True)
        import altair as alt
        chart = alt.Chart(chart_data).mark_bar(color="#4e79a7").encode(
            x=alt.X("WinProb(%)", title="Predicted Win Probability (%)"),
            y=alt.Y("HorseName", sort=None, title="Horse"),
            tooltip=["HorseName", "WinProb(%)", "SP"]
        ).properties(height=max(400, 25 * len(chart_data)), width="container")
        st.altair_chart(chart, use_container_width=True)
