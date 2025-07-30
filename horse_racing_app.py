import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
from fractions import Fraction

# Set page config for better layout on mobile
st.set_page_config(page_title="Horse Racing Predictor", layout="wide")

# Configure logging for debug info
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Function to convert odds to fractional string
def frac_str(x):
    if pd.isna(x):
        return ""
    # Convert float (fractional odds like 0.8333) to nice string (e.g., "5/6")
    frac = Fraction(x).limit_denominator(100)  # limit denominator for typical odds
    num, den = frac.numerator, frac.denominator
    if den == 1:
        return f"{num}/1"
    else:
        return f"{num}/{den}"

# Cache data loading and preprocessing
@st.cache_data(show_spinner=False)
def load_data():
    try:
        # Load the compressed CSV data
        df = pd.read_csv("racing_data_upload.csv.gz")
        logging.info(f"Loaded data file 'racing_data_upload.csv.gz' successfully with shape {df.shape}")
    except FileNotFoundError:
        logging.error("Data file not found.")
        st.error("❗ Data file 'racing_data_upload.csv.gz' not found. Please upload the data file and rerun.")
        return None, None
    except Exception as e:
        logging.exception("Failed to load data file.")
        st.error(f"❗ Failed to load data: {e}")
        return None, None

    # Ensure required columns exist
    required_cols = ["RaceDate", "RaceTime", "Seconds", "Yards"]
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            st.error(f"❗ Data is missing required column '{col}'.")
            return None, None

    # Parse RaceDate and RaceTime into a single datetime
    date_part = df["RaceDate"].astype(str).str.split().str[0]
    time_part = df["RaceTime"].astype(str).str.split().str[-1]
    df["RaceDateTime"] = pd.to_datetime(date_part + " " + time_part, dayfirst=True, errors="coerce")

    # Warn if any dates could not be parsed
    failed_dates = df["RaceDateTime"].isna().sum()
    if failed_dates > 0:
        logging.warning(f"{failed_dates} RaceDateTime entries could not be parsed.")
        st.warning("⚠️ Some race dates/times could not be parsed. These entries will be ignored.")
        # Drop entries with unknown date/time
        df = df[df["RaceDateTime"].notna()]

    # Sort data chronologically by race date/time
    df = df.sort_values("RaceDateTime").reset_index(drop=True)
    logging.info("Parsed RaceDate and RaceTime into a combined datetime and sorted data chronologically.")

    # Convert relevant columns to numeric for calculations
    df["Seconds_numeric"] = pd.to_numeric(df["Seconds"], errors="coerce")
    df["Yards_numeric"] = pd.to_numeric(df["Yards"], errors="coerce")
    df["TotalBtn_numeric"] = pd.to_numeric(df["TotalBtn"], errors="coerce")
    df["FPos_numeric"] = pd.to_numeric(df["FPos"], errors="coerce")  # numeric finishing position (NaN if not placed)

    # Calculate SpeedRating = Yards / Seconds for valid entries
    df["SpeedRating"] = np.nan
    valid_speed = df["Seconds_numeric"].notna() & (df["Seconds_numeric"] > 0) & df["Yards_numeric"].notna() & (~df["FPos_numeric"].isna())
    df.loc[valid_speed, "SpeedRating"] = df.loc[valid_speed, "Yards_numeric"] / df.loc[valid_speed, "Seconds_numeric"]
    num_no_speed = len(df) - valid_speed.sum()
    if num_no_speed > 0:
        logging.warning(f"{num_no_speed} entries have missing or invalid times/distances; SpeedRating not computed for these.")

    # Calculate HorseTime (approximate finish time for each horse) = winner time + 0.2s per length behind
    df["HorseTime"] = df["Seconds_numeric"] + 0.2 * df["TotalBtn_numeric"]

    # Compute historical performance metrics for each horse
    df["RaceCount"] = df.groupby("HorseName").cumcount() + 1  # number of races for horse up to current
    df["PreviousRuns"] = df["RaceCount"] - 1
    # Mark winners
    df["Win"] = (df["FPos_numeric"] == 1).astype(int)
    # Cumulative wins prior to current race
    df["PreviousWins"] = df.groupby("HorseName")["Win"].cumsum().shift(fill_value=0)

    # Calculate days since last run for each horse
    df["PrevRaceDate"] = df.groupby("HorseName")["RaceDateTime"].shift()
    df["DaysSinceLast"] = (df["RaceDateTime"] - df["PrevRaceDate"]).dt.days
    df["DaysSinceLast"] = df["DaysSinceLast"].fillna(0).astype(int)

    # Feature engineering: prepare columns for model
    # Convert categorical and fill missing numeric values
    df["OR"] = pd.to_numeric(df["OR"], errors="coerce").fillna(0).astype(int)       # Official Rating (0 if missing)
    df["Allow"] = pd.to_numeric(df["Allow"], errors="coerce").fillna(0).astype(int)  # Jockey allowance (lbs, 0 if none)
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)  # Race class (0 if missing)
    df["Sp"] = pd.to_numeric(df["Sp"], errors="coerce")  # Starting price odds (fractional, as float)
    df["FirstTimeRunner"] = (df["PreviousRuns"] == 0).astype(int)  # 1 if horse's first race

    # Handle race type: blank means Flat
    df["Type"] = df["Type"].fillna("").replace("", "Flat")
    # One-hot encode race type (Flat, hurdle, chase, etc.)
    df = pd.get_dummies(df, columns=["Type"], prefix="Type")
    # Ensure all expected type columns exist (in case some types not in data subset)
    for col in ["Type_Flat", "Type_h", "Type_c", "Type_b"]:
        if col not in df.columns:
            df[col] = 0

    # Prepare race summary table for selecting races
    race_summary = df.groupby("Id").agg({
        "RaceDateTime": "first",
        "Course": "first",
        "Race": "first",
        "Ran": "first"
    }).reset_index()
    race_summary["RaceDate"] = pd.to_datetime(race_summary["RaceDateTime"]).dt.date

    return df, race_summary

# Cache model training to avoid recomputation on each run
@st.cache_resource(show_spinner=False)
def train_model(dataframe):
    # Define feature columns for the model
    feature_cols = [
        "Age", "WeightLBS", "OR", "Allow", "Yards_numeric", "Class", "Ran", "Sp",
        "PreviousRuns", "PreviousWins", "FirstTimeRunner", "DaysSinceLast",
        "Type_Flat", "Type_h", "Type_c", "Type_b"
    ]
    # Target: whether horse won the race (Win column)
    X = dataframe[feature_cols]
    y = dataframe["Win"]

    # Split data into train and test sets for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Train a gradient boosting classifier to predict win probability
    model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logging.info(f"Model trained. Test Accuracy: {acc:.3f}, AUC: {auc:.3f}")

    # Get feature importances for analysis
    feat_names = feature_cols
    try:
        importances = model.feature_importances_
    except AttributeError:
        # If model doesn't have feature_importances_ (should have for tree-based)
        importances = np.zeros(len(feature_cols))
    return model, acc, auc, feat_names, importances

# Load and process data
df, race_summary = load_data()
if df is None or race_summary is None:
    st.stop()

# Display title and introduction
st.title("🏇 Horse Racing Performance and Prediction")
st.markdown(
    "This app loads historical horse racing data, computes performance metrics, and uses a machine learning model to predict each horse's win probability in a race. "
    "Explore past races with predicted win chances and compare them to the actual results and betting odds. 📊"
)
st.info("**SpeedRating** is calculated as **Yards/Seconds** for each run, indicating the horse's average speed over the race distance.")

# Train the prediction model
with st.spinner("Training prediction model..."):
    model, acc, auc, feat_names, feat_importances = train_model(df)

# Section: Racecard Builder and Predictions
st.subheader("🏁 Race Results and Predicted Win Chances")
# Let user select a race date and then a specific race
unique_dates = sorted(race_summary["RaceDate"].unique(), reverse=True)
selected_date = st.selectbox("Select Race Date", unique_dates, format_func=lambda d: d.strftime("%d %b %Y"))
# Filter races on that date and sort by time
races_on_date = race_summary[race_summary["RaceDate"] == selected_date].sort_values("RaceDateTime")
# Create a mapping from race Id to display label
race_options = {}
for _, row in races_on_date.iterrows():
    race_time = pd.to_datetime(row["RaceDateTime"]).strftime("%H:%M")
    race_label = f"{race_time} - {row['Course']} - {row['Race']}"
    race_options[row["Id"]] = race_label
selected_race_id = st.selectbox("Select Race", list(race_options.keys()), format_func=lambda x: race_options[x])

# Get the selected race data
race_df = df[df["Id"] == selected_race_id].copy()
if race_df.empty:
    st.write("No data available for the selected race.")
else:
    # Sort by card number (race order)
    if "CardNo" in race_df.columns:
        race_df.sort_values("CardNo", inplace=True)
    # Predict win probabilities for each horse in the race
    feature_cols = [
        "Age", "WeightLBS", "OR", "Allow", "Yards_numeric", "Class", "Ran", "Sp",
        "PreviousRuns", "PreviousWins", "FirstTimeRunner", "DaysSinceLast",
        "Type_Flat", "Type_h", "Type_c", "Type_b"
    ]
    X_race = race_df[feature_cols]
    race_df["PredProb"] = model.predict_proba(X_race)[:, 1]

    # Build race info header
    race_info = race_summary[race_summary["Id"] == selected_race_id].iloc[0]
    race_time_str = pd.to_datetime(race_info["RaceDateTime"]).strftime("%d %b %Y %I:%M %p")
    st.markdown(f"**Race:** {race_info['Race']}  \n**Date & Time:** {race_time_str} at {race_info['Course']}  \n**Runners:** {int(race_info['Ran'])}")

    # Prepare race table with results and predictions
    # Mark the winner with a trophy emoji
    winner_mask = (race_df["FPos_numeric"] == 1)
    race_df.loc[winner_mask, "HorseName"] = "🏆 " + race_df.loc[winner_mask, "HorseName"].astype(str)
    # Create weight in st-lb format
    race_df["Weight"] = race_df.apply(lambda r: f"{int(r['WeightLBS']//14)}-{int(r['WeightLBS']%14)}", axis=1)
    # Create fractional odds string for SP
    race_df["SP"] = race_df["Sp"].apply(frac_str)
    # Calculate odds-implied probability (%)
    race_df["Odds%"] = race_df["Sp"].apply(lambda x: f"{(100/(x+1)):.1f}%" if pd.notna(x) and x >= 0 else "N/A")
    # Format model win probability (%)
    race_df["WinChance%"] = (race_df["PredProb"] * 100).round(1).astype(str) + "%"
    # Prepare finish position (including non-finish codes)
    race_df["Finish"] = race_df["FPos"].astype(str)

    # Replace '0' OR with blank for display (horses without an official rating)
    race_df["OR"] = race_df["OR"].replace(0, np.nan)
    # Select and rename columns for display
    display_cols = ["HorseName", "Age", "Weight", "OR", "SP", "Odds%", "WinChance%", "Finish"]
    display_df = race_df[display_cols].copy()
    display_df.rename(columns={
        "HorseName": "Horse",
        "OR": "OR", 
        "SP": "SP (Odds)", 
        "Odds%": "Odds (%)", 
        "WinChance%": "Predicted (%)", 
        "Finish": "Finish"
    }, inplace=True)
    # Show the race table
    st.dataframe(display_df, use_container_width=True)

# Section: Horse Performance Lookup
with st.expander("🔍 Look up a Horse's Performance History"):
    st.subheader("📈 Horse Performance Trends")
    all_horses = sorted(df["HorseName"].unique())
    selected_horse = st.selectbox("Select Horse", all_horses)
    horse_data = df[df["HorseName"] == selected_horse].copy()
    if horse_data.empty:
        st.write("No data found for the selected horse.")
    else:
        # Calculate summary stats
        total_runs = len(horse_data)
        total_wins = horse_data["Win"].sum()
        win_rate = total_wins / total_runs if total_runs > 0 else 0.0

        # Sort horse data by date
        horse_data.sort_values("RaceDateTime", inplace=True)
        # Speed trend chart (if any speed data available)
        speed_series = horse_data.dropna(subset=["SpeedRating"])
        # Display performance metrics
        cols = st.columns(3)
        cols[0].metric("Total Runs", int(total_runs))
        cols[1].metric("Total Wins", int(total_wins))
        cols[2].metric("Win Rate", f"{win_rate:.1%}")
        # If at least one SpeedRating available, show last speed and trend
        if not speed_series.empty:
            last_speed = speed_series["SpeedRating"].iloc[-1]
            prev_speed = speed_series["SpeedRating"].iloc[-2] if len(speed_series) >= 2 else None
            if prev_speed is not None:
                # Show last speed rating with change vs previous run
                st.metric("Last SpeedRating (yd/s)", f"{last_speed:.2f}", delta=f"{(last_speed - prev_speed):.2f}", delta_color="normal")
            else:
                st.metric("Last SpeedRating (yd/s)", f"{last_speed:.2f}")
            # Create speed rating over time chart
            chart_data = horse_data.copy()
            chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x=alt.X("RaceDateTime:T", title="Race Date"),
                y=alt.Y("SpeedRating", title="SpeedRating (yards/sec)"),
                tooltip=[
                    alt.Tooltip("RaceDateTime:T", title="Date"),
                    alt.Tooltip("SpeedRating", title="SpeedRating", format=".2f"),
                    alt.Tooltip("FPos", title="Finish")
                ]
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("*(SpeedRating not available for this horse, possibly due to all runs being incomplete.)*")

        # Show recent race history for the horse
        horse_data["Date"] = pd.to_datetime(horse_data["RaceDateTime"]).dt.strftime("%d %b %Y")
        horse_data["Finish"] = horse_data["FPos"].astype(str)
        horse_data["SP"] = horse_data["Sp"].apply(frac_str)
        recent_runs = horse_data[["Date", "Course", "Distance", "Finish", "SP"]].copy()
        if len(recent_runs) > 10:
            st.write(f"Last {min(10, len(recent_runs))} runs:")
            st.table(recent_runs.tail(10).reset_index(drop=True))
            with st.expander("Show all runs"):
                st.dataframe(recent_runs.reset_index(drop=True), use_container_width=True)
        else:
            st.table(recent_runs.reset_index(drop=True))

# Section: Model Performance (for the curious)
with st.expander("🧠 Model Performance and Feature Importance"):
    st.subheader("Model Evaluation")
    st.write(f"**Test Accuracy:** {acc:.2%}   |   **ROC AUC:** {auc:.3f}")
    # Feature importance bar chart
    importance_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_importances})
    importance_df.sort_values("Importance", ascending=False, inplace=True)
    chart_imp = alt.Chart(importance_df).mark_bar().encode(
        x=alt.X("Importance:Q", title="Importance"),
        y=alt.Y("Feature:N", sort="-x", title="Feature")
    )
    st.altair_chart(chart_imp, use_container_width=True)
