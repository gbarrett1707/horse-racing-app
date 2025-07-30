import streamlit as st
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression

# Page configuration for better layout on wide screens and mobile
st.set_page_config(page_title="Horse Racing Analysis", layout="wide")

# Configure logging for debugging and data parsing info
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

@st.cache_data
def load_data():
    """Load and preprocess the racing data from a compressed CSV file."""
    try:
        # Read the compressed CSV, skipping malformed lines
        df = pd.read_csv("racing_data_upload.csv.gz", compression="gzip", low_memory=False, on_bad_lines="skip")
        logging.info(f"Loaded data file 'racing_data_upload.csv.gz' successfully with shape {df.shape}")
    except Exception as e:
        logging.error(f"Failed to load data file: {e}")
        st.error(f"Error loading data: {e}")
        return None, None

    # Parse RaceDate and RaceTime into a single datetime column
    try:
        date_str = df["RaceDate"].astype(str)
        time_str = df["RaceTime"].astype(str)
        # Extract time portion (HH:MM:SS) from RaceTime (which may include a dummy date)
        time_only = time_str.str.extract(r'(\d{1,2}:\d{2}:\d{2})')[0]
        combined_dt = date_str + " " + time_only
        df["RaceDateTime"] = pd.to_datetime(combined_dt, dayfirst=True, errors="coerce")
        invalid_dates = df["RaceDateTime"].isna().sum()
        if invalid_dates > 0:
            logging.warning(f"{invalid_dates} RaceDateTime entries could not be parsed.")
        # Sort data chronologically by race datetime
        df.sort_values("RaceDateTime", inplace=True)
        logging.info("Parsed RaceDate and RaceTime into a combined datetime and sorted data chronologically.")
    except Exception as e:
        logging.error(f"Error parsing dates/times: {e}")
        df["RaceDateTime"] = pd.NaT

    # Convert columns to numeric where applicable
    numeric_cols = ["Yards", "Seconds", "Age", "OR", "WeightLBS", "Ran", "Prize", "Draw", "CardNo", "Stone", "Lbs"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Extract numeric part of Class (e.g., "Class 4" -> 4)
    if "Class" in df.columns:
        df["Class"] = pd.to_numeric(df["Class"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    # Calculate SpeedRating = Yards / Seconds for valid entries
    df["SpeedRating"] = np.nan
    if "Yards" in df.columns and "Seconds" in df.columns:
        valid_mask = df["Yards"].notna() & df["Seconds"].notna() & (df["Seconds"] > 0)
        df.loc[valid_mask, "SpeedRating"] = df.loc[valid_mask, "Yards"] / df.loc[valid_mask, "Seconds"]
        num_no_speed = len(df) - int(valid_mask.sum())
        logging.warning(f"{num_no_speed} entries have missing or invalid times/distances; SpeedRating not computed for these.")

    # Prepare a summary of unique races for fast selection (one entry per race)
    try:
        races = df[["RaceDateTime", "Course", "Race", "RaceTime"]].drop_duplicates().dropna(subset=["RaceDateTime"])
        races["RaceDate"] = races["RaceDateTime"].dt.date
        races.sort_values("RaceDateTime", inplace=True)
    except Exception as e:
        logging.error(f"Error creating race summary: {e}")
        races = pd.DataFrame(columns=["RaceDateTime", "Course", "Race", "RaceTime", "RaceDate"])

    return df, races

# Load data (with caching to avoid re-reading on every run)
df, race_list = load_data()
if df is None or race_list is None:
    st.stop()  # Stop execution if data failed to load

# Sidebar: Race and horse selection
st.sidebar.header("Select Race and Horse")
# Unique race dates (latest first for convenience)
unique_dates = sorted(race_list["RaceDate"].unique(), reverse=True)
selected_date = st.sidebar.selectbox("Race Date", unique_dates)
# Races on the selected date
races_today = race_list[race_list["RaceDate"] == selected_date].copy()
races_today.sort_values("RaceDateTime", inplace=True)
# Create a display name for each race (e.g., "14:30 - CourseName - RaceTitle")
races_today["RaceDisplay"] = races_today.apply(
    lambda r: f"{pd.to_datetime(r['RaceTime']).strftime('%H:%M') if pd.notna(r['RaceTime']) else '?'} - {r['Course']} - {r['Race']}",
    axis=1
)
race_options = races_today["RaceDisplay"].tolist()
selected_race = st.sidebar.selectbox("Race", race_options)
# Find the selected race details
race_row = races_today[races_today["RaceDisplay"] == selected_race].iloc[0]
race_dt = race_row["RaceDateTime"]
race_course = race_row["Course"]
# Filter main DataFrame for the selected race
race_mask = (df["RaceDateTime"] == race_dt) & (df["Course"] == race_course)
df_race = df[race_mask].copy()
# Sidebar: Horse selection within the race (optional)
horse_names = df_race["HorseName"].tolist()
default_horse_option = "None"
horse_options = [default_horse_option] + horse_names
selected_horse = st.sidebar.selectbox("Horse (optional)", horse_options)
if selected_horse == default_horse_option:
    selected_horse = None

# Prepare function to create feature matrix for model
TYPE_CATEGORIES = []  # Will hold categories of 'Type' seen during training for one-hot encoding
def prepare_features(df_subset, fit=True):
    """Prepare feature matrix X (and target y if fit=True) from a subset of the DataFrame."""
    # Features to use for prediction
    X_df = pd.DataFrame()
    X_df["Age"] = df_subset["Age"]
    X_df["WeightLBS"] = df_subset["WeightLBS"]
    X_df["OR"] = df_subset["OR"]
    X_df["ClassNum"] = df_subset["Class"]
    X_df["Ran"] = df_subset["Ran"]
    X_df["DistanceYards"] = df_subset["Yards"]
    # Indicator if horse was the favorite (Fav or joint Fav)
    fav_flags = df_subset["Favs"].fillna("")
    X_df["isFav"] = ((fav_flags == "Fav") | (fav_flags == "JFav")).astype(int)
    # One-hot encode race type
    type_series = df_subset["Type"].fillna("Unknown")
    global TYPE_CATEGORIES
    if fit:
        # Determine all unique categories during training
        TYPE_CATEGORIES = sorted(type_series.unique().tolist())
    for t in TYPE_CATEGORIES:
        X_df[f"Type_{t}"] = (type_series == t).astype(int)
    # Ensure all columns are present (in case some categories missing in subset when not fitting)
    for t in TYPE_CATEGORIES:
        col = f"Type_{t}"
        if col not in X_df.columns:
            X_df[col] = 0
    # Replace any NaNs in features with 0 (for missing numeric values)
    X_df = X_df.fillna(0)
    # Target variable: 1 if won the race (FPos == 1), 0 otherwise
    y = None
    if fit:
        y = (pd.to_numeric(df_subset["FPos"], errors="coerce") == 1).astype(int).values
    return (X_df.values, y, X_df.columns.tolist()) if fit else (X_df.values, X_df.columns.tolist())

@st.cache_resource
def train_model():
    """Train the machine learning model to predict win probabilities."""
    # Prepare training data (features and target)
    X, y, feature_names = prepare_features(df, fit=True)
    # Split data into training and testing sets for evaluation
    from sklearn.model_selection import train_test_split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except Exception as e:
        # If stratify fails (e.g., if only one class present, which shouldn't happen here), do a simple split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000, solver="saga", random_state=42)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        st.error(f"Model training failed: {e}")
        return None
    # Evaluate on the test set
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = np.nan
    logging.info(f"Model trained. Test Accuracy: {acc:.4f}, Test AUC: {auc:.4f}")
    # Store metrics and feature names in the model object for later use
    model.acc_ = acc
    model.auc_ = auc
    model.feature_names_ = feature_names
    return model

# Train or retrieve the model (cached to avoid re-training on each run)
model = train_model()

# Create tabs in the main interface
tab_explore, tab_model, tab_race = st.tabs(["Data Exploration", "Model Predictions", "Racecard"])

# Tab 1: Data Exploration
with tab_explore:
    st.markdown("## Data Exploration")
    st.write("Explore the historical horse racing dataset, view overall trends, and inspect individual horse performance.")
    # Overall summary statistics
    total_races = race_list.shape[0]
    total_entries = len(df)
    unique_horses = df["HorseName"].nunique()
    st.write(f"**Total Races:** {total_races:,} &nbsp;&nbsp; **Total Entries:** {total_entries:,} &nbsp;&nbsp; **Unique Horses:** {unique_horses:,}")
    # SpeedRating distribution by race type (density plot)
    st.markdown("**Speed Rating Distribution by Race Type**")
    # Use a sample for plotting to reduce data size if very large
    plot_sample = df if len(df) < 100000 else df.sample(100000, random_state=42)
    plot_sample = plot_sample.dropna(subset=["SpeedRating", "Type"])
    # Map short type codes to readable labels
    type_map = {"f": "Flat", "c": "Chase", "h": "Hurdle", "b": "NH Flat", "Unknown": "Unknown"}
    plot_sample["RaceType"] = plot_sample["Type"].fillna("Unknown").map(type_map)
    # Create an Altair density chart for speed ratings
    import altair as alt
    density_chart = alt.Chart(plot_sample).transform_density(
        density="SpeedRating",
        groupby=["RaceType"],
        as_=["SpeedRating", "density"]
    ).mark_area(opacity=0.5).encode(
        x=alt.X("SpeedRating:Q", title="SpeedRating (Yards/Second)"),
        y="density:Q",
        color="RaceType:N"
    )
    st.altair_chart(density_chart, use_container_width=True)
    # Trend: average SpeedRating over years
    st.markdown("**Average SpeedRating Over Years**")
    df_year = df.dropna(subset=["SpeedRating"]).copy()
    df_year["Year"] = df_year["RaceDateTime"].dt.year
    yearly_speed = df_year.groupby("Year")["SpeedRating"].mean().reset_index()
    line_chart = alt.Chart(yearly_speed).mark_line(point=True).encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("SpeedRating:Q", title="Average SpeedRating")
    )
    st.altair_chart(line_chart, use_container_width=True)
    # If a specific horse is selected, show its performance trend
    if selected_horse:
        st.markdown(f"**Performance of {selected_horse}**")
        horse_data = df[df["HorseName"] == selected_horse].copy()
        if not horse_data.empty:
            horse_data.sort_values("RaceDateTime", inplace=True)
            horse_data["Date"] = horse_data["RaceDateTime"].dt.date
            # Mark win (1) or not (0) for each race
            horse_data["WonRace"] = (pd.to_numeric(horse_data["FPos"], errors="coerce") == 1).astype(int)
            # Plot win history as a line (0/1 over time)
            win_chart = alt.Chart(horse_data).mark_line(point=True).encode(
                x=alt.X("Date:T", title="Race Date"),
                y=alt.Y("WonRace:Q", title="Won Race (1=yes, 0=no)")
            )
            st.altair_chart(win_chart, use_container_width=True)
            # Plot Official Rating (OR) over time if available
            if horse_data["OR"].notna().any():
                or_chart = alt.Chart(horse_data.dropna(subset=["OR"])).mark_line(point=True).encode(
                    x=alt.X("Date:T", title="Race Date"),
                    y=alt.Y("OR:Q", title="Official Rating")
                )
                st.altair_chart(or_chart, use_container_width=True)
            else:
                st.write("*(No official ratings available for this horse.)*")
        else:
            st.write("*(No historical records found for the selected horse.)*")

# Tab 2: Model Predictions
with tab_model:
    st.markdown("## Model Predictions")
    if model is None:
        st.write("Model could not be trained. Please check data and retry.")
    else:
        # Display model performance metrics
        if hasattr(model, "acc_") and hasattr(model, "auc_"):
            st.write(f"**Model Accuracy (Test set):** {model.acc_:.3f}")
            st.write(f"**Model AUC (Test set):** {model.auc_:.3f}")
        # Feature importance or coefficients
        st.markdown("**Feature Importance**")
        if isinstance(model, LogisticRegression):
            # For logistic regression, show coefficients
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            feature_names = model.feature_names_ if hasattr(model, "feature_names_") else None
            if feature_names is None:
                # If feature names not stored, assume order from prepare_features
                feature_names = TYPE_CATEGORIES.copy()
                base_feats = ["Age", "WeightLBS", "OR", "ClassNum", "Ran", "DistanceYards", "isFav"]
                # Combine base features and one-hot type features
                feature_names = base_feats + [f"Type_{t}" for t in TYPE_CATEGORIES]
            coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef.flatten()})
            coef_df["Importance"] = coef_df["Coefficient"].abs()
            coef_df.sort_values("Importance", ascending=False, inplace=True)
            # Bar chart of coefficients
            coef_chart = alt.Chart(coef_df).mark_bar().encode(
                x=alt.X("Feature:N", sort=None, title="Feature"),
                y=alt.Y("Coefficient:Q", title="Coefficient"),
                color=alt.condition(alt.datum.Coefficient > 0, alt.value("#4caf50"), alt.value("#f44336"))
            )
            st.altair_chart(coef_chart, use_container_width=True)
        else:
            # For tree-based models if any (not used here, but just in case)
            if hasattr(model, "feature_importances_"):
                feature_names = model.feature_names_ if hasattr(model, "feature_names_") else []
                fi_df = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
                fi_df.sort_values("Importance", ascending=False, inplace=True)
                st.bar_chart(fi_df.set_index("Feature")["Importance"])
            else:
                st.write("Feature importance not available for this model.")

        st.write("The model is trained on historical race data to predict the probability of a horse winning a race based on its features (age, weight, rating, etc.).")

# Tab 3: Racecard
with tab_race:
    st.markdown("## Racecard")
    # Race details heading
    race_date_str = race_dt.strftime("%Y-%m-%d") if pd.notna(race_dt) else str(selected_date)
    race_time_str = pd.to_datetime(race_row["RaceTime"]).strftime("%H:%M") if pd.notna(race_row["RaceTime"]) else "??:??"
    st.write(f"**{race_course} – {race_date_str} – {race_time_str} – {race_row['Race']}**")
    if df_race.empty:
        st.write("*(No data available for the selected race.)*")
    else:
        # Prepare race data for display
        display_cols = ["HorseName", "Draw", "Age", "WeightLBS", "OR", "Jockey", "Trainer", "Sp", "FPos"]
        df_display = df_race[display_cols].copy()
        # Mark the winner and compute model probabilities for each horse
        df_display["Winner"] = (pd.to_numeric(df_display["FPos"], errors="coerce") == 1)
        if model is not None:
            X_race, feature_cols = prepare_features(df_race, fit=False)
            try:
                win_probs = model.predict_proba(X_race)[:, 1]
            except Exception:
                win_probs = np.zeros(len(df_race))
        else:
            win_probs = np.zeros(len(df_race))
        df_display["PredictedWinProb"] = win_probs
        # Sort by finishing position (winners first, then others by position number)
        df_display["PosNum"] = pd.to_numeric(df_display["FPos"], errors="coerce").fillna(999).astype(int)
        df_display.sort_values("PosNum", inplace=True)
        df_display.drop(columns=["PosNum"], inplace=True)
        # Rename columns for clarity
        df_display.rename(columns={
            "HorseName": "Horse", "WeightLBS": "Weight (lbs)", "OR": "Official Rating",
            "Sp": "SP", "FPos": "Finish"
        }, inplace=True)
        # Highlight winner and top predicted probability
        styler = df_display.style.format({"PredictedWinProb": "{:.3f}"}).apply(
            lambda vals: ["background-color: gold" if v is True else "" for v in vals], subset=["Winner"]
        ).highlight_max(subset=["PredictedWinProb"], color="lightgreen")
        st.dataframe(styler, use_container_width=True)
        # Optionally, highlight the selected horse in the table
        if selected_horse:
            st.write(f"Selected horse: **{selected_horse}** (highlighted above if present).")
        # Explain table
        st.caption("🏆 Winner is highlighted in gold. The highest predicted win probability is highlighted in green.")
