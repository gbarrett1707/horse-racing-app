import streamlit as st
import pandas as pd
import numpy as np
import joblib  # for loading a pre-trained model if available
import plotly.express as px

# Set page configuration for mobile-friendly layout
st.set_page_config(page_title="Horse Racing Predictor & Racecard Builder", page_icon="🐎", layout="wide")

# App title and description
st.title("🏇 Horse Racing Predictor & Racecard Builder")
st.write("Explore horse racing data, view derived performance metrics, and predict win probabilities for each horse. Select a race from the sidebar to get started.")

@st.cache_data
def load_data():
    """Load historical horse racing data from CSV files (supports combined file or multiple part files)."""
    import os
    df = None
    # Try loading a single combined CSV (or compressed CSV) if it exists
    for file_name in ["racing_data.csv", "racing_data.csv.gz", "racing_data_upload.csv", "racing_data_upload.csv.gz"]:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            break
    # If no single file, try loading multiple part files and concatenate
    if df is None:
        part_dfs = []
        part_num = 1
        while True:
            part_file = f"racing_data_part_{part_num}.csv"
            if os.path.exists(part_file):
                part_df = pd.read_csv(part_file)
                part_dfs.append(part_df)
                part_num += 1
            else:
                break
        if part_dfs:
            df = pd.concat(part_dfs, ignore_index=True)
    if df is None:
        st.error("Data files not found. Please ensure the racing data CSV files are available.")
        return None
    # Optionally drop unused columns to save memory (e.g., long text comments)
    if 'Comments' in df.columns:
        df.drop('Comments', axis=1, inplace=True)
    # Parse RaceDate as datetime (date only) and RaceTime as time
    if 'RaceDate' in df.columns:
        try:
            df['RaceDate'] = pd.to_datetime(df['RaceDate'], format="%d/%m/%y %H:%M:%S")
        except:
            df['RaceDate'] = pd.to_datetime(df['RaceDate'])
    if 'RaceTime' in df.columns:
        try:
            df['RaceTime'] = pd.to_datetime(df['RaceTime'], format="%m/%d/%y %H:%M:%S")
        except:
            df['RaceTime'] = pd.to_datetime(df['RaceTime'])
    # If RaceTime is full datetime (with a dummy date), extract just the time component for grouping
    if 'RaceTime' in df.columns and df['RaceTime'].dtype == 'datetime64[ns]':
        df['RaceTimeOnly'] = df['RaceTime'].dt.time
    # Compute winner's finish time (Seconds) for each race to use for speed calculations
    if 'Seconds' in df.columns:
        race_key_cols = ['Course', 'RaceDate']
        race_time_col = 'RaceTime'
        if 'RaceTimeOnly' in df.columns:
            # Use time-only for grouping if available (RaceDate is already date)
            race_key_cols.append('RaceTimeOnly')
            race_time_col = 'RaceTimeOnly'
        else:
            race_key_cols.append('RaceTime')
        winners = df[df['FPos'] == 1][race_key_cols + ['Seconds']].copy()
        winners.rename(columns={'Seconds': 'WinSeconds'}, inplace=True)
        df = df.merge(winners, on=race_key_cols, how='left')
    else:
        df['WinSeconds'] = np.nan
    # Calculate each horse's actual race time in seconds (estimate for non-winners using lengths behind)
    if 'WinSeconds' in df.columns:
        def actual_time(row):
            if pd.isna(row['WinSeconds']):
                return np.nan
            if 'FPos' in row and row['FPos'] == 1:
                return row['WinSeconds']  # winner's time
            # If not winner, add time based on lengths behind (TotalBtn) and an estimated conversion factor
            if 'TotalBtn' in row and pd.notna(row['TotalBtn']):
                # Use a different length-to-seconds conversion for flat vs jumps
                factor = 0.2 if ('Type' in row and str(row['Type']).lower().startswith('f')) else 0.3
                return row['WinSeconds'] + row['TotalBtn'] * factor
            else:
                # No distance-behind info, assume finished around the same time as winner
                return row['WinSeconds']
        df['ActualSeconds'] = df.apply(actual_time, axis=1)
    else:
        df['ActualSeconds'] = np.nan
    # Compute speed rating for each run (yards per second adjusted for weight, scaled for readability)
    if 'Yards' in df.columns and 'ActualSeconds' in df.columns:
        def baseline_weight(row):
            # Baseline weight: use different baselines for flat vs jumps (approximate)
            return 130.0 if ('Type' in row and str(row['Type']).lower().startswith('f')) else 160.0
        if 'WeightLBS' in df.columns:
            baseline = df.apply(baseline_weight, axis=1)
            weight_factor = df['WeightLBS'] / baseline
        else:
            weight_factor = 1.0
        df['SpeedRating'] = (df['Yards'] / df['ActualSeconds']) * weight_factor * 5  # scale factor 5 to get a nicer range
    else:
        df['SpeedRating'] = np.nan
    return df

@st.cache_data
def compute_trainer_jockey_stats(df):
    """Compute overall win% and place% for each trainer and jockey from the data."""
    trainer_stats_df = None
    jockey_stats_df = None
    if df is None:
        return None, None
    data = df.copy()
    # Group by trainer and calculate wins, places, total runs
    if 'Trainer' in data.columns:
        trainer_group = data.groupby('Trainer')
        total_runs = trainer_group.size()
        wins = trainer_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') == 1).sum())
        places = trainer_group['FPos'].apply(lambda s: (pd.to_numeric(s, errors='coerce') <= 3).sum())
        trainer_stats_df = pd.DataFrame({'Runs': total_runs, 'Wins': wins, 'Places': places})
        trainer_stats_df['WinPct'] = (trainer_stats_df['Wins'] / trainer_stats_df['Runs'] * 100).round(1)
        trainer_stats_df['PlacePct'] = (trainer_stats_df['Places'] / trainer_stats_df['Runs'] * 100).round(1)
    # Group by jockey and calculate wins, places, total rides
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
    """Compute PaceScore and SuperAbilityScore for given horses based on their past SpeedRatings."""
    pace_scores = {}
    ability_scores = {}
    if df is None:
        return pace_scores, ability_scores
    for horse in horses:
        # All runs for this horse
        horse_hist = df[df['HorseName'] == horse].copy()
        if horse_hist.empty:
            pace_scores[horse] = 0.0
            ability_scores[horse] = 0.0
            continue
        # Sort chronologically by date and time
        horse_hist.sort_values(['RaceDate', 'RaceTime'], inplace=True)
        # Exclude the current race (if included in history and race_key provided) to use only prior runs
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
            # No past races (debut runner)
            pace_scores[horse] = 0.0
            ability_scores[horse] = 0.0
        else:
            # PaceScore = SpeedRating of last run (most recent race)
            last_speed = horse_hist['SpeedRating'].iloc[-1]
            # SuperAbilityScore = combination of recent average ability and last run (simple formula)
            recent_speeds = horse_hist['SpeedRating'].tail(3)
            avg_recent = recent_speeds.mean()
            super_score = (last_speed + avg_recent) / 2  # combine last performance with recent average
            pace_scores[horse] = round(last_speed, 1) if pd.notna(last_speed) else 0.0
            ability_scores[horse] = round(super_score, 1) if pd.notna(super_score) else 0.0
    return pace_scores, ability_scores

def predict_win_probabilities(race_df):
    """Predict win probabilities for the horses in the given race using a trained model if available, otherwise a heuristic."""
    if race_df.empty:
        return []
    # Try to load a pre-trained model (placeholder path)
    model = None
    try:
        model = joblib.load("win_model.pkl")
    except:
        model = None
    if model:
        # Prepare feature matrix for the model
        feature_cols = ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']
        X = race_df[feature_cols].fillna(0.0).astype(float)
        # Get win probabilities from model (assuming binary classification with positive class = win)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            win_probs = probs[:, 1]
        else:
            # If model doesn't have predict_proba, assume predict outputs a probability or score
            win_probs = model.predict(X)
        return win_probs
    else:
        # No model available: use a simple weighted heuristic combining stats
        df = race_df.copy()
        # Ensure required columns exist and fill missing with 0
        for col in ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']:
            if col not in df.columns:
                df[col] = 0.0
        # Normalize each feature to 0-1 range within this race
        for col in ['SpeedRating', 'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct']:
            max_val = df[col].max()
            if pd.notna(max_val) and max_val > 0:
                df[col] = df[col] / max_val
            else:
                df[col] = 0.0
        # Assign weights to features (tweak these as needed)
        # Here we use equal weighting for simplicity
        df['Score'] = (df['SpeedRating'] + df['PaceScore'] + df['SuperAbilityScore'] + df['TrainerWinPct'] + df['JockeyWinPct']) / 5.0
        # Convert scores to probabilities (normalize to sum to 1)
        total_score = df['Score'].sum()
        if total_score == 0:
            # If all scores are zero (e.g., all debut horses), assign equal probability
            probs = np.array([1.0 / len(df)] * len(df))
        else:
            probs = df['Score'] / total_score
        return probs.values

# Load data
with st.spinner("Loading data..."):
    df = load_data()

if df is not None:
    # Pre-compute trainer and jockey statistics (cached)
    trainer_stats, jockey_stats = compute_trainer_jockey_stats(df)
    
    # Sidebar controls for race selection
    st.sidebar.header("Select Race")
    selected_date = None
    if 'RaceDate' in df.columns:
        # List of available race dates
        available_dates = sorted(pd.to_datetime(df['RaceDate'].dt.date.unique()).date)
        if available_dates:
            # Default to the most recent date
            default_date = available_dates[-1]
            selected_date = st.sidebar.date_input("Race Date", value=default_date, 
                                                  min_date=min(available_dates), max_date=max(available_dates))
    if selected_date:
        # Filter data for the selected date
        date_mask = df['RaceDate'].dt.date == selected_date
        day_data = df[date_mask]
        # Courses on that date
        courses = sorted(day_data['Course'].unique())
        if courses:
            selected_course = st.sidebar.selectbox("Course", courses)
            # Filter for selected course on that date
            course_mask = (day_data['Course'] == selected_course)
            course_data = day_data[course_mask]
            # Get list of races at this course (unique by time & name)
            races = []
            for time_val, race_name in course_data[['RaceTime', 'Race']].drop_duplicates().values:
                # Format time for display
                if pd.notna(time_val):
                    try:
                        t_str = pd.to_datetime(str(time_val)).strftime("%H:%M")
                    except:
                        # If already a time object or string
                        t_str = str(time_val)[-8:] if ":" in str(time_val) else str(time_val)
                else:
                    t_str = ""
                label = f"{t_str} - {race_name}" if t_str else race_name
                races.append((time_val, label))
            races = sorted(races, key=lambda x: x[0])  # sort by time
            race_labels = [lbl for _, lbl in races]
            selected_label = st.sidebar.selectbox("Race", race_labels) if race_labels else None
            # Determine which race was selected and filter data for that race
            race_df = pd.DataFrame()
            if selected_label:
                # Find the corresponding time for the selected label
                for time_val, lbl in races:
                    if lbl == selected_label:
                        selected_time = time_val
                        break
                # Filter main data by this unique race (date, course, time)
                race_mask = (df['Course'] == selected_course) & (df['RaceDate'].dt.date == selected_date)
                if 'RaceTimeOnly' in df.columns and pd.notna(selected_time):
                    # If we have a separate time-only column
                    race_mask &= (df['RaceTimeOnly'] == (selected_time if not isinstance(selected_time, pd.Timestamp) else selected_time.time()))
                else:
                    race_mask &= (df['RaceTime'] == selected_time)
                race_df = df[race_mask].copy()
        else:
            race_df = pd.DataFrame()
    else:
        race_df = pd.DataFrame()
    
    # If a race is selected (race_df not empty), display the analysis
    if not race_df.empty:
        # Define a unique key for this race (course, date, time) to pass to functions
        selected_race_key = None
        try:
            race_time_key = race_df.iloc[0]['RaceTimeOnly'] if 'RaceTimeOnly' in race_df.columns else race_df.iloc[0]['RaceTime']
            selected_race_key = (selected_course, pd.to_datetime(selected_date), race_time_key)
        except:
            pass
        # List of horses in this race
        horses = race_df['HorseName'].unique()
        # Compute PaceScore and SuperAbilityScore for these horses
        pace_scores, ability_scores = compute_pace_and_ability(df, horses, race_key=selected_race_key)
        race_df['PaceScore'] = race_df['HorseName'].map(pace_scores)
        race_df['SuperAbilityScore'] = race_df['HorseName'].map(ability_scores)
        # Add trainer and jockey stats (win%, place%) to the race data
        if trainer_stats is not None:
            race_df['TrainerWinPct'] = race_df['Trainer'].map(trainer_stats['WinPct'])
            race_df['TrainerPlacePct'] = race_df['Trainer'].map(trainer_stats['PlacePct'])
        else:
            race_df['TrainerWinPct'] = 0.0
            race_df['TrainerPlacePct'] = 0.0
        if jockey_stats is not None:
            race_df['JockeyWinPct'] = race_df['Jockey'].map(jockey_stats['WinPct'])
            race_df['JockeyPlacePct'] = race_df['Jockey'].map(jockey_stats['PlacePct'])
        else:
            race_df['JockeyWinPct'] = 0.0
            race_df['JockeyPlacePct'] = 0.0
        # Predict win probabilities for the race
        win_probs = predict_win_probabilities(race_df)
        race_df['PredictedWinProb'] = (win_probs * 100).round(1)  # as percentage
        
        # Display race info and table of horses with stats
        st.subheader(f"Race: {selected_course} — {selected_label}")
        # Prepare a cleaner table for display
        display_cols = ['HorseName', 'Trainer', 'Jockey', 'PaceScore', 'SuperAbilityScore', 
                        'TrainerWinPct', 'JockeyWinPct', 'PredictedWinProb']
        display_df = race_df[display_cols].copy()
        display_df.rename(columns={
            'HorseName': 'Horse',
            'PaceScore': 'PaceScore',
            'SuperAbilityScore': 'SuperAbility',
            'TrainerWinPct': 'Trainer Win%', 
            'JockeyWinPct': 'Jockey Win%', 
            'PredictedWinProb': 'Predicted Win %'
        }, inplace=True)
        st.dataframe(display_df, height=400)
        
        # Plot 1: Bar chart of model-predicted win probabilities for each horse
        st.markdown("**Win Probabilities**")
        prob_chart_data = display_df[['Horse', 'Predicted Win %']].copy()
        fig_bar = px.bar(prob_chart_data, x='Horse', y='Predicted Win %', text='Predicted Win %')
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_yaxes(range=[0, 100], title="Win Probability (%)")
        fig_bar.update_layout(xaxis_title=None, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Plot 2: Odds vs Predicted Probability scatter plot (if starting odds available)
        if 'Sp' in race_df.columns:
            # Compute implied win probabilities from starting odds (SP)
            odds = race_df['Sp'].astype(float, errors='ignore')
            implied_probs = []
            for sp in odds:
                if pd.notna(sp):
                    try:
                        implied = 1.0 / (float(sp) + 1.0)
                        implied_probs.append(round(implied * 100, 1))
                    except:
                        implied_probs.append(np.nan)
                else:
                    implied_probs.append(np.nan)
            scatter_data = pd.DataFrame({
                'Horse': race_df['HorseName'],
                'Predicted': race_df['PredictedWinProb'], 
                'Implied': implied_probs
            })
            fig_scatter = px.scatter(scatter_data, x='Predicted', y='Implied', text='Horse')
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_xaxes(title="Predicted Win Probability (%)", range=[0, 100])
            fig_scatter.update_yaxes(title="Implied Win Probability (%)", range=[0, 100])
            # Add a reference diagonal (y = x) for perfect agreement
            fig_scatter.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
                                   line=dict(color='gray', dash='dash'))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Plot 3: Horse form chart (SpeedRating trends for a selected horse)
        st.markdown("**Horse Form (Speed Rating Trend)**")
        selected_horse = st.selectbox("Select Horse to view form chart", horses)
        if selected_horse:
            hist = df[df['HorseName'] == selected_horse].copy()
            hist.sort_values(['RaceDate', 'RaceTime'], inplace=True)
            # Exclude current race for form (if present in history)
            if selected_race_key:
                if 'RaceTimeOnly' in hist.columns:
                    hist = hist[~((hist['Course'] == selected_race_key[0]) & 
                                  (hist['RaceDate'] == selected_race_key[1]) & 
                                  (hist['RaceTimeOnly'] == selected_race_key[2]))]
                else:
                    hist = hist[~((hist['Course'] == selected_race_key[0]) & 
                                  (hist['RaceDate'] == selected_race_key[1]) & 
                                  (hist['RaceTime'] == selected_race_key[2]))]
            if not hist.empty:
                fig_line = px.line(hist, x='RaceDate', y='SpeedRating', title=f"{selected_horse} - Recent Speed Ratings")
                fig_line.update_traces(mode='lines+markers')
                fig_line.update_layout(xaxis_title="Race Date", yaxis_title="Speed Rating")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.write("No past performance data available for this horse.")
        
        # Plot 4: Sectional time simulation (simplified pace line plot for all horses in the race)
        st.markdown("**Sectional Time Simulation**")
        if 'Yards' in race_df.columns:
            total_dist = race_df['Yards'].iloc[0]  # distance is same for all in race
        else:
            total_dist = None
        lines_data = []
        for _, row in race_df.iterrows():
            horse = row['HorseName']
            finish_time = row['ActualSeconds']
            if pd.notna(finish_time) and total_dist:
                # Simulate pace: determine if horse started fast or slow (alternate pattern for demo)
                fast_start = (_ % 2 == 0)
                # Fraction of total time spent in first half of race
                f = 0.47 if fast_start else 0.53
                t_half = f * finish_time
                d_half = 0.5 * total_dist
                # Add points for the line: start, halfway, finish
                lines_data.append({'Horse': horse, 'Time': 0, 'Distance': 0})
                lines_data.append({'Horse': horse, 'Time': t_half, 'Distance': d_half})
                lines_data.append({'Horse': horse, 'Time': finish_time, 'Distance': total_dist})
        if lines_data:
            lines_df = pd.DataFrame(lines_data)
            fig_lines = px.line(lines_df, x='Time', y='Distance', color='Horse', 
                                 title="Estimated Distance Covered Over Time")
            fig_lines.update_layout(xaxis_title="Time (seconds)", yaxis_title="Distance (yards)")
            st.plotly_chart(fig_lines, use_container_width=True)
        
        # Plot 5: Trainer and Jockey win% comparison for this race
        st.markdown("**Trainer & Jockey Win%**")
        if trainer_stats is not None:
            trainers_in_race = race_df['Trainer'].unique()
            comp_df = trainer_stats.loc[trainers_in_race][['WinPct']].reset_index()
            comp_df = comp_df.sort_values('WinPct', ascending=False)
            fig_tr = px.bar(comp_df, x='Trainer', y='WinPct', title="Trainer Overall Win%")
            fig_tr.update_yaxes(range=[0, min(100, comp_df['WinPct'].max() + 5)])
            fig_tr.update_layout(xaxis_title=None, yaxis_title="Win Percentage (%)")
            st.plotly_chart(fig_tr, use_container_width=True)
        if jockey_stats is not None:
            jockeys_in_race = race_df['Jockey'].unique()
            comp_jdf = jockey_stats.loc[jockeys_in_race][['WinPct']].reset_index()
            comp_jdf = comp_jdf.sort_values('WinPct', ascending=False)
            fig_jk = px.bar(comp_jdf, x='Jockey', y='WinPct', title="Jockey Overall Win%")
            fig_jk.update_yaxes(range=[0, min(100, comp_jdf['WinPct'].max() + 5)])
            fig_jk.update_layout(xaxis_title=None, yaxis_title="Win Percentage (%)")
            st.plotly_chart(fig_jk, use_container_width=True)
        
        # Download button for CSV output of this race's data and predictions
        out_cols = ['HorseName', 'Course', 'RaceDate', 'Race', 'Trainer', 'Jockey', 
                    'PaceScore', 'SuperAbilityScore', 'TrainerWinPct', 'JockeyWinPct', 'PredictedWinProb']
        if 'Sp' in race_df.columns:
            out_cols.append('Sp')
        output_df = race_df[out_cols].copy()
        output_df.rename(columns={'HorseName': 'Horse', 'PredictedWinProb': 'PredictedWin%'}, inplace=True)
        csv_data = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Race Data (CSV)", data=csv_data, file_name="race_predictions.csv")
    else:
        st.write("**Please select a race from the sidebar to view the racecard and predictions.**")
