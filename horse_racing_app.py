import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import plotly.express as px

st.set_page_config(page_title="Horse Racing Custom Racecard", page_icon="🐎", layout="wide")
st.title("🏇 Custom Horse Racing Racecard & Predictor")
st.write("Build your own racecard, compare horses from any date/track, and view ultra-detailed stats and win predictions.")

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("racing_data_upload.csv.gz")
    df['RaceDate'] = pd.to_datetime(df['RaceDate'], errors='coerce')
    if 'RaceTime' in df.columns:
        df['RaceTime'] = pd.to_datetime(df['RaceTime'], errors='coerce')
    df['Yards'] = pd.to_numeric(df['Yards'], errors='coerce')
    df['Seconds'] = pd.to_numeric(df['Seconds'], errors='coerce')
    df['OR'] = pd.to_numeric(df['OR'], errors='coerce')
    df['FPos_num'] = pd.to_numeric(df['FPos'], errors='coerce')
    df['HorseName'] = df['HorseName'].astype(str)
    df['Trainer'] = df['Trainer'].astype(str)
    df['Jockey'] = df['Jockey'].astype(str)
    df = df.dropna(subset=['Yards', 'Seconds', 'HorseName', 'Trainer', 'Jockey', 'RaceDate'])
    df['SpeedRating'] = df['Yards'] / df['Seconds']
    df['Win'] = (df['FPos_num'] == 1).astype(int)
    return df

def compute_form_stats(hist):
    last5 = hist.tail(5)['SpeedRating'].tolist()
    trend = "Stable"
    if len(last5) >= 2:
        slope = last5[-1] - last5[0]
        if slope > 0.5:
            trend = "Improving"
        elif slope < -0.5:
            trend = "Declining"
    career_runs = hist.shape[0]
    wins = (hist['Win'] == 1).sum()
    places = (hist['FPos_num'] <= 3).sum()
    win_pct = 100 * wins / career_runs if career_runs else 0
    place_pct = 100 * places / career_runs if career_runs else 0
    mean_speed = hist['SpeedRating'].mean()
    consistency = hist['SpeedRating'].std() if hist['SpeedRating'].count() > 1 else 0
    best_speed = hist['SpeedRating'].max()
    last_speed = last5[-1] if last5 else np.nan
    return {
        "Last5": last5,
        "Trend": trend,
        "Runs": career_runs,
        "Wins": wins,
        "Places": places,
        "Win%": win_pct,
        "Place%": place_pct,
        "AvgSpeed": mean_speed,
        "Consistency": consistency,
        "BestSpeed": best_speed,
        "LastSpeed": last_speed
    }

def preferred_track(df, entity, entity_col):
    group = df[df[entity_col] == entity]
    by_track = group.groupby('Course').agg(
        Runs=('Win', 'count'),
        Wins=('Win', 'sum')
    )
    by_track = by_track[by_track['Runs'] >= 3]
    if by_track.empty:
        return ("-", 0, 0.0)
    by_track['Win%'] = by_track['Wins'] / by_track['Runs'] * 100
    best = by_track.sort_values('Win%', ascending=False).iloc[0]
    return (best.name, int(best['Runs']), round(best['Win%'], 1))

def in_form_stats(df, entity, entity_col, last_n_days=14):
    recent = df[(df[entity_col] == entity) & (df['RaceDate'] >= (df['RaceDate'].max() - timedelta(days=last_n_days)))]
    runs = recent.shape[0]
    wins = (recent['Win'] == 1).sum()
    win_pct = 100 * wins / runs if runs else 0.0
    return (runs, wins, round(win_pct,1))

@st.cache_data(show_spinner=False)
def train_model(df):
    features = ['SpeedRating', 'OR', 'TrainerWinPct', 'JockeyWinPct']
    model_df = df.dropna(subset=features + ['Win']).copy()
    for col in features:
        model_df = model_df[np.isfinite(model_df[col])]
    model_df = model_df[model_df['Win'].isin([0,1])]
    if model_df.empty or model_df['Win'].nunique() < 2:
        model = DummyClassifier(strategy='uniform')
        model.fit([[0]*len(features), [1]*len(features)], [0,1])
        return model
    X = model_df[features].astype(float)
    y = model_df['Win'].astype(int)
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
    except Exception as e:
        st.warning(f"⚠️ Fallback to dummy model due to: {e}")
        model = DummyClassifier(strategy='uniform')
        model.fit([[0]*len(features), [1]*len(features)], [0,1])
    return model

df = load_data()
df['TrainerWinPct'] = df['Trainer'].map(df.groupby('Trainer')['Win'].mean() * 100).fillna(0)
df['JockeyWinPct'] = df['Jockey'].map(df.groupby('Jockey')['Win'].mean() * 100).fillna(0)

horse_choices = sorted(df['HorseName'].unique())
selected_horses = st.sidebar.multiselect(
    "Select horses for your custom racecard (any date, any course):", horse_choices, default=horse_choices[:8]
)

if not selected_horses:
    st.info("Please select at least one horse.")
    st.stop()

custom_race_df = df[df['HorseName'].isin(selected_horses)].copy()
latest_rows = custom_race_df.groupby('HorseName').apply(lambda g: g.sort_values('RaceDate').iloc[-1]).reset_index(drop=True)

stats_table = []
for _, row in latest_rows.iterrows():
    horse = row['HorseName']
    hist = custom_race_df[custom_race_df['HorseName'] == horse].sort_values('RaceDate')
    trainer = row['Trainer']
    jockey = row['Jockey']
    course = row['Course']
    form = compute_form_stats(hist)
    t_pref_track, t_runs, t_winpct = preferred_track(df, trainer, 'Trainer')
    t_inform_runs, t_inform_wins, t_inform_pct = in_form_stats(df, trainer, 'Trainer')
    j_pref_track, j_runs, j_winpct = preferred_track(df, jockey, 'Jockey')
    j_inform_runs, j_inform_wins, j_inform_pct = in_form_stats(df, jockey, 'Jockey')
    h_pref_track, h_runs, h_winpct = preferred_track(df, horse, 'HorseName')

    stats_table.append({
        "Horse": horse,
        "Trainer": trainer,
        "Jockey": jockey,
        "Course": course,
        "OR": row['OR'],
        "LastSpeed": form['LastSpeed'],
        "BestSpeed": form['BestSpeed'],
        "AvgSpeed": form['AvgSpeed'],
        "FormTrend": form['Trend'],
        "Consistency": form['Consistency'],
        "Runs": form['Runs'],
        "Wins": form['Wins'],
        "Places": form['Places'],
        "Win%": form['Win%'],
        "Place%": form['Place%'],
        "TrainerWin%": row['TrainerWinPct'],
        "JockeyWin%": row['JockeyWinPct'],
        "T_PrefTrack": t_pref_track,
        "T_Runs@Track": t_runs,
        "T_Win%@Track": t_winpct,
        "T_14dRuns": t_inform_runs,
        "T_14dWins": t_inform_wins,
        "T_14dWin%": t_inform_pct,
        "J_PrefTrack": j_pref_track,
        "J_Runs@Track": j_runs,
        "J_Win%@Track": j_winpct,
        "J_14dRuns": j_inform_runs,
        "J_14dWins": j_inform_wins,
        "J_14dWin%": j_inform_pct,
        "H_PrefTrack": h_pref_track,
        "H_Runs@Track": h_runs,
        "H_Win%@Track": h_winpct,
        "Last5SpeedRatings": form['Last5']
    })

stats_df = pd.DataFrame(stats_table)

model = train_model(df)
features = ['SpeedRating', 'OR', 'TrainerWinPct', 'JockeyWinPct']
X = latest_rows[features].fillna(0).astype(float)
try:
    win_probs = model.predict_proba(X)[:, 1]
except:
    win_probs = np.full(X.shape[0], 1/X.shape[0])
stats_df['PredictedWin%'] = (win_probs * 100).round(2)

if not stats_df.empty:
    st.header("🏆 Custom Racecard Results & Stats")
    def highlight_leader(col, higher_better=True):
        vals = stats_df[col]
        if vals.nunique() == 1:
            return ["" for _ in vals]
        leader = vals.idxmax() if higher_better else vals.idxmin()
        return ["🟢" if i == leader else "" for i in range(len(vals))]
    stats_df["🔥FastestLast"] = highlight_leader("LastSpeed")
    stats_df["⭐MostConsistent"] = highlight_leader("Consistency", higher_better=False)
    stats_df["🏁BestAbility"] = highlight_leader("BestSpeed")
    stats_df["💪TrainerInForm"] = highlight_leader("T_14dWin%")
    stats_df["🏇JockeyInForm"] = highlight_leader("J_14dWin%")
    display_cols = [
        "Horse","Trainer","Jockey","Course","OR","LastSpeed","🔥FastestLast",
        "BestSpeed","🏁BestAbility","AvgSpeed","FormTrend","Consistency","⭐MostConsistent",
        "Runs","Wins","Places","Win%","Place%",
        "TrainerWin%","T_14dWin%","💪TrainerInForm",
        "JockeyWin%","J_14dWin%","🏇JockeyInForm",
        "T_PrefTrack","T_Runs@Track","T_Win%@Track",
        "J_PrefTrack","J_Runs@Track","J_Win%@Track",
        "H_PrefTrack","H_Runs@Track","H_Win%@Track",
        "PredictedWin%"
    ]
    st.dataframe(stats_df[display_cols].set_index("Horse"), use_container_width=True)

    st.subheader("Predicted Win Probability")
    chart = px.bar(stats_df, x="Horse", y="PredictedWin%", color="PredictedWin%", text="PredictedWin%")
    chart.update_traces(textposition='outside')
    chart.update_layout(yaxis_title="Win %", xaxis_title="Horse")
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Recent SpeedRating Trends (last 5 runs)")
    for i, row in stats_df.iterrows():
        st.markdown(f"**{row['Horse']}**: " +
            " → ".join([f"{x:.2f}" for x in row['Last5SpeedRatings']]))
        fig = px.line(
            x=list(range(1, len(row['Last5SpeedRatings'])+1)),
            y=row['Last5SpeedRatings'],
            markers=True,
            title=f"SpeedRating Trend: {row['Horse']}"
        )
        fig.update_layout(showlegend=False, xaxis_title="Run #", yaxis_title="SpeedRating")
        st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Download Custom Racecard Stats", stats_df.to_csv(index=False), file_name="custom_racecard_stats.csv")
else:
    st.warning("No runners found for this selection.")
