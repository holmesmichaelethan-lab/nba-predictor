import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import numpy as np
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.metric_cards import style_metric_cards

# Set Streamlit page configuration
st.set_page_config(page_title="NBA Player Performance Prediction", layout="centered")

# Customizing UI Theme
st.markdown(
    """
    <style>
        body { font-family: 'Arial', sans-serif; background-color: white; text-align: center; }
        .stButton>button { background-color: #F5832A; color: white; font-weight: bold; }
        .stButton>button:hover { background-color: #29739E; color: white; }
        .stSelectbox>div>div { border-radius: 10px; }
        .stDataFrame { border-radius: 10px; margin: 0 auto; }
        .stMarkdown { font-size: 16px; }
        .stTitle { color: #073966; }
        .stSubheader { color: #29739E; }
        .stHeader { color: #073966; }
        .stImage { display: block; margin-left: auto; margin-right: auto; width: 50%; }
        .stTitle, .stSubheader, .stHeader { text-align: center; }
        .stPlotlyChart { display: block; margin-left: auto; margin-right: auto; }
    </style>
    """,
    unsafe_allow_html=True
)

# Placeholder for the logo
st.image("predictivestats.png", width=800)

# --- Load dataset using Parquet with caching ---
@st.cache_data
def load_data():
    # Read Parquet file
    df = pd.read_parquet("nba_active_players_game_logs_2021_24.parquet")

    # Convert GAME_DATE to datetime and sort
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by=['PLAYER_NAME', 'GAME_DATE'])

    # Defensive ratings for all 30 teams
    defensive_ratings = {
        "Oklahoma City Thunder": 107.0, "Orlando Magic": 109.6, "Houston Rockets": 110.1,
        "LA Clippers": 110.4, "Memphis Grizzlies": 110.7, "Boston Celtics": 111.2,
        "Minnesota Timberwolves": 111.8, "Milwaukee Bucks": 112.0, "Cleveland Cavaliers": 112.1,
        "Golden State Warriors": 112.4, "Miami Heat": 112.8, "Dallas Mavericks": 112.9,
        "Sacramento Kings": 113.0, "Detroit Pistons": 113.1, "Charlotte Hornets": 113.2,
        "New York Knicks": 114.4, "Denver Nuggets": 114.5, "Atlanta Hawks": 114.6,
        "Indiana Pacers": 114.7, "San Antonio Spurs": 114.8, "Philadelphia 76ers": 115.0,
        "LA Lakers": 115.1, "Chicago Bulls": 115.2, "Brooklyn Nets": 115.3,
        "Phoenix Suns": 115.4, "Portland Trail Blazers": 115.5, "Toronto Raptors": 115.6,
        "New Orleans Pelicans": 115.7, "Utah Jazz": 115.9, "Washington Wizards": 116.2
    }

    df = df.merge(pd.DataFrame(list(defensive_ratings.items()), columns=["TEAM", "DEF_RATING"]),
                  on="TEAM", how="left")

    # Encode categorical columns
    label_encoders = {}
    for col in ["MATCHUP", "TEAM", "PLAYER_NAME"]:
        le = LabelEncoder()
        df[f"{col}_original"] = df[col]  # keep original
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Compute rolling averages
    rolling_window = 5
    for stat in ['PTS', 'REB', 'AST']:
        df[f'{stat}_L5G'] = df.groupby('PLAYER_NAME')[stat].transform(
            lambda x: x.rolling(rolling_window, min_periods=1).mean().round(1)
        )

    return df, label_encoders

df, label_encoders = load_data()

# --- Features and targets ---
features = ["MATCHUP", "TEAM", "PLAYER_NAME", "FGM", "FGA", "FTM", "FTA",
            "OREB", "DREB", "TOV", "PTS_L5G", "REB_L5G", "AST_L5G", "DEF_RATING"]
targets = ["PTS", "REB", "AST", "FG3M", "BLK", "STL"]

features = [col for col in features if col in df.columns]
df = df.dropna(subset=targets)
X = df[features]
y = df[targets]

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train models with caching ---
@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    return rf, xgb_model

rf_model, xgb_model = train_models(X_train, y_train)

# --- Streamlit UI ---
st.title("NBA Player Performance Prediction")
st.subheader("Predict player performance for upcoming matchups.")
st.divider()

col1, col2 = st.columns(2)
players = sorted(df["PLAYER_NAME_original"].unique())
teams = sorted(df["TEAM_original"].unique())

with col1:
    selected_player = st.selectbox("Select Player", players)
with col2:
    selected_team = st.selectbox("Select Opponent Team", teams)

# --- Prediction logic ---
def predict_player_performance(player_name, opponent_team):
    def encode_value(encoder, value):
        return encoder.transform([value])[0] if value in encoder.classes_ else -1

    player_encoded = encode_value(label_encoders["PLAYER_NAME"], player_name)
    team_encoded = encode_value(label_encoders["TEAM"], opponent_team)

    matchup_games = df[(df["PLAYER_NAME_original"] == player_name) &
                       df["MATCHUP_original"].str.contains(opponent_team)]

    if matchup_games.empty:
        return pd.DataFrame({"Stat": targets,
                             "Prediction_Avg": [0]*len(targets),
                             "Chance_Avg (%)": [0]*len(targets)})

    recent_games = matchup_games.iloc[-1]
    sample_input = pd.DataFrame([[recent_games[col] if col in recent_games else 0 for col in X_train.columns]],
                                columns=X_train.columns)

    rf_preds = rf_model.predict(sample_input)
    xgb_preds = xgb_model.predict(sample_input)
    avg_pred = np.round((rf_preds[0] + xgb_preds[0]) / 2, 1)

    predicted_df = pd.DataFrame({
        "Stat": targets,
        "Prediction_Avg": avg_pred,
        "Chance_Avg (%)": [100]*len(targets)  # simplified confidence
    })
    return predicted_df

def get_player_averages(player_name):
    player_games = df[df["PLAYER_NAME_original"] == player_name].tail(5)
    avg_df = pd.DataFrame({
        "Stat": targets,
        "Average": [round(player_games[t].mean(), 1) for t in targets]
    })
    return avg_df

def plot_comparative_line_chart(average_df, predicted_df):
    comparison_df = pd.merge(average_df, predicted_df, left_on='Stat', right_on='Stat', how='inner')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison_df['Stat'], y=comparison_df['Average'],
                             mode='lines+markers', name='Player Average', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=comparison_df['Stat'], y=comparison_df['Prediction_Avg'],
                             mode='lines+markers', name='Prediction', line=dict(color='red', width=2)))
    fig.update_layout(title="Comparative Stats: Player Average vs. Predicted Performance",
                      xaxis_title="Stat", yaxis_title="Value", showlegend=True)
    st.plotly_chart(fig)

# --- Run predictions on button click ---
if st.button('Predict Performance'):
    predicted_df = predict_player_performance(selected_player, selected_team)
    average_df = get_player_averages(selected_player)

    st.subheader("Player Prediction Table")
    st.dataframe(predicted_df)

    plot_comparative_line_chart(average_df, predicted_df)
