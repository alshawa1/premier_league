import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import streamlit as st

# --- Load Data ---
import pandas as pd
df = pd.read_csv("https://github.com/alshawa1/premier_league/blob/main/my_app/data/final_leg_data.csv")
# Filter players by position and minutes
df_forwards = df[(df['player_position'] == 'Forward') & (df['Minutes Played'] > 100)]
df_midfielders = df[(df['player_position'] == 'Midfielder') & (df['Minutes Played'] > 100)]
df_defenders = df[(df['player_position'] == 'Defender') & (df['Minutes Played'] > 100)]
df_goalkeepers = df[df['player_position'] == 'Goalkeeper']


# --- Targets لكل مركز ---
targets_by_position = {
    'Forward': ['Goals', 'Yellow Cards', 'Red Cards', 'Penalties Taken'],
    'Midfielder': ['Goals', 'Assists', 'Yellow Cards', 'Red Cards'],
    'Defender': ['Goals', 'Yellow Cards', 'Red Cards', 'Own Goals'],
    'Goalkeeper': ['Clean Sheets', 'Goals Conceded', 'Yellow Cards', 'Red Cards']
}

# --- Features مشتركة لكل المراكز ---
features = ['Age', 'Minutes Played', 'XG', 'Shots On Target Inside the Box',
            'dribble_accuracy', 'cross_accuracy', 'Touches in the Opposition Box']

# --- Streamlit App ---
st.title("⚽ Player Performance Predictor")

# اختيار المركز
position = st.selectbox("اختر المركز:", list(targets_by_position.keys()), key="position_select")

# اختيار DataFrame حسب المركز
if position == "Forward":
    df_selected = df_forwards
elif position == "Midfielder":
    df_selected = df_midfielders
elif position == "Defender":
    df_selected = df_defenders
else:
    df_selected = df_goalkeepers

# اختيار اللاعب
player_name = st.selectbox("اختر اللاعب:", df_selected['player_name'].unique(), key="player_select")

# Features & Targets
X = df_selected[features].fillna(0)
y = df_selected[targets_by_position[position]].fillna(0)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# زر التوقع
if st.button("🔮 Predict"):
    player_data = df_selected[df_selected['player_name'] == player_name][features]
    prediction = model.predict(player_data)[0]

    st.success(f"✅ Predictions for {player_name}:")
    for target, value in zip(targets_by_position[position], prediction):
        st.write(f"- {target}: {value:.2f}")
