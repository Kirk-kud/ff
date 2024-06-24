
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown

# Load the model and scaler

url = 'https://drive.google.com/uc?export=download&id=1ugPa26CQtAlO7BiEGk_kejkS-n7hRYBm'
u = 'https://drive.google.com/file/d/1ugPa26CQtAlO7BiEGk_kejkS-n7hRYBm/view?usp=drive_link'
filename = 'model.joblib'
gdown.cached_download(url, filename)

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')  # assuming the scaler was saved separately

fe = ['potential', 'value_eur', 'wage_eur', 'age', 'international_reputation', 'shooting', 'passing',
 'dribbling', 'physic', 'attacking_short_passing', 'skill_curve', 'skill_long_passing', 'skill_ball_control', 'movement_reactions',
'power_shot_power', 'power_long_shots', 'mentality_vision', 'mentality_composure']

ft = ['Potential', 'Player value (in Euros)', 'Player wages (in Euros)', 'Player Age', 'International Reputation (1-5)', 'Shooting', 'Passing',
 'Dribbling', 'Physic', 'Attacking short passing', 'Skill curve', 'Skill long passing', 'Skill ball control', 'Movement reactions',
'Power (shot_power)', 'Power (long_shots)', 'Mentality vision', 'Mentality composure']

# Streamlit app
st.image('fifa_logo.png',use_column_width=True)

st.title("FIFA™ Rating Predictor")

st.write("Enter the values for the footballer's 18 features:")

st.write("Most of the attributes range between 1 and 100")

st.write()
st.write("All the best with the predictor")

# Create input fields for each feature
features = {}
for i in range(18):
    feature_name = fe[i]
    if feature_name == 'international_reputation':
        features[feature_name] = st.slider(ft[i], value=1.0, step=1.0, min_value = 1.0, max_value = 5.0)
    elif feature_name == 'value_eur':
        features[feature_name] = st.slider(ft[i], value=0.0, step=100.0, min_value = 0.0, max_value = 194000000.0)
        st.write("NOTE: Player values range from about €70000 to €194,000,000")
    elif feature_name == 'wage_eur':
        features[feature_name] = st.slider(ft[i], value=0.0, step=100.0, min_value = 0.0, max_value = 900000.0)
        st.write("NOTE: Wages range from about €500 to €350000")
    elif feature_name == 'age':
        features[feature_name] = st.slider(ft[i], value=14.0, step=1.0, min_value = 14.0, max_value = 60.0)
    else:
        features[feature_name] = st.slider(ft[i], value=0.0, step=1.0, min_value = 0.0, max_value = 100.0)
    
    

# Convert input data to DataFrame
input_df = pd.DataFrame([features])

# Display input data
st.write('Input Data:', input_df)

# Scale the input data using the same scaler used during training
scaled_input_df = scaler.transform(input_df)

# Prediction
if st.button('Predict'):
    prediction_scaled = model.predict(scaled_input_df)
    conf_score = 96
    
    prediction = round(int(prediction_scaled[0]))
    st.write(f"Prediction: Your player's predicted rating is {prediction} overall")
    st.write(f"The confidence score for the prediction is {conf_score}%")
