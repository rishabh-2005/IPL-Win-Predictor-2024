import pickle
import streamlit as st
import pandas as pd

# Load the model pipeline (with the fitted ColumnTransformer inside)
with open('model1.pkl', 'rb') as f:
    pipe = pickle.load(f)

st.title('IPL Win Predictor')

# Teams list
teams = sorted(['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bengaluru',
                'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
                'Rajasthan Royals', 'Delhi Capitals','Gujarat Titans','Rising Pune Supergiants','Lucknow Super Giants'])

# Team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

# Cities list
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 
          'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 
          'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 
          'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# City selection
selected_city = st.selectbox('Select the city', sorted(cities))

# Target input
target = st.number_input('Target', min_value=0)

# Score, Wickets, and Overs input
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    wickets = st.number_input('Wickets', min_value=0, max_value=9)
with col5:
    overs = st.number_input('Overs completed', min_value=0, max_value=20)

# Button for prediction
if st.button('Predict Probability'):
    
    # Calculating the required variables
    runs_left = target - score
    balls_left = 120 - overs * 6
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    # Create a DataFrame
    df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Prediction using the fitted pipeline
    
    prediction = pipe.predict_proba(df)
    r_1 = round(prediction[0][0] * 100)
    r_2 = round(prediction[0][1] * 100)

    # Display probabilities
    st.header('Winning Probability')
    st.header(f"{batting_team}: {r_2} %")
    st.header(f"{bowling_team}: {r_1} %")
    
    
