#  WinMeter: Win Probability Estimation Tool for T20 Cricket

**WinMeter** is a machine learning-based tool that predicts the win probability of the batting team in IPL T20 matches using real-time, delivery-level match data.

The model considers key features that influence match outcomes, including `runs_left`, `balls_left`, `wickets`, `current run rate (crr)`, and `required run rate (rrr)`. Categorical variables like `batting_team`, `bowling_team`, and `city` were encoded during preprocessing.

Extensive exploratory data analysis was performed using **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn** to uncover patterns and validate assumptions. Feature engineering played a crucial role in building meaningful predictors.

Two machine learning models were trained using **Scikit-learn**:
- **Logistic Regression**: achieving ~81% accuracy.
- **Linear Discriminant Analysis (LDA)**: achieving ~83% accuracy.

The final tool can be used to estimate win probabilities dynamically throughout a match. This project demonstrates how statistical modeling can be applied to sports analytics with interpretable and accurate outcomes.

->  Dataset: IPL match data with delivery-level features
->  Output: Real-time win probability for the batting side
