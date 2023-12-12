import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib 
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.models import Model

# Function to create the GRU model using the functional API
def create_gru_model(window_size=12, units=50):
    input_layer = Input(shape=(window_size, X_train_seq_reshaped.shape[2]))
    gru_layer = GRU(units, activation='relu', return_sequences=True)(input_layer)
    output_layer = Dense(1)(gru_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequence = X[i:i + sequence_length]
        sequences.append(sequence)
    sequences = np.array(sequences)
    return sequences





# Load GRU model
gru_model_path = "C:\\Users\\Angela Benning\\Desktop\\CryptoPrediction\\best_gru_model.pkl"
gru_model = joblib.load(gru_model_path)

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Cryptocurrency Prediction App",
        page_icon=":money_with_wings:",
        layout="centered",
    )
    
    st.markdown(
        """
        <style>
            body {
                font-family: 'Roboto', sans-serif;
                background: url('https://www.katznerlawgroup.com/wp-content/uploads/2023/11/shutterstock_732138853.jpg') no-repeat center center fixed;
                background-size: cover;
                color: #d35400; /* Set text color to white */
            }
            .stApp {
                background: none;
            }
            .stContent {
                background: none;
                padding: 20px;
                height: 100vh;
                width: 100vw;
                position: fixed;
                z-index: -1;
            }
            .stButton>button {
                background-color: #ffffff!important; /* Orange color for buttons */
                color: #ffffff !important; /* Set text color to white */
            }
            .stButton>button:hover {
                background-color: #d35400 !important; /* Slightly darker shade on hover */
            }
            .stSelectbox {
                background-color:#ffffff !important; /* Orange color for select boxes */
                color: #ffffff !important; /* Set text color to white */
            }
            .stSlider>div>div>div {
                background-color: #ffffff !important; /* white color for sliders */
            }
            .stSlider>div>div>div>div {
                background-color: #d35400 !important; /* white color for slider handles */
            }
            .stTitle {
                color: rgba !important; /* Light grey color for titles */
            }
            .st-RgMPj {
                background-color: rgba(255, 255, 255, 0.7) !important; /* Slightly transparent white for the table */
            }
            .st-RgMPj tr th {
                background-color: rgba(255, 255, 255, 0.7) !important; /* Slightly transparent white for the table header */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    sequence_length = 10  # Define sequence length

    # Cryptocurrency selection
    crypto_options = {
        'XRP': "C:\\Users\\Angela Benning\\Downloads\\coin_XRP.csv",
        'USDCoin': "C:\\Users\\Angela Benning\\Downloads\\coin_USDCoin.csv",
        'Tether': "C:\\Users\\Angela Benning\\Downloads\\coin_Tether.csv",
        'Solana': "C:\\Users\\Angela Benning\\Downloads\\coin_Solana.csv",
        'Ethereum': "C:\\Users\\Angela Benning\\Downloads\\coin_Ethereum.csv",
        'Bitcoin': "C:\\Users\\Angela Benning\\Downloads\\coin_Bitcoin.csv",
    }

    # Streamlit layout
    st.title("🚀 Cryptocurrency Price Prediction App")
    st.subheader("Personalized Predictions and Visualizations")
    
    # User input section
    user_name = st.text_input("Enter Your Name:")
    prediction_period = st.date_input("Select Prediction Period:")

    # Load selected dataset
    selected_crypto = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    data = pd.read_csv(crypto_options[selected_crypto])

    # Feature selection
    selected_features = [
        'High', 'Low', 'Open', 'Marketcap', 'RSI', 'SMA_10', 'SNo', 'Volume', 'SMA_50', 'Symbol', 'Name', 'Year'
    ]

    # Input values for selected features
    input_values = {}
    for feature in selected_features:
        if feature in ['Name', 'Symbol']:
            input_values[feature] = st.text_input(f"Enter value for {feature}")
        else:
            input_values[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
    
    prediction = None  # Initialize prediction variable
    

    if st.button("Make Prediction"):
        # Create a DataFrame from input values
        input_data = pd.DataFrame([input_values])

        # Preprocess input data
        X_numeric = input_data.drop(['Name', 'Symbol'], axis=1)
        X_encoded = pd.get_dummies(input_data[['Name', 'Symbol']], drop_first=True)
    
        X_combined = pd.concat([X_numeric, X_encoded], axis=1)

        # Scale numeric features
        scaler = StandardScaler()
        X_scaled_numeric = scaler.fit_transform(X_numeric)
        X_scaled_numeric = pd.DataFrame(X_scaled_numeric, columns=X_numeric.columns)

        # Combine scaled numeric features with encoded features
        X_scaled = pd.concat([X_scaled_numeric, X_encoded], axis=1)

        # Create sequences
        X_seq = create_sequences(X_scaled, sequence_length)
        
        # Before making predictions
        print("X_scaled shape:", X_scaled.shape)
        print("X_seq shape:", X_seq.shape)

        if X_seq.size == 0:
            st.warning("Not enough data for prediction.")
        else:
            # Make prediction using the loaded GRU model
            prediction = gru_model.predict(X_seq)[0]  # Assuming model returns a single value

        st.subheader("Predicted Price")
        if prediction is not None:
            st.write(f"The predicted price is: {prediction:.2f}")

    # Additional visualizations
    st.subheader("Additional Visualizations")

    # Line chart of closing prices
    st.subheader("Line chart of closing prices")
    st.line_chart(data['Close'])
    
    

    # Statistics summary
    st.subheader("Statistics Summary")
    st.table(data.describe())

if __name__ == "__main__":
    main()
