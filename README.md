# Cryptocurrency Price Prediction using Machine Learning

## LSTM and GRU Networks for Price Forecasting

### Summary

This research focuses on predicting cryptocurrency prices using both Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks. The dataset comprises historical price and market data, and the goal is to build robust models that accurately forecast future cryptocurrency prices, comparing the performance of LSTM and GRU architectures.

### Directory and File Structure

- **Jupyter Notebook:** Contains the primary code for data preprocessing, LSTM and GRU model development, and price prediction (`Crypto_Price_Prediction_LSTM_GRU.ipynb`).
- **Dataset:** A CSV file comprising historical price and market data.

    ```python
    data1 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GROUP 15 _ FINAL PROJECT/coin_XRP.csv')
    data2 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GROUP 15 _ FINAL PROJECT/coin_USDCoin.csv')
    data3 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GROUP 15 _ FINAL PROJECT/coin_Tether.csv')
    data4 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GROUP 15 _ FINAL PROJECT/coin_Solana.csv')
    data5 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GROUP 15 _ FINAL PROJECT/coin_Ethereum.csv')
    data6 = pd.read_csv('/content/drive/My Drive/Colab Notebooks/GROUP 15 _ FINAL PROJECT/coin_Bitcoin.csv')
    
    # Combine the datasets into one DataFrame
    data = pd.concat([data1, data2, data3, data4, data5, data6], ignore_index=True)
    
    # Display the combined DataFrame
    data
    ```
  
- **Model Files:**
  - `crypto_price_model_gru.pkl`: A pickle file storing the best model, GRU model.
- **README.md:** A readme file providing an overview of the project.

### Code Synopsis

1. **Data Loading and Exploration:** Load historical cryptocurrency price and market data, and conduct initial exploration.

2. **Data Preprocessing:** Clean and transform the data to prepare it for LSTM and GRU model training.
   - Normalize and scale the data for consistency.
   - Create sequences of historical prices as input features and the next day's price as the target variable.

3. **Data Splitting:** Split the dataset into training and testing sets.

4. **LSTM Model:** Develop a Long Short-Term Memory (LSTM) neural network using Keras to capture temporal dependencies.
   - Configure the architecture and hyperparameters of the LSTM model.

5. **GRU Model:** Develop a Gated Recurrent Unit (GRU) neural network using Keras.
   - Configure the architecture and hyperparameters of the GRU model.

6. **Model Training:** Train both the LSTM and GRU models on the training set.

7. **Model Evaluation:** Evaluate the LSTM and GRU models on the test set and visualize the predicted prices against the actual prices.
   - Compare performance metrics to determine the effectiveness of each model.

8. **Best Model Selection:** Select the best-performing model (LSTM or GRU) based on relevant metrics.

9. **Model Saving:** Save the trained LSTM and GRU models as HDF5 files for future use.

### Streamlit Web Application

- The project includes a Streamlit web application for interactive model usage, which you would find in our repository.

### Usage

1. Open and execute the `Crypto_Price_Prediction_LSTM_GRU.ipynb` notebook in a Jupyter environment.
2. Ensure the necessary libraries are installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`).
3. Adjust paths and filenames if necessary.
4. Follow the code comments and run the cells in the correct order.

### Prerequisites

- Jupyter Notebook - Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`

### Best Model Selection

- Based on rigorous evaluation, select the best-performing model (LSTM or GRU) for final use.

### Writer

[Ashigbui & Benning]

### Acknowledgments

- The project utilizes historical price and market data from [https://www.kaggle.com/].
- Gratitude to [Dr. Tatenda and Percy Brown] for valuable guidance and support.
