import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error


df_processed = pd.read_csv("./Data/output_file_final.csv")

## Date format Conversion
df_processed['Date'] = pd.to_datetime(df_processed['Date'])


### UI design #############################################
st.title("Fuel Price Analysis")

# Display the DataFrame
st.subheader("Fuel Price Data")
st.dataframe(df_processed)

# Sidebar for user interaction
st.sidebar.header("Fuel Type  :")
selected_columns =st.sidebar.multiselect("Select Fuel Type for Visualization", df_processed.columns[1:], default=df_processed.columns[1])


# Plotting the historial data based on user selections
if selected_columns:
    st.subheader("Variation of selected fuel price:")

    for col in selected_columns:
        # Create an interactive plot using Plotly
        fig = px.line(df_processed, x='Date', y=col, title=f"Variation of {col}")

        # Add hover functionality for labels
        fig.update_traces(mode='lines+markers', hoverinfo='x+y+text', 
                          marker=dict(size=8), hovertemplate='Date: %{x}<br>Value: %{y}')

        # Show the plot in Streamlit
        st.plotly_chart(fig)
else:
    st.write("Please select columns to visualize.")


######## MODELING AND  FOREASTING ##############

# Calculate moving averages for different window sizes
window_sizes = range(3, 31)  # Trying window sizes from 3 to 30
best_window_size = None
best_mse = float('inf')
best_moving_avg = None

# Choose a specific column to model, e.g., 'LP 95'


column_to_model = selected_columns

if selected_columns:
    st.subheader("Modeling of selected fuel price:")

    for col in selected_columns:

        column_to_model = col     

        for window_size in window_sizes:
            moving_avg = df_processed[column_to_model].rolling(window=window_size, min_periods=1).mean()
            mse = mean_squared_error(df_processed[column_to_model], moving_avg)

            if mse < best_mse:
                best_mse = mse
                best_window_size = window_size
                best_moving_avg = moving_avg

        # Step 4: Evaluate the best window size
        round_mse = round(best_mse,2)
        #print(f"Best window size: {best_window_size} with MSE: {best_mse}")

        # Step 5: Visualize the results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_processed['Date'], df_processed[column_to_model], label='Original Data')
        ax.plot(df_processed['Date'], best_moving_avg, label=f'Moving Average (Window={best_window_size})', color='red')
        
        # Set the x-axis format to show only the year

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        #Set the x-axis ticks to show every 5 years
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.set_xlabel('Year')

        ax.set_ylabel('Price')
        ax.set_title(f'Moving Average for {column_to_model}')
        ax.legend()

        st.pyplot(fig)

        ## Visualize the window size and MSE

        st.subheader(f'The best window for the MA model is : {best_window_size}')
        st.subheader(f'The MSE is : {round_mse}')

        # Step 5: Forecast the next 3 price values
        last_date = df_processed['Date'].iloc[-1]  # Last date
        last_price = df_processed[column_to_model].iloc[-1]  # Last price

        forecast_dates = [last_date + np.timedelta64(i, 'D') for i in range(1, 4)]  # Forecast dates
        forecast_prices = [last_price]  # Start with the last price

        # Apply moving average to forecast next 3 values
        for _ in range(3):
            forecast_prices.append(np.mean(forecast_prices[-window_size:]))  # Using the window size to forecast

        forecast_prices = forecast_prices[1:]  # Remove the last known price (it was used to start the forecast)

        # Add forecasted prices to the DataFrame (for plotting)
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            column_to_model: forecast_prices
        })

        st.subheader("Forecasted Fuel Price : ")

        st.dataframe(forecast_df)





