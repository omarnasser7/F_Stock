# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import arabic_reshaper
# from bidi.algorithm import get_display
# from keras.models import load_model
# import joblib
# from datetime import datetime

# # Models
# # models for one day
# rajhi_model_1 = load_model("Al-Rajhi/trained_model/best_lstm_model_1.keras")
# amazon_model_1 = load_model("Amazon/trained_model/Amazon_model_1.keras")
# apple_model_1 = load_model("Apple/trained_model/best_lstm_model_1.keras")
# aramco_model_1 = load_model("Aramco/trained_model/best_lstm_model_1.keras")
# leejam_model_1 = load_model("Leejam/trained_model/best_lstm_model_1.keras")
# sabic_model_1 = load_model("Sabic/trained_model/best_lstm_model_1.keras")
# tesla_model_1 = load_model("Tesla/trained_model/best_lstm_model_1.keras")

# # models for 5 days
# rajhi_model_5 = load_model("Al-Rajhi/trained_model/best_lstm_model_5.keras")
# amazon_model_5 = load_model("Amazon/trained_model/best_lstm_model_5.keras")
# apple_model_5 = load_model("Apple/trained_model/best_lstm_model_5.keras")
# aramco_model_5 = load_model("Aramco/trained_model/best_lstm_model_5.keras")
# leejam_model_5 = load_model("Leejam/trained_model/best_lstm_model_5.keras")
# sabic_model_5 = load_model("Sabic/trained_model/best_lstm_model_5.keras")
# tesla_model_5 = load_model("Tesla/trained_model/best_lstm_model_5.keras")

# # models for 15 days
# rajhi_model_15 = load_model("Al-Rajhi/trained_model/best_lstm_model_15.keras")
# amazon_model_15 = load_model("Amazon/trained_model/best_lstm_model_15.keras")
# apple_model_15 = load_model("Apple/trained_model/best_lstm_model_15.keras")
# aramco_model_15 = load_model("Aramco/trained_model/best_lstm_model_15.keras")
# leejam_model_15 = load_model("Leejam/trained_model/best_lstm_model_15.keras")
# sabic_model_15 = load_model("Sabic/trained_model/best_lstm_model_15.keras")
# tesla_model_15 = load_model("Tesla/trained_model/best_lstm_model_15.keras")


# # Scaler
# # scaler for one day
# rajhi_scaler_1 = joblib.load('Al-Rajhi/scaler/rajhi_scaler_1.joblib')
# amazon_scaler_1 = joblib.load('Amazon/scaler/amazon_scaler_1.joblib')
# apple_scaler_1 = joblib.load('Apple/scaler/appl_scaler_1.joblib')
# aramco_scaler_1 = joblib.load('Aramco/scaler/aramco_stock_1.joblib')
# leejam_scaler_1 = joblib.load('Leejam/scaler/leejam_scaler_1.joblib')
# sabic_scaler_1 = joblib.load('Sabic/scaler/sabic_scaler_1.joblib')
# tesla_scaler_1 = joblib.load('Tesla/scaler/tesla_scaler_1.joblib')

# # scaler for 5 days
# rajhi_scaler_5 = joblib.load('Al-Rajhi/scaler/rajhi_scaler_5.joblib')
# amazon_scaler_5 = joblib.load('Amazon/scaler/amazon_scaler_5.joblib')
# apple_scaler_5 = joblib.load('Apple/scaler/appl_scaler_5.joblib')
# aramco_scaler_5 = joblib.load('Aramco/scaler/aramco_stock_5.joblib')
# leejam_scaler_5 = joblib.load('Leejam/scaler/leejam_scaler_5.joblib')
# sabic_scaler_5 = joblib.load('Sabic/scaler/sabic_scaler_5.joblib')
# tesla_scaler_5 = joblib.load('Tesla/scaler/tesla_scaler_5.joblib')

# # scaler for 15 days 
# rajhi_scaler_15 = joblib.load('Al-Rajhi/scaler/rajhi_scaler_15.joblib')
# amazon_scaler_15 = joblib.load('Amazon/scaler/amazon_scaler_15.joblib')
# apple_scaler_15 = joblib.load('Apple/scaler/appl_scaler_15.joblib')
# aramco_scaler_15 = joblib.load('Aramco/scaler/aramco_stock_15.joblib')
# leejam_scaler_15 = joblib.load('Leejam/scaler/leejam_scaler_15.joblib')
# sabic_scaler_15 = joblib.load('Sabic/scaler/sabic_scaler_15.joblib')
# tesla_scaler_15 = joblib.load('Tesla/scaler/tesla_scaler_15.joblib')

# # Set font for matplotlib
# rcParams['font.family'] = 'Arial'

# # Fix Arabic text
# def fix_arabic_text(text):
#     reshaped_text = arabic_reshaper.reshape(text)
#     return get_display(reshaped_text)

# # Load models and scalers
# @st.cache_data
# def load_resources():
#     models = {
#         "1-day": {"AMZN": amazon_model_1, "ARAMCO": aramco_model_1, "APPLE": apple_model_1},
#         "5-day": {"AMZN": amazon_model_5, "ARAMCO": aramco_model_5, "APPLE": apple_model_5},
#         "15-day": {"AMZN": amazon_model_15, "ARAMCO": aramco_model_15, "APPLE": apple_model_15},
#     }
#     scalers = {
#         "1-day": {"AMZN": amazon_scaler_1, "ARAMCO": aramco_scaler_1, "APPLE": apple_scaler_1},
#         "5-day": {"AMZN": amazon_scaler_5, "ARAMCO": aramco_scaler_5, "APPLE": apple_scaler_5},
#         "15-day": {"AMZN": amazon_scaler_15, "ARAMCO": aramco_scaler_15, "APPLE": apple_scaler_15},
#     }
#     return models, scalers

# models, scalers = load_resources()

# # Get stock data
# @st.cache_data
# def get_stock_data(symbol):
#     return yf.download(symbol, period="3mo", interval="1d")

# # Prediction functions
# def predict_next_value(model, scaler, stock_data, seq_length=60):
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     last_value = stock_prices[-1]
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     prediction = model.predict(X_future)
#     prediction = scaler.inverse_transform(prediction)
#     return last_value[0], prediction[0][0]

# def predict_future_prices(model, scaler, stock_data, seq_length, periods):
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     predictions = model.predict(X_future)
#     predictions = scaler.inverse_transform(predictions)
#     return predictions[0], pd.date_range(start=datetime.now().strftime('%Y-%m-%d'), periods=periods)

# # Streamlit app layout
# st.title("توقع حركة الأسهم باستخدام الذكاء الاصطناعي")

# markets = {"السوق الأمريكي": "US", "السوق السعودي": "SA"}
# selected_market = st.selectbox("اختر السوق:", list(markets.keys()))
# companies = {
#     "US": {"Apple": "AAPL", "Amazon": "AMZN"},
#     "SA": {"Aramco": "2222.SR", "Sabic": "2010.SR"}
# }[markets[selected_market]]

# selected_company = st.selectbox("اختر الشركة:", list(companies.keys()))

# if selected_company:
#     symbol = companies[selected_company]
#     stock_data = get_stock_data(symbol)

#     if not stock_data.empty:
#         st.subheader(f"حركة سهم {selected_company} للشهرين الماضيين")
#         plt.figure(figsize=(10, 5))
#         plt.plot(stock_data.index, stock_data['Close'], label=fix_arabic_text("السعر الفعلي"))
#         plt.xlabel(fix_arabic_text("التاريخ"))
#         plt.ylabel(fix_arabic_text("السعر"))
#         plt.legend()
#         st.pyplot(plt)

#         st.subheader(f"التوقعات للشركة {selected_company} للجلسات القادمة")

#         prediction_type = st.selectbox("اختر فترة التوقع:", ["1 يوم", "5 أيام", "15 يومًا"])
#         if prediction_type == "1 يوم":
#             last, next_price = predict_next_value(
#                 models["1-day"][selected_company.upper()],
#                 scalers["1-day"][selected_company.upper()],
#                 stock_data
#             )
#             st.write(f"آخر سعر: {last:.2f}, التوقع لليوم التالي: {next_price:.2f}")
#         else:
#             days = 5 if prediction_type == "5 أيام" else 15
#             prices, dates = predict_future_prices(
#                 models[f"{days}-day"][selected_company.upper()],
#                 scalers[f"{days}-day"][selected_company.upper()],
#                 stock_data,
#                 seq_length=60 if days == 5 else 120,
#                 periods=days
#             )
#             plt.figure(figsize=(10, 6))
#             plt.plot(dates, prices, label=fix_arabic_text("السعر المتوقع"), color="orange", marker="o")
#             plt.xlabel(fix_arabic_text("التاريخ"))
#             plt.ylabel(fix_arabic_text("السعر المتوقع"))
#             plt.legend()
#             plt.grid(alpha=0.5)
#             st.pyplot(plt)

#     else:
#         st.warning("لا توجد بيانات متاحة لهذه الشركة.")

# Set page config first, before any other Streamlit commands
#########################
# import streamlit as st
# st.set_page_config(
#     page_title="AI Stock Prediction",
#     page_icon="📈",
#     layout="wide"
# )

# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# from matplotlib import rcParams
# import arabic_reshaper
# from bidi.algorithm import get_display
# import joblib
# from keras.models import load_model
# import plotly.graph_objects as go
# import plotly.express as px
# from functools import lru_cache
# import concurrent.futures

# # Custom CSS
# st.markdown("""
#     <style>
#     .main {
#         padding: 2rem;
#     }
#     .stTitle {
#         color: #2c3e50;
#         font-size: 2.5rem !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Performance Optimization: Cache models and scalers
# @st.cache_resource
# def load_prediction_models():
#     models = {
#         period: {
#             company: load_model(f"{company}/trained_model/best_lstm_model_{period}.keras")
#             for company in ["Al-Rajhi", "Amazon", "Apple", "Aramco", "Leejam", "Sabic", "Tesla"]
#         }
#         for period in [1, 5, 15]
#     }
#     return models

# @st.cache_resource
# def load_prediction_scalers():
#     scalers = {
#         period: {
#             company: joblib.load(f"{company}/scaler/{company.lower()}_scaler_{period}.joblib")
#             for company in ["Al-Rajhi", "Amazon", "Apple", "Aramco", "Leejam", "Sabic", "Tesla"]
#         }
#         for period in [1, 5, 15]
#     }
#     return scalers

# # Enhanced stock data fetching with caching
# @st.cache_data(ttl=3600)  # Cache for 1 hour
# def get_stock_data(symbol, period="3mo"):
#     try:
#         return yf.download(symbol, period=period, interval="1d")
#     except Exception as e:
#         st.error(f"Error fetching data for {symbol}: {str(e)}")
#         return pd.DataFrame()

# def get_market_sentiment(stock_data):
#     """Calculate market sentiment based on technical indicators"""
#     if stock_data.empty:
#         return "Neutral", 0
    
#     # Calculate technical indicators and convert to float
#     sma_20 = float(stock_data['Close'].rolling(window=20).mean().iloc[-1])
#     sma_50 = float(stock_data['Close'].rolling(window=50).mean().iloc[-1])
#     current_price = float(stock_data['Close'].iloc[-1])
    
#     # RSI calculation
#     delta = stock_data['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     rsi = float(100 - (100 / (1 + rs)).iloc[-1])
    
#     # Determine sentiment
#     sentiment_score = 0
#     if current_price > sma_20:
#         sentiment_score += 1
#     if current_price > sma_50:
#         sentiment_score += 1
#     if 30 <= rsi <= 70:
#         sentiment_score += 1
    
#     sentiment_map = {
#         0: "Bearish",
#         1: "Slightly Bearish",
#         2: "Neutral",
#         3: "Bullish"
#     }
    
#     return sentiment_map[sentiment_score], rsi

# def create_interactive_chart(stock_data, predictions=None, company_name=""):
#     fig = go.Figure()
    
#     # Historical data
#     fig.add_trace(go.Scatter(
#         x=stock_data.index,
#         y=stock_data['Close'],
#         name="Historical Price",
#         line=dict(color='blue')
#     ))
    
#     # Predictions if available
#     if predictions is not None:
#         pred_dates = pd.date_range(start=stock_data.index[-1], periods=len(predictions)+1)[1:]
#         fig.add_trace(go.Scatter(
#             x=pred_dates,
#             y=predictions,
#             name="Predicted Price",
#             line=dict(color='orange', dash='dash')
#         ))
    
#     # Layout
#     fig.update_layout(
#         title=f"{company_name} Stock Price Analysis",
#         xaxis_title="Date",
#         yaxis_title="Price",
#         hovermode='x unified',
#         template='plotly_white'
#     )
    
#     return fig

# def prediction_1(model, scaler, stock_data, seq_length=60):
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     prediction = model.predict(X_future, verbose=0)
#     return scaler.inverse_transform(prediction)[0][0]

# def prediction_n(model, scaler, stock_data, n_days, seq_length=60):
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     predictions = model.predict(X_future, verbose=0)
#     return scaler.inverse_transform(predictions)[0]

# def show_technical_analysis(stock_data):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Moving Averages
#         ma_20 = stock_data['Close'].rolling(window=20).mean()
#         ma_50 = stock_data['Close'].rolling(window=50).mean()
        
#         fig_ma = go.Figure()
#         fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name="Price"))
#         fig_ma.add_trace(go.Scatter(x=stock_data.index, y=ma_20, name="20-day MA"))
#         fig_ma.add_trace(go.Scatter(x=stock_data.index, y=ma_50, name="50-day MA"))
#         fig_ma.update_layout(title="Moving Averages Analysis")
#         st.plotly_chart(fig_ma)
    
#     with col2:
#         # Volume Analysis
#         fig_vol = go.Figure()
#         fig_vol.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name="Volume"))
#         fig_vol.update_layout(title="Volume Analysis")
#         st.plotly_chart(fig_vol)

# def main():
#     # Load models and scalers
#     models = load_prediction_models()
#     scalers = load_prediction_scalers()
    
#     # Title with emoji
#     st.title("🤖 AI-Powered Stock Analysis & Prediction")
    
#     # Sidebar for settings
#     with st.sidebar:
#         st.header("📊 Market Settings")
#         markets = {"US Market 🇺🇸": "US", "Saudi Market 🇸🇦": "SA"}
#         selected_market = st.selectbox("Select Market:", list(markets.keys()))
        
#         # Dynamic company selection
#         companies = {
#             "US": {"Apple": "AAPL", "Amazon": "AMZN", "Tesla": "TSLA"},
#             "SA": {
#                 "Aramco": "2222.SR",
#                 "Al-Rajhi": "1180.SR",
#                 "Sabic": "2010.SR",
#                 "Leejam": "1830.SR"
#             }
#         }
        
#         selected_company = st.selectbox(
#             "Select Company:",
#             list(companies[markets[selected_market]].keys())
#         )
        
#     # Main content area
#     if selected_company:
#         company_symbol = companies[markets[selected_market]][selected_company]
        
#         # Fetch data with progress bar
#         with st.spinner("Fetching market data..."):
#             stock_data = get_stock_data(company_symbol)
        
#         if not stock_data.empty:
#             # Market Overview Section
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 current_price = float(stock_data['Close'].iloc[-1])  # Convert to float
#                 price_change = float(stock_data['Close'].pct_change().iloc[-1] * 100)  # Convert to float
#                 st.metric(
#                     "Current Price",
#                     f"${current_price:,.2f}",  # Changed format specifier
#                     f"{price_change:,.2f}%"    # Changed format specifier
#                 )
            
#             with col2:
#                 sentiment, rsi = get_market_sentiment(stock_data)
#                 st.metric("Market Sentiment", sentiment, f"{rsi:,.2f}")  # Changed format specifier
            
#             with col3:
#                 volume = int(stock_data['Volume'].iloc[-1])  # Convert to int
#                 vol_change = float(stock_data['Volume'].pct_change().iloc[-1] * 100)  # Convert to float
#                 st.metric(
#                     "Trading Volume",
#                     f"{volume:,}",          # Changed format specifier
#                     f"{vol_change:,.2f}%"   # Changed format specifier
#                 )

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Performance Optimization: Cache models and scalers
@st.cache_resource
def load_prediction_models():
    models = {
        period: {
            company: load_model(f"{company}/trained_model/best_lstm_model_{period}.keras")
            for company in ["Al-Rajhi", "Amazon", "Apple", "Aramco", "Leejam", "Sabic", "Tesla"]
        }
        for period in [1, 5, 15]
    }
    return models

@st.cache_resource
def load_prediction_scalers():
    scalers = {
        period: {
            company: joblib.load(f"{company}/scaler/{company.lower()}_scaler_{period}.joblib")
            for company in ["Al-Rajhi", "Amazon", "Apple", "Aramco", "Leejam", "Sabic", "Tesla"]
        }
        for period in [1, 5, 15]
    }
    return scalers

# Main Streamlit App
def main():
    # Load models and scalers with caching
    models = load_prediction_models()
    scalers = load_prediction_scalers()

    # Set up page configuration
    rcParams['font.family'] = 'Arial'
    st.set_page_config(page_title="Stock Prediction App", page_icon=":chart_with_upwards_trend:")

    # Title
    st.title("\u062a\u0648\u0642\u0639 \u062d\u0631\u0643\u0629 \u0627\u0644\u0623\u0633\u0647\u0645 \u0639\u0646 \u0637\u0631\u064a\u0642 \u062a\u062d\u0644\u064a\u0644\u0647\u0627 \u0628\u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a")

    # Dropdown menus for market and company selection
    market = st.selectbox("\u0627\u062e\u062a\u0631 \u0627\u0644\u0633\u0648\u0642", ["US", "Saudi"])

    if market == "US":
        company = st.selectbox("\u0627\u062e\u062a\u0631 \u0627\u0644\u0634\u0631\u0643\u0629", ["Amazon", "Apple", "Tesla"])
    else:
        company = st.selectbox("\u0627\u062e\u062a\u0631 \u0627\u0644\u0634\u0631\u0643\u0629", ["Al-Rajhi", "Aramco", "Leejam", "Sabic"])

    # Dropdown for prediction period
    prediction_period = st.selectbox("\u0627\u062e\u062a\u0631 \u0645\u062f\u0629 \u0627\u0644\u062a\u0648\u0642\u0639", [1, 5, 15])

    # Fetch selected model and scaler
    selected_model = models[prediction_period][company]
    selected_scaler = scalers[prediction_period][company]

    # Upload CSV for historical data
    uploaded_file = st.file_uploader("\u0627\u0631\u0641\u0639 \u0645\u0644\u0641 CSV \u0628\u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a \u0627\u0644\u062a\u0627\u0631\u064a\u062e\u064a\u0629")

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Ensure necessary columns exist
        if not set(["Open", "High", "Low", "Close", "Volume"]).issubset(data.columns):
            st.error("\u062a\u0623\u0643\u062f \u0645\u0646 \u0627\u062d\u062a\u0648\u0627\u0621 \u0627\u0644\u0645\u0644\u0641 \u0639\u0644\u0649 \u0627\u0644\u0623\u0639\u0645\u062f\u0629 \u0627\u0644\u0644\u0627\u0632\u0645\u0629: Open, High, Low, Close, Volume")
        else:
            # Preprocess data
            features = data[["Open", "High", "Low", "Close", "Volume"]]
            scaled_features = selected_scaler.transform(features)

            # Prepare input for the model
            sequence_length = 60  # Assuming a fixed sequence length of 60
            sequences = []

            for i in range(sequence_length, len(scaled_features)):
                sequences.append(scaled_features[i-sequence_length:i])

            sequences = np.array(sequences)

            # Make predictions
            predictions = selected_model.predict(sequences)
            inverse_predictions = selected_scaler.inverse_transform(predictions)

            # Visualize predictions
            st.subheader("\u0627\u0644\u062a\u0648\u0642\u0639\u0627\u062a")
            plt.figure(figsize=(10, 6))
            plt.plot(data["Close"].values[sequence_length:], label="\u0627\u0644\u0642\u064a\u0645 \u0627\u0644\u062d\u0642\u064a\u0642\u064a\u0629")
            plt.plot(inverse_predictions, label="\u0627\u0644\u062a\u0648\u0642\u0639\u0627\u062a")
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()
