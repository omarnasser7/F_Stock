# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import arabic_reshaper
# from bidi.algorithm import get_display
# import joblib
# from keras.models import load_model

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


# rcParams['font.family'] = 'Arial'

# def fix_arabic_text(text):
#     reshaped_text = arabic_reshaper.reshape(text)
#     return get_display(reshaped_text)

# st.title("توقع حركة الأسهم عن طريق تحليلها بالذكاء الاصطناعي")

# markets = {"السوق الأمريكي": "US", "السوق السعودي": "SA"}

# selected_market = st.selectbox("اختر السوق:", list(markets.keys()))

# @st.cache_data
# def get_companies(market_code):
#     if market_code == "US":
#         return {"Apple": "AAPL", "Google": "GOOGL", "Amazon":"AMZN"}
#     elif market_code == "SA":
#         return {
#             "Aramco": "2222.SR",
#             "Sabic": "2010.SR",
#             "Al-Rajhi": "1180.SR",
#             "Leejam":"1830.SR"
#         }

# companies = get_companies(markets[selected_market])

# selected_company = st.selectbox("اختر الشركة:", list(companies.keys()))

# if selected_company:
#     company_symbol = companies[selected_company]

#     @st.cache_data
#     def get_stock_data(symbol):
#         return yf.download(symbol, period="3mo", interval="1d")

#     stock_data = get_stock_data(company_symbol)

#     if not stock_data.empty:
#         st.subheader(f"حركة سهم {selected_company} للشهرين الماضيين")
#         plt.figure(figsize=(10, 5))
#         plt.plot(stock_data.index, stock_data['Close'], label=fix_arabic_text("السعر الفعلي"))
#         plt.xlabel(fix_arabic_text("التاريخ"))
#         plt.ylabel(fix_arabic_text("السعر"))
#         plt.title(fix_arabic_text("حركة السهم"))
#         plt.legend()
#         st.pyplot(plt)

#         st.subheader(f"التوقعات للشركة {selected_company} للثلاثين جلسة القادمة")


#     else:
#         st.warning("لا توجد بيانات متاحة لهذه الشركة.")

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from matplotlib import rcParams
# import arabic_reshaper
# from bidi.algorithm import get_display
# import joblib
# from keras.models import load_model
# import base64


# # Set page configuration - must be the first Streamlit command
# st.set_page_config(page_title="Stock Prediction App", page_icon=":chart_with_upwards_trend:")

# # Models Loading Functions
# @st.cache_resource
# def load_prediction_models():
#     # Models for one day
#     models_1 = {
#         "Al-Rajhi": load_model("Al-Rajhi/trained_model/best_lstm_model_1.keras"),
#         "Amazon": load_model("Amazon/trained_model/best_lstm_model_1.keras"),
#         "Apple": load_model("Apple/trained_model/best_lstm_model_1.keras"),
#         "Aramco": load_model("Aramco/trained_model/best_lstm_model_1.keras"),
#         "Leejam": load_model("Leejam/trained_model/best_lstm_model_1.keras"),
#         "Sabic": load_model("Sabic/trained_model/best_lstm_model_1.keras"),
#         "Tesla": load_model("Tesla/trained_model/best_lstm_model_1.keras")
#     }
    
#     # Models for 5 days
#     models_5 = {
#         "Al-Rajhi": load_model("Al-Rajhi/trained_model/best_lstm_model_5.keras"),
#         "Amazon": load_model("Amazon/trained_model/best_lstm_model_5.keras"),
#         "Apple": load_model("Apple/trained_model/best_lstm_model_5.keras"),
#         "Aramco": load_model("Aramco/trained_model/best_lstm_model_5.keras"),
#         "Leejam": load_model("Leejam/trained_model/best_lstm_model_5.keras"),
#         "Sabic": load_model("Sabic/trained_model/best_lstm_model_5.keras"),
#         "Tesla": load_model("Tesla/trained_model/best_lstm_model_5.keras")
#     }
    
#     # Models for 15 days
#     models_15 = {
#         "Al-Rajhi": load_model("Al-Rajhi/trained_model/best_lstm_model_15.keras"),
#         "Amazon": load_model("Amazon/trained_model/best_lstm_model_15.keras"),
#         "Apple": load_model("Apple/trained_model/best_lstm_model_15.keras"),
#         "Aramco": load_model("Aramco/trained_model/best_lstm_model_15.keras"),
#         "Leejam": load_model("Leejam/trained_model/best_lstm_model_15.keras"),
#         "Sabic": load_model("Sabic/trained_model/best_lstm_model_15.keras"),
#         "Tesla": load_model("Tesla/trained_model/best_lstm_model_15.keras")
#     }
    
#     return models_1, models_5, models_15

# # Scaler Loading Functions
# @st.cache_resource
# def load_prediction_scalers():
#     # Scalers for one day
#     scalers_1 = {
#         "Al-Rajhi": joblib.load('Al-Rajhi\scaler\Al-Rajhi_scaler_1.joblib'),
#         "Amazon": joblib.load('Amazon/scaler/Amazon_scaler_1.joblib'),
#         "Apple": joblib.load('Apple/scaler/Apple_scaler_1.joblib'),
#         "Aramco": joblib.load('Aramco/scaler/Aramco_scaler_1.joblib'),
#         "Leejam": joblib.load('Leejam/scaler/Leejam_scaler_1.joblib'),
#         "Sabic": joblib.load('Sabic/scaler/Sabic_scaler_1.joblib'),
#         "Tesla": joblib.load('Tesla/scaler/Tesla_scaler_1.joblib')
#     }
    
#     # Scalers for 5 days
#     scalers_5 = {
#         "Al-Rajhi": joblib.load('Al-Rajhi\scaler\Al-Rajhi_scaler_5.joblib'),
#         "Amazon": joblib.load('Amazon/scaler/Amazon_scaler_5.joblib'),
#         "Apple": joblib.load('Apple/scaler/Apple_scaler_5.joblib'),
#         "Aramco": joblib.load('Aramco/scaler/Aramco_scaler_5.joblib'),
#         "Leejam": joblib.load('Leejam/scaler/Leejam_scaler_5.joblib'),
#         "Sabic": joblib.load('Sabic/scaler/Sabic_scaler_5.joblib'),
#         "Tesla": joblib.load('Tesla/scaler/Tesla_scaler_5.joblib')
#     }
    
#     # Scalers for 15 days
#     scalers_15 = {
#         "Al-Rajhi": joblib.load('Al-Rajhi\scaler\Al-Rajhi_scaler_15.joblib'),
#         "Amazon": joblib.load('Amazon/scaler/Amazon_scaler_15.joblib'),
#         "Apple": joblib.load('Apple/scaler/Apple_scaler_15.joblib'),
#         "Aramco": joblib.load('Aramco/scaler/Aramco_scaler_15.joblib'),
#         "Leejam": joblib.load('Leejam/scaler/Leejam_scaler_15.joblib'),
#         "Sabic": joblib.load('Sabic/scaler/Sabic_scaler_15.joblib'),
#         "Tesla": joblib.load('Tesla/scaler/Tesla_scaler_15.joblib')
#     }
    
#     return scalers_1, scalers_5, scalers_15

# # Function to set a background image
# def add_background_image(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpg;base64,{encoded_string}");
#             background-size: cover;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
    
# add_background_image("back_2.jpg")

# # Prediction Functions
# def prediction_1(model, scaler, symbol, seq_length=60):
#     stock_data = get_stock_data(symbol)
#     stock_data['Date'] = stock_data.index
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     last_value = stock_prices[-1]
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     prediction = model.predict(X_future)
#     prediction = scaler.inverse_transform(prediction)
    
#     return f"Last Value: {last_value[0]}, Next Value: {prediction[0][0]}"

# def prediction_5(model, scaler, symbol, seq_length=100):
#     stock_data = get_stock_data(symbol)
#     stock_data['Date'] = stock_data.index
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     predictions = model.predict(X_future)
#     predictions = scaler.inverse_transform(predictions)
    
#     predicted_prices = predictions[0]  # Prediction for next 5 days
    
#     start_date = datetime.now().strftime('%Y-%m-%d')
#     predicted_dates = pd.date_range(start=start_date, periods=5)  
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(predicted_dates, predicted_prices, label="Predicted Prices", color="orange", linewidth=2, linestyle="--", marker="o")
#     ax.set_title("Stock Price Prediction (Next 5 Days)", fontsize=16, fontweight='bold')
#     ax.set_xlabel("Date", fontsize=12)
#     ax.set_ylabel("Stock Price", fontsize=12)
#     ax.legend(fontsize=12)
#     ax.grid(alpha=0.5)
    
#     for i, price in enumerate(predicted_prices):
#         ax.text(predicted_dates[i], price, f'{price:.2f}', fontsize=10, ha='center', color="darkorange")
    
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig

# def prediction_15(model, scaler, symbol, seq_length=120):
#     stock_data = get_stock_data(symbol)
#     stock_data['Date'] = stock_data.index
#     stock_prices = stock_data['Close'].values.reshape(-1, 1)
#     stock_prices_scaled = scaler.transform(stock_prices)
#     X_future = stock_prices_scaled[-seq_length:]
#     X_future = np.expand_dims(X_future, axis=0)
#     predictions = model.predict(X_future)
#     predictions = scaler.inverse_transform(predictions)
    
#     predicted_prices = predictions[0] 

#     start_date = datetime.now().strftime('%Y-%m-%d')
#     predicted_dates = pd.date_range(start=start_date, periods=15)  

#     fig, ax = plt.subplots(figsize=(15, 10))

#     # Predicted prices
#     ax.plot(predicted_dates, predicted_prices, label="Predicted Prices", color="orange", linewidth=2, linestyle="--", marker="o")

#     # Enhancements
#     ax.set_title("Stock Price Prediction (Next 15 Days)", fontsize=16, fontweight='bold')
#     ax.set_xlabel("Date", fontsize=12)
#     ax.set_ylabel("Stock Price (USD)", fontsize=12)
#     ax.legend(fontsize=12)
#     ax.grid(alpha=1)

#     # Annotate predictions
#     for i, price in enumerate(predicted_prices):
#         ax.text(predicted_dates[i], price, f'{price:.2f}', fontsize=10, ha='center', color="darkorange")

#     # Formatting dates for better readability
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     return fig

# # Utility Functions
# def fix_arabic_text(text):
#     reshaped_text = arabic_reshaper.reshape(text)
#     return get_display(reshaped_text)

# @st.cache_data
# def get_stock_data(symbol):
#     return yf.download(symbol, period="3mo", interval="1d")

# @st.cache_data
# def get_companies(market_code):
#     if market_code == "US":
#         return {"Apple": "AAPL", "Amazon":"AMZN", "Tesla": "TSLA"}
#     elif market_code == "SA":
#         return {
#             "Aramco": "2222.SR",
#             "Sabic": "2010.SR",
#             "Al-Rajhi": "1180.SR",
#             "Leejam":"1830.SR"
#         }

# # Main Streamlit App
# def main():
#     # Load models and scalers
#     models_1, models_5, models_15 = load_prediction_models()
#     scalers_1, scalers_5, scalers_15 = load_prediction_scalers()

#     # Set up page configuration
#     rcParams['font.family'] = 'Arial'

#     # Title
#     st.title("توقع حركة الأسهم عن طريق تحليلها بالذكاء الاصطناعي")

#     # Market selection
#     markets = {"السوق الأمريكي": "US", "السوق السعودي": "SA"}
#     selected_market = st.selectbox("اختر السوق:", list(markets.keys()))

#     # Companies based on market
#     companies = get_companies(markets[selected_market])
#     selected_company = st.selectbox("اختر الشركة:", list(companies.keys()))

#     if selected_company:
#         company_symbol = companies[selected_company]

#         # Fetch stock data
#         stock_data = get_stock_data(company_symbol)

#         if not stock_data.empty:
#             # Plot historical stock prices
#             st.subheader(f"حركة سهم {selected_company} للشهرين الماضيين")
#             fig_historical, ax = plt.subplots(figsize=(10, 5))
#             ax.plot(stock_data.index, stock_data['Close'], label=fix_arabic_text("السعر الفعلي"))
#             ax.set_xlabel(fix_arabic_text("التاريخ"))
#             ax.set_ylabel(fix_arabic_text("السعر"))
#             ax.set_title(fix_arabic_text("حركة السهم"))
#             ax.legend()
#             st.pyplot(fig_historical)

#             # Prediction options
#             st.subheader(f"التوقعات للشركة {selected_company} للثلاثين جلسة القادمة")
#             prediction_type = st.selectbox("اختر نوع التوقع:", 
#                 ["توقع يوم واحد", "توقع 5 أيام", "توقع 15 يوم"])

#             # Perform predictions based on selected type
#             if prediction_type == "توقع يوم واحد":
#                 result = prediction_1(
#                     models_1[selected_company], 
#                     scalers_1[selected_company], 
#                     company_symbol
#                 )
#                 st.write(result)

#             elif prediction_type == "توقع 5 أيام":
#                 fig_5_days = prediction_5(
#                     models_5[selected_company], 
#                     scalers_5[selected_company], 
#                     company_symbol
#                 )
#                 st.pyplot(fig_5_days)

#             elif prediction_type == "توقع 15 يوم":
#                 fig_15_days = prediction_15(
#                     models_15[selected_company], 
#                     scalers_15[selected_company], 
#                     company_symbol
#                 )
#                 st.pyplot(fig_15_days)

#         else:
#             st.warning("لا توجد بيانات متاحة لهذه الشركة.")

# # Run the app
# if __name__ == "__main__":
#     main()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams
import arabic_reshaper
from bidi.algorithm import get_display
import joblib
from keras.models import load_model
import base64

# Set page configuration
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Models Loading Functions
@st.cache_resource
def load_prediction_models():
    models_1 = {
        "Al-Rajhi": load_model("Al-Rajhi/trained_model/best_lstm_model_1.keras"),
        "Amazon": load_model("Amazon/trained_model/best_lstm_model_1.keras"),
        "Apple": load_model("Apple/trained_model/best_lstm_model_1.keras"),
        "Aramco": load_model("Aramco/trained_model/best_lstm_model_1.keras"),
        "Leejam": load_model("Leejam/trained_model/best_lstm_model_1.keras"),
        "Sabic": load_model("Sabic/trained_model/best_lstm_model_1.keras"),
        "Tesla": load_model("Tesla/trained_model/best_lstm_model_1.keras")
    }
    models_5 = {
        "Al-Rajhi": load_model("Al-Rajhi/trained_model/best_lstm_model_5.keras"),
        "Amazon": load_model("Amazon/trained_model/best_lstm_model_5.keras"),
        "Apple": load_model("Apple/trained_model/best_lstm_model_5.keras"),
        "Aramco": load_model("Aramco/trained_model/best_lstm_model_5.keras"),
        "Leejam": load_model("Leejam/trained_model/best_lstm_model_5.keras"),
        "Sabic": load_model("Sabic/trained_model/best_lstm_model_5.keras"),
        "Tesla": load_model("Tesla/trained_model/best_lstm_model_5.keras")
    }
    models_15 = {
        "Al-Rajhi": load_model("Al-Rajhi/trained_model/best_lstm_model_15.keras"),
        "Amazon": load_model("Amazon/trained_model/best_lstm_model_15.keras"),
        "Apple": load_model("Apple/trained_model/best_lstm_model_15.keras"),
        "Aramco": load_model("Aramco/trained_model/best_lstm_model_15.keras"),
        "Leejam": load_model("Leejam/trained_model/best_lstm_model_15.keras"),
        "Sabic": load_model("Sabic/trained_model/best_lstm_model_15.keras"),
        "Tesla": load_model("Tesla/trained_model/best_lstm_model_15.keras")
    }
    return models_1, models_5, models_15

# Scaler Loading Functions
@st.cache_resource
def load_prediction_scalers():
    scalers_1 = {
        "Al-Rajhi": joblib.load('Al-Rajhi\scaler\Al-Rajhi_scaler_1.joblib'),
        "Amazon": joblib.load('Amazon/scaler/Amazon_scaler_1.joblib'),
        "Apple": joblib.load('Apple/scaler/Apple_scaler_1.joblib'),
        "Aramco": joblib.load('Aramco/scaler/Aramco_scaler_1.joblib'),
        "Leejam": joblib.load('Leejam/scaler/Leejam_scaler_1.joblib'),
        "Sabic": joblib.load('Sabic/scaler/Sabic_scaler_1.joblib'),
        "Tesla": joblib.load('Tesla/scaler/Tesla_scaler_1.joblib')
    }
    scalers_5 = {
        "Al-Rajhi": joblib.load('Al-Rajhi\scaler\Al-Rajhi_scaler_5.joblib'),
        "Amazon": joblib.load('Amazon/scaler/Amazon_scaler_5.joblib'),
        "Apple": joblib.load('Apple/scaler/Apple_scaler_5.joblib'),
        "Aramco": joblib.load('Aramco/scaler/Aramco_scaler_5.joblib'),
        "Leejam": joblib.load('Leejam/scaler/Leejam_scaler_5.joblib'),
        "Sabic": joblib.load('Sabic/scaler/Sabic_scaler_5.joblib'),
        "Tesla": joblib.load('Tesla/scaler/Tesla_scaler_5.joblib')
    }
    scalers_15 = {
        "Al-Rajhi": joblib.load('Al-Rajhi\scaler\Al-Rajhi_scaler_15.joblib'),
        "Amazon": joblib.load('Amazon/scaler/Amazon_scaler_15.joblib'),
        "Apple": joblib.load('Apple/scaler/Apple_scaler_15.joblib'),
        "Aramco": joblib.load('Aramco/scaler/Aramco_scaler_15.joblib'),
        "Leejam": joblib.load('Leejam/scaler/Leejam_scaler_15.joblib'),
        "Sabic": joblib.load('Sabic/scaler/Sabic_scaler_15.joblib'),
        "Tesla": joblib.load('Tesla/scaler/Tesla_scaler_15.joblib')
    }
    return scalers_1, scalers_5, scalers_15

# Add a stylish background
@st.cache_data
def add_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .stMarkdown h1, h2, h3, h4, h5, h6 {{
            color: #ffffff;
            text-shadow: 2px 2px 5px #000000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_background_image("back_4.jpg")

# Enhance plots with custom styling
def enhance_plot(fig, ax, title, x_label, y_label):
    ax.set_title(title, fontsize=18, fontweight='bold', color="#FF4500")
    ax.set_xlabel(x_label, fontsize=14, color="#4682B4")
    ax.set_ylabel(y_label, fontsize=14, color="#4682B4")
    ax.grid(alpha=0.3)
    fig.patch.set_facecolor("#f0f0f0")
    return fig

@st.cache_data
def get_stock_data(symbol):
    return yf.download(symbol, period="3mo", interval="1d")

@st.cache_data
def get_companies(market_code):
    if market_code == "US":
        return {"Apple": "AAPL", "Tesla": "TSLA", "Amazon":"AMZN"}
    elif market_code == "SA":
        return {
            "Aramco": "2222.SR",
            "Sabic": "2010.SR",
            "Al-Rajhi": "1180.SR",
            "Leejam":"1830.SR"
        }

# Prediction Functions
def prediction_1(model, scaler, symbol, seq_length=60):
    stock_data = get_stock_data(symbol)
    stock_data['Date'] = stock_data.index
    stock_prices = stock_data['Close'].values.reshape(-1, 1)
    last_value = stock_prices[-1]
    stock_prices_scaled = scaler.transform(stock_prices)
    X_future = stock_prices_scaled[-seq_length:]
    X_future = np.expand_dims(X_future, axis=0)
    prediction = model.predict(X_future)
    prediction = scaler.inverse_transform(prediction)
    
    return f"Last Value: {round(last_value[0], 3)}, Next Value: {round(prediction[0][0], 3)}"

def prediction_5(model, scaler, symbol, seq_length=100):
    stock_data = get_stock_data(symbol)
    stock_data['Date'] = stock_data.index
    stock_prices = stock_data['Close'].values.reshape(-1, 1)
    stock_prices_scaled = scaler.transform(stock_prices)
    X_future = stock_prices_scaled[-seq_length:]
    X_future = np.expand_dims(X_future, axis=0)
    predictions = model.predict(X_future)
    predictions = scaler.inverse_transform(predictions)
    
    predicted_prices = predictions[0]  # Prediction for next 5 days
    
    start_date = datetime.now().strftime('%Y-%m-%d')
    predicted_dates = pd.date_range(start=start_date, periods=5)  
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(predicted_dates, predicted_prices, label="Predicted Prices", color="orange", linewidth=2, linestyle="--", marker="o")
    ax.set_title("Stock Price Prediction (Next 5 Days)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.5)
    
    for i, price in enumerate(predicted_prices):
        ax.text(predicted_dates[i], price, f'{price:.2f}', fontsize=10, ha='center', color="darkorange")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def prediction_15(model, scaler, symbol, seq_length=120):
    stock_data = get_stock_data(symbol)
    stock_data['Date'] = stock_data.index
    stock_prices = stock_data['Close'].values.reshape(-1, 1)
    stock_prices_scaled = scaler.transform(stock_prices)
    X_future = stock_prices_scaled[-seq_length:]
    X_future = np.expand_dims(X_future, axis=0)
    predictions = model.predict(X_future)
    predictions = scaler.inverse_transform(predictions)
    
    predicted_prices = predictions[0] 

    start_date = datetime.now().strftime('%Y-%m-%d')
    predicted_dates = pd.date_range(start=start_date, periods=15)  

    fig, ax = plt.subplots(figsize=(15, 10))

    # Predicted prices
    ax.plot(predicted_dates, predicted_prices, label="Predicted Prices", color="orange", linewidth=2, linestyle="--", marker="o")

    # Enhancements
    ax.set_title("Stock Price Prediction (Next 15 Days)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Stock Price (USD)", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=1)

    # Annotate predictions
    for i, price in enumerate(predicted_prices):
        ax.text(predicted_dates[i], price, f'{price:.2f}', fontsize=10, ha='center', color="darkorange")

    # Formatting dates for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    models_1, models_5, models_15 = load_prediction_models()
    scalers_1, scalers_5, scalers_15 = load_prediction_scalers()

    st.title("توقع حركة الأسهم عن طريق تحليلها بالذكاء الاصطناعي")

    # Sidebar for market selection
    st.sidebar.header("خيارات السوق")
    markets = {"السوق الأمريكي": "US", "السوق السعودي": "SA"}
    selected_market = st.sidebar.radio("اختر السوق:", list(markets.keys()))

    companies = get_companies(markets[selected_market])
    selected_company = st.sidebar.selectbox("اختر الشركة:", list(companies.keys()))

    if selected_company:
        company_symbol = companies[selected_company]
        stock_data = get_stock_data(company_symbol)

        if not stock_data.empty:
            st.subheader(f"حركة سهم {selected_company} للشهرين الماضيين")
            fig_historical, ax = plt.subplots(figsize=(10, 5))
            ax.plot(stock_data.index, stock_data['Close'], label="Actual Price", color="#32CD32", linewidth=2)
            fig = enhance_plot(fig_historical, ax, "Stock Price Movement", "Date", "Price")
            st.pyplot(fig)

            st.subheader(f"التوقعات للشركة {selected_company} للثلاثين جلسة القادمة")
            prediction_type = st.radio("اختر نوع التوقع:", ["توقع يوم واحد", "توقع 5 أيام", "توقع 15 يوم"])

            if prediction_type == "توقع يوم واحد":
                result = prediction_1(models_1[selected_company], scalers_1[selected_company], company_symbol)
                st.markdown(f"### {result}", unsafe_allow_html=True)

            elif prediction_type == "توقع 5 أيام":
                fig_5_days = prediction_5(models_5[selected_company], scalers_5[selected_company], company_symbol)
                st.pyplot(fig_5_days)

            elif prediction_type == "توقع 15 يوم":
                fig_15_days = prediction_15(models_15[selected_company], scalers_15[selected_company], company_symbol)
                st.pyplot(fig_15_days)

if __name__ == "__main__":
    main()


