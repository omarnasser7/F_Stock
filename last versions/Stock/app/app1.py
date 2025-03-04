# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import arabic_reshaper
# from bidi.algorithm import get_display

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
#         return {"Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL"}
#     elif market_code == "SA":
#         return {
#             "أرامكو": "2222.SR",
#             "سابك": "2010.SR",
#             "البنك الأهلي": "1180.SR",
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

#         # استبدل هذا القسم بالتوقعات الفعلية باستخدام النموذج
#         future_dates = pd.date_range(stock_data.index[-1], periods=31, freq="B")[1:]
#         predicted_prices = stock_data['Close'][-1] + (range(30))  # مثال للتوقعات

#         plt.figure(figsize=(10, 5))
#         plt.plot(stock_data.index, stock_data['Close'], label=fix_arabic_text("السعر الفعلي"))
#         plt.plot(future_dates, predicted_prices, label=fix_arabic_text("السعر المتوقع"), linestyle="--")
#         plt.xlabel(fix_arabic_text("التاريخ"))
#         plt.ylabel(fix_arabic_text("السعر"))
#         plt.title(fix_arabic_text("التوقعات المستقبلية"))
#         plt.legend()
#         st.pyplot(plt)
#     else:
#         st.warning("لا توجد بيانات متاحة لهذه الشركة.")

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import arabic_reshaper
from bidi.algorithm import get_display

rcParams['font.family'] = 'Arial'

def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

st.title(fix_arabic_text("توقع حركة الأسهم عن طريق تحليلها بالذكاء الاصطناعي"))

markets = {"السوق الأمريكي": "US", "السوق السعودي": "SA"}

selected_market = st.selectbox(fix_arabic_text("اختر السوق:"), list(markets.keys()))

@st.cache_data
def get_us_companies():
    # قائمة الشركات الكبرى في مؤشر S&P 500
    sp500 = yf.Ticker("^GSPC").history(period="1d")
    # مثال للشركات الكبرى (يمكنك تحديث القائمة بجلب كامل الشركات)
    return {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
    }

@st.cache_data
def get_sa_companies():
    # قائمة الشركات الكبرى في السوق السعودي
    return {
        "أرامكو": "2222.SR",
        "سابك": "2010.SR",
        "البنك الأهلي": "1180.SR",
        "الاتصالات السعودية": "7010.SR",
        "الراجحي": "1120.SR",
    }

@st.cache_data
def get_companies(market_code):
    if market_code == "US":
        return get_us_companies()
    elif market_code == "SA":
        return get_sa_companies()

companies = get_companies(markets[selected_market])

selected_company = st.selectbox(fix_arabic_text("اختر الشركة:"), list(companies.keys()))

if selected_company:
    company_symbol = companies[selected_company]

    @st.cache_data
    def get_stock_data(symbol):
        return yf.download(symbol, period="3mo", interval="1d")

    stock_data = get_stock_data(company_symbol)

    if not stock_data.empty:
        st.subheader(fix_arabic_text(f"حركة سهم {selected_company} للشهرين الماضيين"))
        plt.figure(figsize=(10, 5))
        plt.plot(stock_data.index, stock_data['Close'], label=fix_arabic_text("السعر الفعلي"))
        plt.xlabel(fix_arabic_text("التاريخ"))
        plt.ylabel(fix_arabic_text("السعر"))
        plt.title(fix_arabic_text("حركة السهم"))
        plt.legend()
        st.pyplot(plt)

        st.subheader(fix_arabic_text(f"التوقعات للشركة {selected_company} للثلاثين جلسة القادمة"))

        # استبدل هذا القسم بالتوقعات الفعلية باستخدام النموذج
        last_close = stock_data['Close'].iloc[-1]
        future_dates = pd.date_range(stock_data.index[-1], periods=31, freq="B")[1:]
        predicted_prices = last_close + pd.Series(range(30))  # مثال للتوقعات

        plt.figure(figsize=(10, 5))
        plt.plot(stock_data.index, stock_data['Close'], label=fix_arabic_text("السعر الفعلي"))
        plt.plot(future_dates, predicted_prices, label=fix_arabic_text("السعر المتوقع"), linestyle="--")
        plt.xlabel(fix_arabic_text("التاريخ"))
        plt.ylabel(fix_arabic_text("السعر"))
        plt.title(fix_arabic_text("التوقعات المستقبلية"))
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning(fix_arabic_text("لا توجد بيانات متاحة لهذه الشركة."))
