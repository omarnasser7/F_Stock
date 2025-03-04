import gradio as gr
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

markets = {"السوق الأمريكي": "US", "السوق السعودي": "SA"}

def get_companies(market):
    """
    Retrieve list of companies based on the selected market.
    
    Args:
        market (str): Market code (US or SA)
    
    Returns:
        list: List of companies with their ticker symbols
    """
    if market == "US":
        return ["AAPL (Apple)", "MSFT (Microsoft)", "GOOGL (Alphabet)"]
    elif market == "SA":
        return ["2222.SR (Aramco)", "2010.SR (SABIC)", "1120.SR (Al Rajhi Bank)"]
    return []

def fetch_stock_data(market, company):
    """
    Fetch stock data and create a plot for the selected company.
    
    Args:
        market (str): Market code (US or SA)
        company (str): Company ticker and name
    
    Returns:
        str: Path to the generated plot image
    """
    ticker = company.split()[0]
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    hist = stock.history(start=start_date, end=end_date)

    if hist.empty:
        return "لا توجد بيانات متوفرة لهذا السهم."
    
    # Simple linear projection for demonstration
    pred_dates = [end_date + timedelta(days=i) for i in range(1, 31)]
    last_price = hist["Close"].iloc[-1]
    
    # Basic linear trend projection (can be replaced with more sophisticated prediction models)
    pred_values = [last_price + (i * (hist["Close"].std() / 30)) for i in range(1, 31)]

    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist["Close"], label="الحركة الفعلية")
    plt.plot(pred_dates, pred_values, label="الحركة المتوقعة", linestyle="--")
    plt.title(f"حركة سهم {company}")
    plt.xlabel("التاريخ")
    plt.ylabel("السعر")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("stock_plot.png")
    plt.close()

    return "stock_plot.png"

def update_companies(market):
    """
    Update company dropdown based on selected market.
    
    Args:
        market (str): Selected market
    
    Returns:
        gr.Dropdown: Updated dropdown with companies for the market
    """
    companies = get_companies(market)
    return gr.Dropdown.update(choices=companies)

def display_plot(market, company):
    """
    Display stock plot for selected market and company.
    
    Args:
        market (str): Selected market
        company (str): Selected company
    
    Returns:
        str: Path to the generated plot image
    """
    return fetch_stock_data(market, company)

def create_stock_prediction_interface():
    """
    Create Gradio interface for stock market prediction.
    
    Returns:
        gr.Blocks: Gradio interface
    """
    with gr.Blocks() as demo:
        gr.Markdown("# توقع حركة الأسهم عن طريق تحليلها بالذكاء الاصطناعي")
        
        with gr.Row():
            market_dropdown = gr.Dropdown(label="اختر السوق", choices=list(markets.keys()))
            company_dropdown = gr.Dropdown(label="اختر الشركة", choices=[])
        
        plot_output = gr.Image(label="حركة السهم")
        
        market_dropdown.change(fn=update_companies, inputs=market_dropdown, outputs=company_dropdown)
        company_dropdown.change(fn=display_plot, inputs=[market_dropdown, company_dropdown], outputs=plot_output)
    
    return demo

# Main execution
if __name__ == "__main__":
    demo = create_stock_prediction_interface()
    demo.launch()