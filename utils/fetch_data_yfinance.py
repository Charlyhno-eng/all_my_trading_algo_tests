import os
import yfinance as yf

ticker = "BTC-USD"
start_date = "2021-01-01"
end_date = "2025-08-31"

os.makedirs("data/data_ohlc", exist_ok=True)
save_data_filename = f"data/data_ohlc/{ticker}_yfinance_{start_date}_{end_date}.csv"

df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
df = df.reset_index()
df = df.rename(columns={
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

df.to_csv(save_data_filename, index=False)
print("saved rows:", len(df))
print(df.head())
