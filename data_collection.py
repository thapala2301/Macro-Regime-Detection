# We want to pull macro indicators from FRED
# We want to pull asset returns from yfinance
# The aim is to build a monthly dataset covering 20 years of macro conditions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
import yfinance as yf

FRED_API_KEY = '388d880cf9778a87a3bc760c1f771a80'
fred = Fred(api_key = FRED_API_KEY)

print("Connected to FRED API")

# After connecting to the FRED API, we want to pull our macro indicators

print("Pulling macro indicitaors from FRED...")

# Fed fund rates - interest rates set by the Fed Reserve
# High - Tight Monetary Policy, Low - Loose Monetary Policy

fed_funds = fred.get_series('FEDFUNDS', observation_start='2000-01-01')

# CPI Inflation Rate
# High - Inflationary, Low - Deflationary

cpi = fred.get_series('CPIAUCSL', observation_start='2000-01-01')
inflation = cpi.pct_change(12, fill_method=None) * 100

# Unemployment Rates
# High - Recession, Low - Expansion

unemployment = fred.get_series('UNRATE', observation_start= '2000-01-01')

# Yield Curve Slope (Difference between interest rates on long and short term bonds)
# Positive- Nomal, Negative - Inverted = Recession Warning
# A very reliable recessin predictor

treasury_10y = fred.get_series('GS10', observation_start= '2000-01-01')
treasury_2y = fred.get_series('GS2', observation_start= '2000-01-01')
yield_curve = treasury_10y - treasury_2y

# Credit spread: BAA corporate bond yield minus 10-year Treasury
# High spread = credit stress, Low spread = risk-on environment
baa = fred.get_series('BAA', observation_start='2000-01-01')
credit_spread = baa - treasury_10y

print("Macro indicators pulled successfully")

# Now we have macro indicators to work with, we need our asset returns from yfinance
# We are going to target the major investable asset classses
# In short, after extracting hidden regimes from macro data, we want to know which assets perform best in each regime.

print ("Printing asset data from yfinance")

tickers = {
    'SPY': 'S&P 500 (Equities)',
    'TLT': '20-Year Treasury (Bonds)',
    'GLD': 'Gold',
    'DJP': 'Commodities'
}

asset_returns = {}

for ticker, name in tickers.items():
    data = yf.download(ticker, start='2000-01-01', progress=False)
    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    monthly = data['Close'].resample('MS').last()
    returns = monthly.pct_change() * 100
    asset_returns[ticker] = returns
    print(f"  {name}: {len(returns)} monthly observations")

asset_df = pd.DataFrame(asset_returns)

# Now we sample all to ensure an aligned monthly frequency

macro_df = pd.DataFrame({
    'fed_funds': fed_funds.resample('MS').last(),
    'inflation': inflation.resample('MS').last(),
    'unemployment': unemployment.resample('MS').last(),
    'yield_curve': yield_curve.resample('MS').last(),
    'credit_spread': credit_spread.resample('MS').last()

})

df = pd.concat([macro_df, asset_df], axis = 1)
df = df.dropna()

print(f"Final dataset: {len(df)} monthly observations")
print(f"Date range: {df.index[0].strftime('%b %Y')} to {df.index[-1].strftime('%b %Y')}")
print(df.head())
print("=== BASIC STATISTICS ===")
print(df.describe().round(2))

df.to_csv('macro_data.csv')
print('Data moved to macro_data.csv')

print(df[['fed_funds','inflation','unemployment','yield_curve','credit_spread']].corr().round(2))
print()
print("Worst single months:")
print(df[['SPY','TLT','GLD','DJP']].idxmin())
print()
print("Best single months:")
print(df[['SPY','TLT','GLD','DJP']].idxmax())
print()
print("Months where ALL assets negative:")
all_negative = df[(df['SPY']<0) & (df['TLT']<0) & (df['GLD']<0) & (df['DJP']<0)]
print(f"{len(all_negative)} months where every asset fell simultaneously")
print(all_negative[['SPY','TLT','GLD','DJP']].round(2))

# Now we visualize the macro indicators over time

fig, axes = plt.subplots(5,1,figsize=(14,16))

indicators = [
    ('fed_funds', 'Fed Funds Rate (%)', 'steelblue'),
    ('inflation', 'CPI Inflation YoY (%)', 'coral'),
    ('unemployment', 'Unemployment Rate (%)', 'orange'),
    ('yield_curve', 'Yield Curve Slope (10Y-2Y)', 'green'),
    ('credit_spread', 'Credit Spread (BAA-10Y)', 'purple')
]

for i, (col, label, color) in enumerate(indicators):
    axes[i].plot(df.index, df[col], color=color, linewidth=1.2)
    axes[i].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    axes[i].set_ylabel(label, fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].fill_between(df.index, df[col], 0,
                         where=(df[col] < 0),
                         color='red', alpha=0.15)

axes[0].set_title('Macroeconomic Indicators 2006-2026', fontsize=13)
plt.tight_layout()
plt.savefig('macro_indicators.png', dpi=150)
plt.show()
print("Chart saved as macro_indicators.png")

