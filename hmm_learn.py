import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

df = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)
features = ['fed_funds', 'inflation', 'unemployment', 'yield_curve', 'credit_spread']
X_raw = df[features].values

scaler = StandardScaler() # Creating an instance from the class
X = scaler.fit_transform(X_raw)

model = GaussianHMM(n_components = 4, covariance_type = 'full', n_iter = 1000, random_state=42)
model.fit(X)

regimes = model.predict(X)
print(regimes)

plt.figure(figsize=(14, 4))
plt.plot(df.index, regimes, color='steelblue', linewidth=1)
plt.title('Macro Regime Sequence 2006-2026')
plt.ylabel('Regime')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regime_sequence.png', dpi=150)
plt.show()

# Better visualisation with colour coded regimes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# Define colours for each regime
colours = {0: 'red', 1: 'orange', 2: 'green', 3: 'steelblue'}
labels = {0: 'Regime 0', 1: 'Regime 1', 2: 'Regime 2', 3: 'Regime 3'}

# Chart 1 — regime sequence with colour shading
for regime in range(4):
    mask = regimes == regime
    ax1.fill_between(df.index, 0, 1,
                     where=mask,
                     color=colours[regime],
                     alpha=0.4,
                     label=labels[regime])

ax1.set_title('Macro Regime Sequence 2006-2026', fontsize=13)
ax1.set_ylabel('Regime')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Chart 2 — unemployment over time with regime shading
ax2.plot(df.index, df['unemployment'], color='black', linewidth=1.2)
for regime in range(4):
    mask = regimes == regime
    ax2.fill_between(df.index, 0, df['unemployment'].max(),
                     where=mask,
                     color=colours[regime],
                     alpha=0.15)

ax2.set_title('Unemployment Rate with Regime Overlay', fontsize=13)
ax2.set_ylabel('Unemployment %')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regime_overlay.png', dpi=150)
plt.show()
print("Chart saved as regime_overlay.png")

df['regime'] = regimes
regime_means = df.groupby('regime')[features].mean()
print(regime_means)

regime_names = {
    0: 'Inflationary Shock',
    1: 'Goldilocks',
    2: 'Pre-Crisis Normal',
    3: 'Crisis & Recovery'
}

print("\nRegime characterisation:")
for regime, name in regime_names.items():
    print(f"\nRegime {regime} — {name}")
    print(regime_means.loc[regime].round(2))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print("\nTransition Matrix:")
print(pd.DataFrame(
    model.transmat_.round(3),
    columns=[regime_names[i] for i in range(4)],
    index=[regime_names[i] for i in range(4)]
))