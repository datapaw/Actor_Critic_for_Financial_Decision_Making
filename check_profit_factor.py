import pandas as pd
import numpy as np

# Load data
sp = pd.read_csv('data/dataset/sp500_prep.csv')
pred = pd.read_csv('results/chunked_main_votes.csv')

# Get test set
test_sp = sp.tail(941).reset_index(drop=True)
test_pred = pred.tail(941).reset_index(drop=True)

CLASS_TO_HORIZON = {0: 10, 1: 15, 2: 30, 3: 60}

# Calculate returns
returns = []
for i in range(len(test_sp)):
    pred_class = test_pred.loc[i, 'ensemble_vote_pred']
    horizon = CLASS_TO_HORIZON[pred_class]
    forward_ret_per_day = test_sp.loc[i, f'forward_return_{horizon}']
    close = test_sp.loc[i, 'Close']
    
    total_points = forward_ret_per_day * horizon
    return_pct = (total_points / close) * 100
    returns.append(return_pct)

returns = np.array(returns)
valid = returns[~np.isnan(returns)]

# Split into wins and losses
wins = valid[valid > 0]
losses = valid[valid <= 0]

# Calculate totals
total_gains = np.sum(wins)
total_losses = abs(np.sum(losses))

print("="*80)
print("PROFIT FACTOR VERIFICATION")
print("="*80)
print(f"\nWinning trades: {len(wins)}")
print(f"Losing trades: {len(losses)}")
print(f"\nTotal gains: {total_gains:.2f}%")
print(f"Total losses: {total_losses:.2f}%")
print(f"Net profit: {total_gains - total_losses:.2f}%")

print("\n" + "-"*80)
print("PROFIT FACTOR DEFINITIONS")
print("-"*80)

# Definition 1: Gross Profit Factor (standard)
pf_gross = total_gains / total_losses
print(f"\n1. Gross Profit Factor = Total Gains / Total Losses")
print(f"   = {total_gains:.2f} / {total_losses:.2f}")
print(f"   = {pf_gross:.4f}")
print(f"   (This means you make ${pf_gross:.2f} for every $1 you lose)")

# Definition 2: Net Profit Factor
pf_net = (total_gains - total_losses) / total_losses
print(f"\n2. Net Profit Factor = (Total Gains - Total Losses) / Total Losses")
print(f"   = ({total_gains:.2f} - {total_losses:.2f}) / {total_losses:.2f}")
print(f"   = {pf_net:.4f}")
print(f"   (This is your net return as a ratio of losses)")

print(f"\n" + "-"*80)
print(f"Difference: {pf_gross:.4f} - {pf_net:.4f} = {pf_gross - pf_net:.4f}")
print(f"\nThe standard definition in trading is Gross Profit Factor (Definition 1)")
print(f"But some systems report it as Net Profit Factor (Definition 2)")
print(f"\nCurrent output shows: 3.1545")
print(f"If you want net profit factor, it should be: {pf_net:.4f}")
