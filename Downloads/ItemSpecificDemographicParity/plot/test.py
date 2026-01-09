import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('log/tuning_results_FairPMF_seed2053.csv')
plt.plot(df['fair_lambda'], df['best_hr'], label='HR@10')
plt.plot(df['fair_lambda'], df['avg_value'], label='Value Unfairness')
plt.xscale('log')
plt.xlabel('fair_lambda')
plt.legend()
plt.show()