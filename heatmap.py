# heatmap drawing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# calplot
import calplot
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


# read the data
df = pd.read_excel('Energy Heatmap.xlsx')

# clean the first column
df = df.iloc[:, 1:]

# set index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

for date in df.columns:
    df[date] = df[date].astype('float')

for date in df.columns:
    print(date, type(date))
    calplot.calplot(df[date], cmap='YlOrBr', figsize=(10, 10))
    plt.suptitle(date)
    plt.savefig(date)
    plt.show()
