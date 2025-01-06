# import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and configure FEC and CS timestamp data as dataframe
df_fec = pd.read_csv("../FEC/mouse_0_fec.csv")  # df containing eyeblink fraction info
df_cs = pd.read_csv("../stim/mouse_0_stim.csv")  # df containing timestamp of CS for each trial

# Merge the dataframes based on the 'Trial #' column
merged_df = pd.merge(df_fec, df_cs, on='Trial #', how='left')
merged_df['Trial #'] = merged_df['Trial #'].astype(int)  # convert trial # to integer value
merged_df.head()

# Remove timestamps before the CS onset
filter = (merged_df['Current Timestamp'] >= merged_df['CS Timestamp'])
df = merged_df[filter]

# Recreate graph of "FEC vs. Time from CS onset" from the paper
plt.figure(figsize=(8, 5))
plt.scatter(df['Current Timestamp'], df['FEC'], color='red', label='Scatter Plot')

plt.xlabel('Time from CS Onset')
plt.ylabel('Fraction Eyelid Closure')
plt.title('FEC vs Time from CS Onset')
plt.legend()
plt.grid(True)
plt.show()