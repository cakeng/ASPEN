import pandas as pd
import matplotlib.pyplot as plt

ROW_FROM=0
ROW_TO=127

max_timestamp = 0

plt.subplot(2, 1, 1)

pip_data_rx = pd.read_csv('./logs/pipeline_ninst_time_logs_RX.csv')
pip_data_tx = pd.read_csv('./logs/pipeline_ninst_time_logs_TX.csv')

input_pip_data_rx = pip_data_rx.iloc[ROW_FROM:ROW_TO]
input_pip_data_tx = pip_data_tx.iloc[ROW_FROM:ROW_TO]
compute_data_rx = pip_data_rx.iloc[ROW_TO:]

max_timestamp = max(max_timestamp, *compute_data_rx['computed time (ms)'])

for i in range(ROW_FROM, ROW_TO):
    plt.plot([input_pip_data_tx["sent time (ms)"].iloc[i], input_pip_data_rx["received time (ms)"].iloc[i]], [i, i], linestyle='-', marker='.', color='dimgray')
    plt.xlabel('timestamp (ms)')
    plt.ylabel('ninst idx')

plt.scatter(compute_data_rx["computed time (ms)"], [128 for _ in range(len(compute_data_rx))])

plt.xlim([0, 300])


plt.subplot(2, 1, 2)

seq_data_rx = pd.read_csv('./logs/sequential_ninst_time_logs_RX.csv')
seq_data_tx = pd.read_csv('./logs/sequential_ninst_time_logs_TX.csv')

input_seq_data_rx = seq_data_rx.iloc[ROW_FROM:ROW_TO]
input_seq_data_tx = seq_data_tx.iloc[ROW_FROM:ROW_TO]
compute_data_rx = seq_data_rx.iloc[ROW_TO:]

max_timestamp = max(max_timestamp, *compute_data_rx['computed time (ms)'])

for i in range(ROW_FROM, ROW_TO):
    plt.plot([input_seq_data_tx["sent time (ms)"].iloc[i], input_seq_data_rx["received time (ms)"].iloc[i]], [i, i], linestyle='-', marker='.', color='dimgray')
    plt.xlabel('timestamp (ms)')
    plt.ylabel('ninst idx')

plt.scatter(compute_data_rx["computed time (ms)"], [128 for _ in range(len(compute_data_rx))])

plt.xlim([0, max_timestamp+50])

plt.show()