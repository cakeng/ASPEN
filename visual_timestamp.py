import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

COL_COMPUTE = "computed time (ms)"
COL_SEND = "sent time (ms)"
COL_RECV = "received time (ms)"

max_timestamp = 0

plt.subplot(2, 1, 1)

pip_data_rx = pd.read_csv('./logs/pipeline_ninst_time_logs_RX.csv')
pip_data_tx = pd.read_csv('./logs/pipeline_ninst_time_logs_TX.csv')

pip_data_rx.columns = ['idx', 'rx_comp', 'rx_recv', 'rx_send']
pip_data_tx.columns = ['idx_', 'tx_comp', 'tx_recv', 'tx_send']

pip_data = pd.concat([pip_data_rx, pip_data_tx], axis=1).drop('idx_', axis=1)

seq_data_rx = pd.read_csv('./logs/sequential_ninst_time_logs_RX.csv')
seq_data_tx = pd.read_csv('./logs/sequential_ninst_time_logs_TX.csv')

seq_data_rx.columns = ['idx', 'rx_comp', 'rx_recv', 'rx_send']
seq_data_tx.columns = ['idx_', 'tx_comp', 'tx_recv', 'tx_send']

seq_data = pd.concat([seq_data_rx, seq_data_tx], axis=1).drop('idx_', axis=1)

local_data = pd.read_csv('./logs/local_logs.csv')
local_finish = local_data[COL_COMPUTE].max()
print(local_finish)

max_timestamp = max(*pip_data.drop(['idx'], axis=1).max(), *seq_data.drop(['idx'], axis=1).max())


comp_data_rx = pip_data[pip_data['rx_comp'] != 0.0]
comp_data_tx = pip_data[pip_data['tx_comp'] != 0.0]

send_data_rx = pip_data[pip_data['rx_send'] != 0.0]
send_data_tx = pip_data[pip_data['tx_send'] != 0.0]

i = 0
for row in send_data_tx.iloc:
    plt.plot(row[['tx_send', 'rx_recv']], [i, i], linestyle='-', marker='.', color='dimgray')
    i += 1
for row in send_data_rx.iloc:
    plt.plot(row[['rx_send', 'tx_recv']], [i, i], linestyle='-', marker='.', color='dimgray')
    i += 1


plt.scatter(list(comp_data_rx['rx_comp'].iloc), [i for _ in range(len(list(comp_data_rx['rx_comp'].iloc)))], color='blue')
plt.scatter(list(comp_data_tx['tx_comp'].iloc), [i for _ in range(len(list(comp_data_tx['tx_comp'].iloc)))], color='orange')


plt.xlim([0, max_timestamp+50])


plt.subplot(2, 1, 2)

comp_data_rx = seq_data[seq_data['rx_comp'] != 0.0]
comp_data_tx = seq_data[seq_data['tx_comp'] != 0.0]

send_data_rx = seq_data[seq_data['rx_send'] != 0.0]
send_data_tx = seq_data[seq_data['tx_send'] != 0.0]

i = 0
for row in send_data_tx.iloc:
    plt.plot(row[['tx_send', 'rx_recv']], [i, i], linestyle='-', marker='.', color='dimgray')
    i += 1
for row in send_data_rx.iloc:
    plt.plot(row[['rx_send', 'tx_recv']], [i, i], linestyle='-', marker='.', color='dimgray')
    i += 1


plt.scatter(list(comp_data_rx['rx_comp'].iloc), [i for _ in range(len(list(comp_data_rx['rx_comp'].iloc)))], color='blue')
plt.scatter(list(comp_data_tx['tx_comp'].iloc), [i for _ in range(len(list(comp_data_tx['tx_comp'].iloc)))], color='orange')

plt.xlim([0, max_timestamp+50])

plt.show()

