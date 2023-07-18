import pandas as pd
import matplotlib.pyplot as plt

NUM_DEVICE=2

ROW_FROM=0
ROW_TO=8505

VISUAL_ALPHA=0.008

plt.subplot(2, 1, 1)

sync = 0

max_timestamp = 0
min_timestamp = 0

# pip_data_rx = [pd.read_csv(f'./logs/multiuser/pip_dev{i}_RX.txt') for i in range(1, NUM_DEVICE)]
# pip_data_tx = [pd.read_csv(f'./logs/multiuser/pip_dev{i}_TX.txt') for i in range(1, NUM_DEVICE)]
pip_data_rx = [pd.read_csv(f'./logs/temp/temp_pip_dev0_0.csv')]
pip_data_tx = [pd.read_csv(f'./logs/temp/temp_pip_dev1_0.csv')]

sync_arr = [8.118, 29.194, 59.601]

transfer_count = 0
for dev in range(1, NUM_DEVICE):
    rx_log = pip_data_rx[dev-1]
    tx_log = pip_data_tx[dev-1]
    sync = -sync_arr[dev-1]
    sync = 0
    log_length = len(rx_log)
    for i in range(log_length):
        if tx_log["sent time (ms)"].iloc[i] != 0:
            plt.plot([tx_log["sent time (ms)"].iloc[i] + sync, rx_log["received time (ms)"].iloc[i]], [transfer_count, transfer_count], linestyle='-', marker='.', color='dimgray')
            transfer_count += 1
        if tx_log["received time (ms)"].iloc[i] != 0:
            plt.plot([tx_log["received time (ms)"].iloc[i] + sync, rx_log["sent time (ms)"].iloc[i]], [transfer_count, transfer_count], linestyle='-', marker='.', color='lightgray')
            transfer_count += 1
    
    plt.scatter(rx_log["computed time (ms)"], [transfer_count for _ in range(len(rx_log))], alpha=VISUAL_ALPHA, linewidths=0)
    plt.scatter(tx_log["computed time (ms)"] + sync, [transfer_count for _ in range(len(tx_log))], alpha=VISUAL_ALPHA, linewidths=0)

    plt.scatter(rx_log["computed time (ms)"], [-5 for _ in range(len(rx_log))], alpha=VISUAL_ALPHA, linewidths=0, color='black')
    transfer_count += 5

max_timestamp = max(max_timestamp, *pip_data_rx[0]['computed time (ms)'])

plt.subplot(2, 1, 2)
# seq_data_rx = [pd.read_csv(f'./logs/multiuser/seq_dev{i}_RX.txt') for i in range(1, NUM_DEVICE)]
# seq_data_tx = [pd.read_csv(f'./logs/multiuser/seq_dev{i}_TX.txt') for i in range(1, NUM_DEVICE)]
seq_data_rx = [pd.read_csv(f'./logs/temp/temp_seq_dev0_0.csv')]
seq_data_tx = [pd.read_csv(f'./logs/temp/temp_seq_dev1_0.csv')]

sync_arr = [0]

transfer_count = 0
for dev in range(1, NUM_DEVICE):
    rx_log = seq_data_rx[dev-1]
    tx_log = seq_data_tx[dev-1]
    sync = -sync_arr[dev-1]
    sync = 0
    log_length = len(rx_log)
    for i in range(log_length):
        if tx_log["sent time (ms)"].iloc[i] != 0:
            plt.plot([tx_log["sent time (ms)"].iloc[i] + sync, rx_log["received time (ms)"].iloc[i]], [transfer_count, transfer_count], linestyle='-', marker='.', color='dimgray')
            transfer_count += 1
        if tx_log["received time (ms)"].iloc[i] != 0:
            plt.plot([tx_log["received time (ms)"].iloc[i] + sync, rx_log["sent time (ms)"].iloc[i]], [transfer_count, transfer_count], linestyle='-', marker='.', color='lightgray')
            transfer_count += 1
    
    plt.scatter(rx_log["computed time (ms)"], [transfer_count for _ in range(len(rx_log))], alpha=VISUAL_ALPHA, linewidths=0)
    plt.scatter(tx_log["computed time (ms)"] + sync, [transfer_count for _ in range(len(tx_log))], alpha=VISUAL_ALPHA, linewidths=0)

    plt.scatter(rx_log["computed time (ms)"], [-5 for _ in range(len(rx_log))], alpha=VISUAL_ALPHA, linewidths=0, color='black')
    transfer_count += 5

max_timestamp = max(max_timestamp, *seq_data_rx[0]['computed time (ms)'])

plt.subplot(2, 1, 1)
plt.xlim([min_timestamp, max_timestamp+50+max(sync_arr)])
plt.subplot(2, 1, 2)
plt.xlim([min_timestamp, max_timestamp+50+max(sync_arr)])

plt.show()