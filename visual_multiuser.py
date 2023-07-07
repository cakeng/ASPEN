import pandas as pd
import matplotlib.pyplot as plt

NUM_DEVICE=4

ROW_FROM=0
ROW_TO=8505

sync = 0

max_timestamp = 0
min_timestamp = 0

pip_data_rx = [pd.read_csv(f'./logs/multiuser/pipeline_dev{i}_RX.txt') for i in range(1, NUM_DEVICE)]
pip_data_tx = [pd.read_csv(f'./logs/multiuser/pipeline_dev{i}_TX.txt') for i in range(1, NUM_DEVICE)]

transfer_count = 0
for dev in range(1, NUM_DEVICE):
    rx_log = pip_data_rx[dev-1]
    tx_log = pip_data_tx[dev-1]
    log_length = len(rx_log)
    for i in range(log_length):
        if tx_log["sent time (ms)"].iloc[i] != 0:
            plt.plot([tx_log["sent time (ms)"].iloc[i], rx_log["received time (ms)"].iloc[i]], [transfer_count, transfer_count], linestyle='-', marker='.', color='dimgray')
            transfer_count += 1
        if tx_log["received time (ms)"].iloc[i] != 0:
            plt.plot([tx_log["received time (ms)"].iloc[i], rx_log["sent time (ms)"].iloc[i]], [transfer_count, transfer_count], linestyle='-', marker='.', color='lightgray')
            transfer_count += 1
    
    plt.scatter(rx_log["computed time (ms)"], [transfer_count for _ in range(len(rx_log))])
    plt.scatter(tx_log["computed time (ms)"], [transfer_count for _ in range(len(tx_log))])
    transfer_count += 5

max_timestamp = max(max_timestamp, *pip_data_rx[0]['computed time (ms)'], *pip_data_rx[1]['computed time (ms)'], *pip_data_rx[2]['computed time (ms)'])
plt.xlim([min_timestamp, max_timestamp+50])

plt.show()