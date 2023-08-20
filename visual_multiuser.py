import math
import pandas as pd
import matplotlib.pyplot as plt

NUM_DEVICE=1
NUM_ITERATIONS=10

DNN="vgg16"
BATCH=1
NUM_TILES=100

VISUAL_ALPHA=0.008


edge_dir="./logs/visual/multiuser"
server_dir="./logs/visual/multiuser"
edge_file_name=""
server_file_name=""

for iter in range(0, NUM_ITERATIONS):
    dynamic_edge_data = [pd.read_csv(f'{edge_dir}/edge_{edge_id}/{DNN}_dynamic_EDGE_{DNN}_B{BATCH}_T{NUM_TILES}_Iter{iter}.csv') for edge_id in range(0, NUM_DEVICE)]
    dynamic_server_data = [pd.read_csv(f'{server_dir}/edge_{edge_id}/{DNN}_dynamic_SERVER_{DNN}_B{BATCH}_T{NUM_TILES}_Iter{iter}.csv') for edge_id in range(0, NUM_DEVICE)]

    spinn_edge_data = [pd.read_csv(f'{edge_dir}/edge_{edge_id}/{DNN}_spinn_EDGE_{DNN}_B{BATCH}_T{NUM_TILES}_Iter{iter}.csv') for edge_id in range(0, NUM_DEVICE)]
    spinn_server_data = [pd.read_csv(f'{server_dir}/edge_{edge_id}/{DNN}_spinn_SERVER_{DNN}_B{BATCH}_T{NUM_TILES}_Iter{iter}.csv') for edge_id in range(0, NUM_DEVICE)]

    transfer_count = 0
    plt.figure(figsize=(10,10))
    plt.subplot(2, 1, 1)

    sent_legend_flag = 0
    recv_legend_flag = 0
    
    for edge_id in range(0, NUM_DEVICE):
        edge_log = dynamic_edge_data[edge_id]
        server_log = dynamic_server_data[edge_id]
        
        log_length = len(server_log)
        min_edge_computed_time = edge_log['computed time (ms)'][edge_log['computed time (ms)'] > 0].min()
        min_edge_sent_time = edge_log['sent time (ms)'][edge_log['sent time (ms)'] > 0].min()
        min_edge_received_time = edge_log['received time (ms)'][edge_log['received time (ms)'] > 0].min()

        values = [value for value in [min_edge_computed_time, min_edge_sent_time, min_edge_received_time] if not math.isnan(value)]

        if values:
            min_timestamp = min(values)

        dynamic_max_timestamp = max(edge_log['computed time (ms)'].max(), edge_log['sent time (ms)'].max(), edge_log['received time (ms)'].max()) - min_timestamp

        for i in range(log_length):
            if edge_log["sent time (ms)"].iloc[i] != 0 and server_log["received time (ms)"].iloc[i] != 0:
                if sent_legend_flag == 0:
                    plt.plot([edge_log["sent time (ms)"].iloc[i] - min_timestamp, server_log["received time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='green', label="Mobile -> Server")    
                    sent_legend_flag = 1
                else:
                    plt.plot([edge_log["sent time (ms)"].iloc[i] - min_timestamp, server_log["received time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='green')    
                transfer_count += 1
            if edge_log["received time (ms)"].iloc[i] != 0 and server_log["sent time (ms)"].iloc[i] != 0:
                if recv_legend_flag == 0:
                    plt.plot([edge_log["received time (ms)"].iloc[i] - min_timestamp, server_log["sent time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='purple', label="Server -> Mobile")
                    recv_legend_flag = 1
                else:
                    plt.plot([edge_log["received time (ms)"].iloc[i] - min_timestamp, server_log["sent time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='purple')
                
                transfer_count += 1
        
        nonzero_server_computed_time = server_log["computed time (ms)"][server_log['computed time (ms)'] > 0]
        nonzero_edge_computed_time = edge_log["computed time (ms)"][edge_log['computed time (ms)'] > 0]
        plt.scatter(nonzero_server_computed_time - min_timestamp, [transfer_count for _ in range(len(nonzero_server_computed_time))], color='red', alpha=VISUAL_ALPHA, linewidths=0, label="Server Computation")
        plt.scatter(nonzero_edge_computed_time - min_timestamp, [transfer_count for _ in range(len(nonzero_edge_computed_time))], color='blue', alpha=VISUAL_ALPHA, linewidths=0, label="Mobile Computation")
        plt.legend()
        # plt.scatter(server_log["computed time (ms)"], [-5 for _ in range(len(server_log))], alpha=VISUAL_ALPHA, linewidths=0, color='black')
        # transfer_count += 5
    
    sent_legend_flag = 0
    recv_legend_flag = 0
    plt.subplot(2, 1, 2)
    for edge_id in range(0, NUM_DEVICE):
        edge_log = spinn_edge_data[edge_id]
        server_log = spinn_server_data[edge_id]
        
        log_length = len(server_log)

        min_edge_computed_time = edge_log['computed time (ms)'][edge_log['computed time (ms)'] > 0].min()
        min_edge_sent_time = edge_log['sent time (ms)'][edge_log['sent time (ms)'] > 0].min()
        min_edge_received_time = edge_log['received time (ms)'][edge_log['received time (ms)'] > 0].min()

        values = [value for value in [min_edge_computed_time, min_edge_sent_time, min_edge_received_time] if not math.isnan(value)]

        if values:
            min_timestamp = min(values)

        spinn_max_timestamp = max(edge_log['computed time (ms)'].max(), edge_log['sent time (ms)'].max(), edge_log['received time (ms)'].max()) - min_timestamp

        for i in range(log_length):
            if edge_log["sent time (ms)"].iloc[i] != 0 and server_log["received time (ms)"].iloc[i] != 0:
                if sent_legend_flag == 0:
                    plt.plot([edge_log["sent time (ms)"].iloc[i] - min_timestamp, server_log["received time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='green', label="Mobile -> Server")    
                    sent_legend_flag = 1
                else:
                    plt.plot([edge_log["sent time (ms)"].iloc[i] - min_timestamp, server_log["received time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='green')    
                transfer_count += 1
            if edge_log["received time (ms)"].iloc[i] != 0 and server_log["sent time (ms)"].iloc[i] != 0:
                if recv_legend_flag == 0:
                    plt.plot([edge_log["received time (ms)"].iloc[i] - min_timestamp, server_log["sent time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='purple', label="Server -> Mobile")
                    recv_legend_flag = 1
                else:
                    plt.plot([edge_log["received time (ms)"].iloc[i] - min_timestamp, server_log["sent time (ms)"].iloc[i] - min_timestamp], \
                         [transfer_count, transfer_count], linestyle='-', marker='.', color='purple')
                
                transfer_count += 1
        
        nonzero_server_computed_time = server_log["computed time (ms)"][server_log['computed time (ms)'] > 0]
        nonzero_edge_computed_time = edge_log["computed time (ms)"][edge_log['computed time (ms)'] > 0]
        plt.scatter(nonzero_server_computed_time - min_timestamp, [transfer_count for _ in range(len(nonzero_server_computed_time))], color='red', alpha=VISUAL_ALPHA, linewidths=0, label="Server Computation")
        plt.scatter(nonzero_edge_computed_time - min_timestamp, [transfer_count for _ in range(len(nonzero_edge_computed_time))], color='blue', alpha=VISUAL_ALPHA, linewidths=0, label="Mobile Computation")
        plt.legend()
    
    max_timestamp = max(dynamic_max_timestamp, spinn_max_timestamp)
    plt.subplot(2, 1, 1)
    # plt.xlim((0, max_timestamp+10))
    plt.xlabel("Time (ms)")
    # plt.xticks(range(0, int(max_timestamp+10), 20))
    plt.subplot(2, 1, 2)
    # plt.xlim((0, max_timestamp+10))
    plt.xlabel("Time (ms)")
    # plt.xticks(range(0, int(max_timestamp+10), 20))
    plt.show()


