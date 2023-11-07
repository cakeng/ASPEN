#!/usr/bin/env python3
from heft.core import (wbar, cbar, ranku, schedule, Event, start_time,
        makespan, endtime, insert_recvs, insert_sends, insert_sendrecvs, recvs,
        sends)
import sys
import numpy as np

"""
This is a simple script to use the HEFT function provided based on the example given in the original HEFT paper.
You have to define the DAG, compcost function and commcost funtion.

Each task/job is numbered 1 to 10
Each processor/agent is named 'a', 'b' and 'c'

Output expected:
Ranking:
[10, 8, 7, 9, 6, 5, 2, 4, 3, 1]
Schedule:
('a', [Event(job=2, start=27, end=40), Event(job=8, start=57, end=62)])
('b', [Event(job=4, start=18, end=26), Event(job=6, start=26, end=42), Event(job=9, start=56, end=68), Event(job=10, start=73, end=80)])
('c', [Event(job=1, start=0, end=9), Event(job=3, start=9, end=28), Event(job=5, start=28, end=38), Event(job=7, start=38, end=49)])
{1: 'c', 2: 'a', 3: 'c', 4: 'b', 5: 'c', 6: 'b', 7: 'c', 8: 'a', 9: 'b', 10: 'b'}
"""
dag = {}
computation_costs = []
communication_costs = []

def parse_heft_data (filepath):
    with open(filepath) as f:
        lines = f.readlines()
        number_of_tasks = int(lines[0])
        number_of_processors = int(lines[1])
        computation_costs = np.zeros((number_of_processors, number_of_tasks), dtype=int)
        communication_costs = np.zeros((number_of_tasks, number_of_tasks), dtype=int)
        dag = {}
        for i in range (number_of_processors):
            string_input = lines[2+i].split(' ')
            for j in range (number_of_tasks):
                computation_costs[i][j] = int(string_input[j])
        for k in range (number_of_tasks):
            string_input = lines[2+number_of_processors+k].split(' ')
            child_list = ()
            for l in range (number_of_tasks):
                communication_costs[k][l] = int(string_input[l])
                if (communication_costs[k][l] != 0):
                    child_list = child_list + (l,)
            dag.update({k:child_list})
    return number_of_tasks, number_of_processors, dag, computation_costs, communication_costs

def compcost(job, agent):
    return computation_costs[agent][job]

def commcost(ni, nj, A, B):
    if(A==B):
        return 0
    else:
        return communication_costs[ni][nj]
    
def export_schedule (filename, orders, jobson):
    with open(filename, 'w') as f:
        for j in jobson:
            f.write(str(j) + " " + str(jobson[j]) + "\n")

if __name__ == "__main__":
    filepath = sys.argv[1]
    number_of_tasks, number_of_processors, dag, computation_costs, communication_costs = parse_heft_data(filepath)
    sys.setrecursionlimit(10**6)
    orders, jobson = schedule(dag, range(number_of_processors), compcost, commcost, 32, 2009)
    for eachP in sorted(orders):
        print("Device ", eachP, orders[eachP])
    export_schedule (filepath + ".schedule", orders, jobson)