# utils.py
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

os.makedirs("runs", exist_ok=True)

run_id = 0

def simulate_graph(params):
    time, Lchain, Nchain, trials, dose_rate = [int(p) if i != 4 else float(p) for i, p in enumerate(params)]
    global run_id
    run_id += 1

    avg_sec, avg_conn, graphs = 0.0, 0.0, []
    for t in range(trials):
        G = nx.barabasi_albert_graph(Lchain * Nchain, m=2)
        graphs.append(G)
        L = nx.laplacian_matrix(G).todense()
        eigvals = np.linalg.eigvalsh(L)
        sec_smallest = eigvals[1] if len(eigvals) > 1 else 0.0
        avg_sec += sec_smallest
        avg_conn += np.mean([deg for _, deg in G.degree()])

    avg_sec /= trials
    avg_conn /= trials

    reward = avg_sec  # or some function of avg_sec, conn
    next_state = np.random.uniform(low=0.1, high=1.0, size=5)

    result = {
        "run_id": run_id,
        "time": time,
        "Lchain": Lchain,
        "Nchain": Nchain,
        "trials": trials,
        "dose_rate": dose_rate,
        "avg_sec": avg_sec,
        "avg_conn": avg_conn,
        "graphs": graphs
    }

    return result, reward, next_state

def save_csv_data(result):
    filename = f"runs/Days{result['time']}_LChain{result['Lchain']}_Nch{result['Nchain']}.csv"
    data = {
        "Run": [result['run_id']],
        "Time": [result['time']],
        "LChain": [result['Lchain']],
        "NChain": [result['Nchain']],
        "DoseRate": [result['dose_rate']],
        "Trials": [result['trials']],
        "AvgConn": [result['avg_conn']],
        "AvgSecEig": [result['avg_sec']]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def make_graphs(result):
    fig, axs = plt.subplots(1, len(result['graphs']), figsize=(15, 5))
    if len(result['graphs']) == 1:
        axs = [axs]
    for ax, G in zip(axs, result['graphs']):
        nx.draw(G, ax=ax, node_size=10)
    plt.savefig(f"runs/graph_run{result['run_id']}.png")
    plt.close()

