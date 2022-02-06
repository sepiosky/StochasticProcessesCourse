import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging

def draw_trans_graph(adajency_matrix, graph_rows_cols=(5,5), ax=None):
    G=nx.Graph()
    for i in range(graph_rows_cols[0]):
        for j in range(graph_rows_cols[1]):
            node = i*5+j
            G.add_node(node,pos=(j,5-i))
            for k in range(0,adajency_matrix.shape[0]):
                if adajency_matrix[node][k]==1:
                    G.add_edge(node, k)
    if ax is None:
        ax=plt.subplots()[1]
    nx.draw(G, pos=nx.get_node_attributes(G,'pos'), with_labels=True, ax=ax)
    
def logsumexp(a, axis=None):
    maxel = np.amax(a)
    if np.isposinf(maxel):
        return maxel
    return maxel + np.log(np.sum(np.exp(a-maxel), axis=axis)) 

def set_logger(path, logfile):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_handlers = [logging.StreamHandler()]
    if logfile:
        log_handlers.append(logging.FileHandler(path, mode="w"))
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=log_handlers)