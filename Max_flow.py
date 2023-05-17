import itertools
import numpy as np
import networkx as nx
from math import factorial
from queue import Queue
# Зададим граф
capacity = np.array(eval(input("Задайте граф его матрицей пропускных способностей: ")))
G = nx.DiGraph()
for i in range(1, capacity.shape[0] + 1):
    for j in range(1, capacity.shape[1] + 1):
        if capacity[i - 1][j - 1] > 0:
            G.add_edge(i, j, capacity=capacity[i - 1][j - 1])
def max_flow(coalition):
    H = G.subgraph(coalition)
    n = len(coalition)
    nodes = list(H.nodes())
    edges = list(H.edges())
    node_dict = {node: i for i, node in enumerate(nodes)}
    # Дерево Гомори-Ху
    parent = {i: 0 for i in range(n)}
    capacity = [[0] * n for _ in range(n)]
    tree_nodes = list(range(n))
    for i in range(n - 1):
        s, t = tree_nodes[0], tree_nodes[-1]
        if s == t:
            break
        # Поиск минимального разреза
        bfs_queue = [s]
        visited = {s}
        cut = {node_dict[node] for node in nodes if node not in visited}
        while bfs_queue:
            u = bfs_queue.pop(0)
            for v in H.neighbors(nodes[u]):
                if v not in visited and capacity[u][node_dict[v]] < np.inf:
                    visited.add(v)
                    bfs_queue.append(node_dict[v])
                    if v in cut:
                        cut.remove(v)
        # Обновление дерева Гомори-Ху
        parent[t] = s
        for i in cut:
            if i != s and i != t:
                tree_nodes.append(i)
                parent[i] = t
        tree_nodes.remove(s)
        # Обновление матрицы
        flow_value = nx.maximum_flow_value(H, nodes[s], nodes[t], capacity="capacity")
        for i in cut:
            for j in cut:
                if capacity[i][j] < np.inf:
                    capacity[i][j] += flow_value
        for i in cut:
            if i != s:
                capacity[s][i] = capacity[i][s] = np.inf
        for i in cut:
            if i != t:
                capacity[t][i] = capacity[i][t] = np.inf
    # Максимальные потоки между вершинами
    flow_matrix = np.zeros((n, n), dtype=int)
    for i, node_i in enumerate(coalition):
        for j, node_j in enumerate(coalition):
            if i != j:
                flow_value = nx.maximum_flow_value(H, node_i, node_j, capacity="capacity")
                flow_matrix[i][j] = round((flow_value),0)
                # flow_matrix[i][j] = round((flow_value),0)
    return flow_matrix
# Максимальные потоки коалиций
n = G.number_of_nodes()
nodes = list(G.nodes())
max_flow_matrices = {}
v_values = {}
shapley_value = {i: 0 for i in nodes}
for i in range(2, len(nodes) + 1):
    for comb in itertools.combinations(nodes, i):
        comb_key = tuple(sorted(comb))
        max_flow_matrix = max_flow(comb_key)
        max_flow_matrices[comb_key] = max_flow(comb_key)
        v_value = sum(max_flow_matrix.flatten()) / 2
        v_values[comb_key] = v_value
# Вектор Шепли
shapley_value = {}
norm_shapley = {}
for i in nodes:
    shapley_value[i] = 0
    for j in range(1, len(nodes)):
        for comb in itertools.combinations(set(nodes) - {i}, j):
            # исключаем одиночные коалиции
            if len(comb) > 0:
                comb_key = tuple(sorted(comb + (i,)))
                marg_contribution = (v_values.get(comb_key, 0) - v_values.get(tuple(sorted(comb)), 0)) * factorial(
                    len(comb)) * factorial(len(nodes) - len(comb) -1) / factorial(len(nodes))
                shapley_value[i] += marg_contribution
                norm_shapley[i] = shapley_value[i]/v_value
# Вывод
for coalition, flow_matrix in max_flow_matrices.items():
    print(f"Коалиция: {coalition}")
    print("Матрица максимальных потоков:")
    print(flow_matrix)
    print("Значение характеристической функции:")
    print(v_values[coalition])
    print()
print("Вектор Шепли:")
print(shapley_value)
print("Вектор центральности:")
print(norm_shapley)