import networkx as nx
import numpy as np
from itertools import combinations
from math import factorial
import time
import random

# Глобальный кэш для максимальных потоков (временно отключим)
flow_cache = {}

def build_graph(matrix):
    matrix = np.array(matrix)
    G = nx.Graph()
    n = len(matrix)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0:
                G.add_edge(i + 1, j + 1, capacity=matrix[i][j])
                edges.append((i + 1, j + 1))
    is_symmetric = np.allclose(matrix, matrix.T, rtol=1e-5, equal_nan=True)
    if not is_symmetric:
        print("Внимание: Матрица не симметрична! Граф будет интерпретирован как неориентированный.")
    return G, edges

def max_flow_value(G, source, target):
    # Временно отключили кэширование для проверки
    try:
        flow_value, _ = nx.maximum_flow(G, source, target, capacity='capacity')
        return flow_value
    except nx.NetworkXError:
        return 0

def compute_v_N(G, nodes, sample_size=None):
    v_N = 0
    n = len(nodes)
    flow_matrix = np.zeros((n, n))
    node_map = {node: idx for idx, node in enumerate(sorted(nodes))}

    pairs = list(combinations(nodes, 2))
    if sample_size and n > 5 and len(pairs) > sample_size:
        pairs = random.sample(pairs, sample_size)

    for s, t in pairs:
        s_shifted = s + 1
        t_shifted = t + 1
        flow = max_flow_value(G, s_shifted, t_shifted)
        s_idx = node_map[s]
        t_idx = node_map[t]
        flow_matrix[s_idx][t_idx] = flow
        flow_matrix[t_idx][s_idx] = flow
        v_N += flow

    if sample_size and n > 5 and len(pairs) < n * (n - 1) // 2:
        total_pairs = n * (n - 1) // 2
        v_N = v_N * (total_pairs / len(pairs)) if len(pairs) > 0 else v_N

    return v_N, flow_matrix

def characteristic_function(G, coalitions):
    v_values = {}
    flow_matrices = {}

    for coalition in coalitions:
        coalition = set(coalition)
        if len(coalition) < 2:
            v_values[frozenset(coalition)] = 0
            flow_matrices[frozenset(coalition)] = np.zeros((len(coalition), len(coalition)))
            continue
        coalition_shifted = {x + 1 for x in coalition}
        H = nx.Graph()
        H.add_nodes_from(coalition_shifted)
        for u, v, data in G.edges(data=True):
            if u in coalition_shifted and v in coalition_shifted:
                H.add_edge(u, v, **data)

        flow_matrix = np.zeros((len(coalition), len(coalition)))
        total_flow = 0

        node_map = {node: idx for idx, node in enumerate(sorted(coalition_shifted))}
        pairs = list(combinations(range(len(coalition)), 2))
        for i, j in pairs:
            s = sorted(coalition_shifted)[i]
            t = sorted(coalition_shifted)[j]
            if nx.has_path(H, s, t):
                flow = max_flow_value(H, s, t)
                flow_matrix[i][j] = flow
                flow_matrix[j][i] = flow
                total_flow += flow

        print(f"Матрица потоков для коалиции {coalition_shifted}:")
        print(flow_matrix)
        v_values[frozenset(coalition)] = total_flow
        flow_matrices[frozenset(coalition)] = flow_matrix
    return v_values, flow_matrices, None

def arc_game_shapley(G, edges, v_N, max_subset_size=None, sample_subsets=True, num_samples=1000):
    A = set(edges)
    r_N = {frozenset(): 0}
    v_cache = {}

    print("Вычисление r_N для подмножеств ребер...")
    nodes = list(G.nodes)
    n = len(nodes)
    max_subset_size = len(A) if max_subset_size is None else min(max_subset_size, len(A))
    if len(A) <= 10:
        sample_subsets = False

    subsets_to_compute = []
    for L in range(1, max_subset_size + 1):
        subsets = list(combinations(A, L))
        if sample_subsets and len(subsets) > num_samples // max_subset_size:
            subsets = random.sample(subsets, min(len(subsets), num_samples // max_subset_size))
        subsets_to_compute.extend(subsets)

    H = nx.Graph()
    H.add_nodes_from(nodes)
    for subset in subsets_to_compute:
        subset = frozenset(subset)
        H_copy = H.copy()
        H_copy.add_edges_from(subset, capacity={e: G.edges[e]['capacity'] for e in subset if e in G.edges})

        components = list(nx.connected_components(H_copy))
        value = 0
        for component in components:
            component_set = frozenset(i - 1 for i in component)
            if len(component_set) < 2:
                continue
            if component_set in v_cache:
                v_component = v_cache[component_set]
            else:
                component_nodes = [x + 1 for x in component_set]
                component_graph = G.subgraph(component_nodes).copy()
                v_component = 0
                pairs = list(combinations(component_set, 2))
                for s, t in pairs:
                    s_shifted = s + 1
                    t_shifted = t + 1
                    v_component += max_flow_value(component_graph, s_shifted, t_shifted)
                v_cache[component_set] = v_component
            value += v_component
        r_N[subset] = value

    shapley_arcs = {}
    print("Вычисление значений Шепли для ребер...")
    n_edges = len(A)
    for a in A:
        shapley_a = 0
        for L in range(1, max_subset_size + 1):
            subsets = list(combinations(A, L))
            if sample_subsets and len(subsets) > num_samples // max_subset_size:
                subsets = random.sample(subsets, min(len(subsets), num_samples // max_subset_size))
            for subset in subsets:
                subset = frozenset(subset)
                if a in subset:
                    subset_without_a = subset - frozenset({a})
                    marginal_contribution = r_N.get(subset, 0) - r_N.get(subset_without_a, 0)
                    weight = factorial(L - 1) * factorial(n_edges - L) / factorial(n_edges)
                    shapley_a += weight * marginal_contribution
        shapley_arcs[a] = max(shapley_a, 0)

    current_sum = sum(shapley_arcs.values())
    if current_sum > 0:
        correction_factor = v_N / current_sum
        shapley_arcs = {edge: value * correction_factor for edge, value in shapley_arcs.items()}

    return shapley_arcs, r_N

def position_value(G, edges, v_N):
    n = len(G.nodes)
    position = np.zeros(n)
    shapley_arcs, _ = arc_game_shapley(G, edges, v_N)
    for i in range(n):
        node = i + 1
        A_i = {(u, v) for u, v in edges if u == node or v == node}
        position[i] = sum(0.5 * shapley_arcs[a] for a in A_i) if A_i else 0
    return position, shapley_arcs

def shapley_value_nodes(G, coalitions, v_values):
    n = len(G.nodes)
    shapley = np.zeros(n)

    for i in range(n):
        for coalition in coalitions:
            coalition = set(coalition)
            if i not in coalition:
                continue
            s = len(coalition)
            coalition_without_i = coalition - {i}
            v_with_i = v_values[frozenset(coalition)]
            v_without_i = v_values.get(frozenset(coalition_without_i), 0)
            weight = factorial(s - 1) * factorial(n - s) / factorial(n)
            shapley[i] += weight * (v_with_i - v_without_i)
    return shapley

def main():
    print("Введите размер матрицы (количество вершин):")
    n = int(input())

    matrix = []
    print("Введите матрицу пропускных способностей построчно (0 для отсутствия ребра):")
    for i in range(n):
        row = list(map(int, input().split()))
        if len(row) != n:
            print("Ошибка: строка должна содержать", n, "элементов!")
            return
        matrix.append(row)

    print("Выберите тип игроков (1 - вершины, 2 - ребра):")
    player_type = int(input())

    G, edges = build_graph(matrix)
    nodes = list(range(n))

    start_time = time.time()
    if player_type == 1:  # Вершины
        coalitions = []
        for r in range(1, len(nodes) + 1):
            coalitions.extend(list(combinations(nodes, r)))
        coalitions.append(tuple())

        v_values, flow_matrices, _ = characteristic_function(G, coalitions)

        print("\nЗначения характеристической функции и матрицы потоков:")
        for coalition in coalitions:
            coalition = frozenset(coalition)
            coal_set = {x + 1 for x in coalition}
            v = v_values[coalition]
            print(f"Коалиция: {coal_set}, V(S) = {v}")
            if coalition:
                flow_matrix = flow_matrices[coalition]
                print("Матрица потоков:")
                coal_indices = sorted(coalition)
                node_map = {node: idx for idx, node in enumerate(coal_indices)}
                sub_matrix = [[flow_matrix[node_map[i]][node_map[j]] for j in coal_indices] for i in coal_indices]
                print(np.array(sub_matrix))
            print()

        total_value = v_values[frozenset(nodes)]
        print(f"V(N) = {total_value}")

        shapley = shapley_value_nodes(G, coalitions, v_values)
        print("Вектор Шепли:", [round(float(x), 5) for x in shapley])
        if np.sum(shapley) == 0:
            print("Ошибка: сумма значений Шепли равна 0, нормализация невозможна.")
            centrality = np.zeros_like(shapley)
        else:
            centrality = shapley / np.sum(shapley)
        print("Нормализованный вектор Шепли (центральность):", [round(float(x), 5) for x in centrality])

    elif player_type == 2:  # Ребра
        v_N, flow_matrix_N = compute_v_N(G, nodes)

        print("\nЗначение характеристической функции для полной коалиции:")
        print(f"V(N) = {v_N}")
        print("Матрица потоков для полной коалиции:")
        print(flow_matrix_N)

        position, shapley_arcs = position_value(G, edges, v_N)
        print("Позиционное значение (для вершин):", [round(float(x), 5) for x in position])
        print("Значения Шепли для ребер:", {str(edge): round(float(value), 5) for edge, value in shapley_arcs.items()})
        print("Сумма значений Шепли для ребер:", round(sum(shapley_arcs.values()), 5))
        if np.sum(position) == 0:
            print("Ошибка: сумма позиционных значений равна 0, нормализация невозможна.")
            centrality_nodes = np.zeros_like(position)
        else:
            centrality_nodes = position / np.sum(position)
        if np.sum(list(shapley_arcs.values())) == 0:
            print("Ошибка: сумма значений Шепли для ребер равна 0, нормализация невозможна.")
            centrality_edges = {edge: 0 for edge in shapley_arcs}
        else:
            total_shapley = np.sum(list(shapley_arcs.values()))
            centrality_edges = {edge: value / total_shapley for edge, value in shapley_arcs.items()}
        print("Центральность вершин (по позиционному значению):", [round(float(x), 5) for x in centrality_nodes])
        print("Центральность ребер:", {str(edge): round(float(value), 5) for edge, value in centrality_edges.items()})
        print("Сумма центральностей ребер:", round(sum(centrality_edges.values()), 5))

    else:
        print("Неверный выбор типа игроков. Используйте 1 или 2.")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время отработки: {execution_time:.2f} секунд")

if __name__ == "__main__":
    main()
