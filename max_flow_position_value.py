import networkx as nx
import numpy as np
from itertools import combinations, permutations
from math import factorial

def build_graph(matrix):
    G = nx.Graph()
    n = len(matrix)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0:
                G.add_edge(i + 1, j + 1, capacity=matrix[i][j])
                edges.append((i + 1, j + 1))
    return G, edges

def max_flow_from_tree(G, tree, coalition):
    if len(coalition) < 2:
        return 0, np.zeros((len(coalition), len(coalition)))
    
    coalition_shifted = {x + 1 for x in coalition}
    H = nx.Graph()
    H.add_nodes_from(coalition_shifted)
    for u, v in combinations(coalition_shifted, 2):
        if G.has_edge(u, v):
            H.add_edge(u, v, capacity=G[u][v]['capacity'])
    
    flow_matrix = np.zeros((len(coalition), len(coalition)))
    total_flow = 0
    
    node_map = {node: idx for idx, node in enumerate(sorted(coalition_shifted))}
    for s, t in combinations(coalition_shifted, 2):
        if nx.has_path(H, s, t):
            path = nx.shortest_path(tree, s, t)
            min_capacity = float('inf')
            for u, v in zip(path, path[1:]):
                edge_data = tree[u][v]
                capacity = edge_data.get('capacity', edge_data.get('weight', 0))
                min_capacity = min(min_capacity, capacity)
            s_idx = node_map[s]
            t_idx = node_map[t]
            flow_matrix[s_idx][t_idx] = min_capacity
            flow_matrix[t_idx][s_idx] = min_capacity
            total_flow += min_capacity
            print(f"Поток между {s} и {t}: {min_capacity}")
    
    print(f"Матрица потоков для коалиции {coalition_shifted}:")
    print(flow_matrix)
    return total_flow, flow_matrix

def characteristic_function(G, coalitions):
    v_values = {}
    flow_matrices = {}
    tree = nx.gomory_hu_tree(G, capacity='capacity')
    print("Дерево Гомори-Ху:", tree.edges(data=True))
    
    for coalition in coalitions:
        coalition = set(coalition)
        v, flow_matrix = max_flow_from_tree(G, tree, coalition)
        v_values[frozenset(coalition)] = v
        flow_matrices[frozenset(coalition)] = flow_matrix
    return v_values, flow_matrices

def shapley_value_nodes(G, coalitions, v_values):
    n = len(G.nodes)
    shapley = np.zeros(n)
    
    for i in range(n):
        for coalition in coalitions:
            coalition = set(coalition)
            if i + 1 not in [x + 1 for x in coalition]:
                continue
            s = len(coalition)
            coalition_without_i = coalition - {i}
            v_with_i = v_values[frozenset(coalition)]
            v_without_i = v_values[frozenset(coalition_without_i)]
            weight = factorial(s - 1) * factorial(n - s) / factorial(n)
            shapley[i] += weight * (v_with_i - v_without_i)
    return shapley

def arc_game_shapley(G, edges, v_values):
    A = set(edges)
    r_N = {}
    nodes = list(G.nodes)
    
    # Вычисляем r_N(L) = sum(v(C)) для всех компонент C, формируемых ребрами L
    for L in range(len(A) + 1):
        for subset in combinations(A, L):
            subset = frozenset(subset)
            # Создаем подграф с ребрами из subset
            H = nx.Graph()
            H.add_nodes_from(nodes)
            H.add_edges_from(subset)
            
            # Находим компоненты связности
            components = list(nx.connected_components(H))
            
            # Суммируем v(C) для каждой компоненты C
            value = 0
            for component in components:
                component = frozenset({i - 1 for i in component})  # Сдвиг индексов обратно для v_values
                if component in v_values:
                    value += v_values[component]
            r_N[subset] = value
    
    # Вычисляем значение Шепли для каждого ребра
    shapley_arcs = {}
    for a in A:
        shapley_a = 0
        for L in range(1, len(A) + 1):
            for subset in combinations(A, L):
                subset = frozenset(subset)
                if a in subset:
                    subset_without_a = subset - frozenset({a})
                    marginal_contribution = r_N[subset] - r_N[subset_without_a]
                    weight = factorial(len(subset) - 1) * factorial(len(A) - len(subset)) / factorial(len(A))
                    shapley_a += weight * marginal_contribution
        shapley_arcs[a] = shapley_a
    return shapley_arcs, r_N

def position_value(G, edges, v_values):
    n = len(G.nodes)
    position = np.zeros(n)
    shapley_arcs, _ = arc_game_shapley(G, edges, v_values)
    for i in range(n):
        node = i + 1
        A_i = {(u, v) for u, v in edges if u == node or v == node}
        position[i] = sum(0.5 * shapley_arcs[a] for a in A_i) if A_i else 0
    return position, shapley_arcs

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
    
    coalitions = []
    for r in range(1, len(nodes) + 1):
        coalitions.extend(list(combinations(nodes, r)))
    coalitions.append(tuple())
    
    v_values, flow_matrices = characteristic_function(G, coalitions)
    
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
    
    if player_type == 1:  # Вершины
        shapley = shapley_value_nodes(G, coalitions, v_values)
        print("Вектор Шепли:", [round(float(x), 5) for x in shapley])
        if total_value == 0:
            print("Ошибка: V(N) равно 0, нормализация невозможна. Граф может быть несвязным.")
            centrality = np.zeros_like(shapley)
        else:
            centrality = shapley / total_value
        print("Нормализованный вектор Шепли (центральность):", [round(float(x), 5) for x in centrality])
    
    elif player_type == 2:  # Ребра
        position, shapley_arcs = position_value(G, edges, v_values)
        print("Позиционное значение (для вершин):", [round(float(x), 5) for x in position])
        print("Значения Шепли для ребер:", {str(edge): round(float(value), 5) for edge, value in shapley_arcs.items()})
        if total_value == 0:
            print("Ошибка: V(N) равно 0, нормализация невозможна. Граф может быть несвязным.")
            centrality_nodes = np.zeros_like(position)
            centrality_edges = {edge: 0 for edge in shapley_arcs}
        else:
            centrality_nodes = position / total_value
            total_shapley_arcs = sum(shapley_arcs.values())
            centrality_edges = {edge: value / total_value if total_value != 0 else 0 for edge, value in shapley_arcs.items()}
        print("Центральность вершин (по позиционному значению):", [round(float(x), 5) for x in centrality_nodes])
        print("Центральность ребер:", {str(edge): round(float(value), 5) for edge, value in centrality_edges.items()})
    
    else:
        print("Неверный выбор типа игроков. Используйте 1 или 2.")

if __name__ == "__main__":
    main()