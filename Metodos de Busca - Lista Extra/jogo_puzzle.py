import time
from collections import deque
import heapq


# Estado Objetivo (Tupla - imutável e hashable)
OBJETIVO = (1, 2, 3, 4, 5, 6, 7, 8, 0) # Note que estou fixando o 0 no canto inferior direito, mas pode ser mudado conforme vimos em sala


# Mapeia o índice 0-8 para coordenadas (linha, coluna)
# INDICE_PARA_XY é uma lista de tuplas: [(0, 0), (0, 1), (0, 2), (1, 0), ...]
INDICE_PARA_XY = []
for i in range(9):
    r = i // 3  # Linha
    c = i % 3  # Coluna
    INDICE_PARA_XY.append((r, c))

# Mapeia o valor da peça (1-8, 0) para sua posição (linha, coluna) no estado OBJETIVO
# OBJETIVO_POS é um dicionário: {1: (0, 0), 2: (0, 1), ..., 0: (2, 2)}
OBJETIVO_POS = {}
for i in range(9):
    tile = OBJETIVO[i]
    OBJETIVO_POS[tile] = INDICE_PARA_XY[i]



# Estrutura de Resultado (classe Result)
class Result:
    """Armazena os detalhes da solução encontrada."""
    def __init__(self, found, caminho, moves, expanded, prof, elapsed):
        self.found = found  # Booleano: True se a solução foi encontrada
        self.caminho = caminho    # Lista de estados (tuplas) do caminho
        self.moves = moves  # Lista de movimentos 
        self.expanded = expanded  # Quantidade de nós expandidos
        self.prof = prof  # Profundidade da solução (tamanho do caminho - 1)
        self.elapsed = elapsed # Tempo gasto (em segundos)


# Auxiliar "parent" será um dicionário: {estado_filho: ParentLink(estado_pai, movimento)}
class ParentLink:
    """Armazena o estado pai e o movimento para chegar a um estado."""
    def __init__(self, parent_state, move):
        self.parent_state = parent_state
        self.move = move

# Funções Auxiliares
# Função vitál para determinar se é solucionável ou não
def is_solvable(s):
    """Verifica se um estado é solucionável (regra de inversões para o 8-puzzle)."""
    # Filtra o 0 (espaço vazio)
    arr = [x for x in s if x != 0]

    # Conta as inversões (pares onde um número maior vem antes de um menor)
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    # Solucionável se o número de inversões for par
    return inv % 2 == 0

def vizinhos(s):
    """Gera todos os estados vizinhos e o movimento para alcançá-los."""

    # Encontra a posição do espaço vazio (0)
    zero_idx = s.index(0)
    pos_x, pos_y = INDICE_PARA_XY[zero_idx] # Posição (linha, coluna) do zero
    
    moves_list = []

    def swap(i, j):
        """Troca dois elementos na tupla 's' e retorna a nova tupla."""
        lst = list(s) # Converte a tupla para lista (mutável)
        lst[i], lst[j] = lst[j], lst[i]
        return tuple(lst) # Retorna como tupla

    # Movimentos possíveis
    # Move Up (Se não estiver na 1a linha)
    if pos_x > 0:
        target_idx = (pos_x - 1) * 3 + pos_y
        moves_list.append((swap(zero_idx, target_idx), "Up"))
    # Move Down (Se não estiver na 3a linha)
    if pos_x < 2:
        target_idx = (pos_x + 1) * 3 + pos_y
        moves_list.append((swap(zero_idx, target_idx), "Down"))
    # Move Left (Se não estiver na 1a coluna)
    if pos_y > 0:
        target_idx = pos_x * 3 + (pos_y - 1)
        moves_list.append((swap(zero_idx, target_idx), "Left"))
    # Move Right (Se não estiver na 3a coluna)
    if pos_y < 2:
        target_idx = pos_x * 3 + (pos_y + 1)
        moves_list.append((swap(zero_idx, target_idx), "Right"))

    # moves_list é uma lista de tuplas: [(estado_vizinho, "Movimento"), ...]
    return moves_list

def build_result(parent, objetivo, expanded_count, elapsed):
    """Reconstrói o caminho do estado objetivo até o estado inicial."""
    caminho = []
    moves = []
    s = objetivo
    
    # Percorre o dicionário 'parent' de trás para frente
    while s is not None:
        caminho.append(s)
        # O ParentLink guarda o estado pai e o movimento para o estado atual 's'
        link = parent.get(s)
        
        if link:
            if link.move: # Se houver movimento (não é o nó inicial)
                moves.append(link.move)
            s = link.parent_state
        else:
            s = None # Chegou ao nó inicial (que não tem link pai)
            
    caminho.reverse() # Inverte o caminho para ir do início ao fim
    moves.reverse() # Inverte os movimentos
    
    return Result(True, caminho, moves, expanded_count, len(caminho) - 1, elapsed)

def print_matrix(state):
    """Imprime o estado do puzzle em formato 3x3 (substitui 0 por '_')."""
    # Exemplo: state[0:3] é a primeira linha, state[3:6] é a segunda, etc.
    for r in range(3):
        row_slice = state[r * 3 : (r + 1) * 3]
        # Converte cada número para string, e 0 para '_'
        row_str = []
        for x in row_slice:
            if x != 0:
                row_str.append(str(x))
            else:
                row_str.append("_")
        # Imprime os elementos separados por espaço
        print(" ".join(row_str))
    print() # Linha em branco para separação

def show_solution(res, name):
    print(f"\n## Resultado {name}")
    print(f"Tempo: {res.elapsed*1000:.2f} ms | Nós expandidos: {res.expanded} | Profundidade (movimentos): {res.prof}")


    if res.found:
        print("Movimentos: ", end="")
        for i, s in enumerate(res.caminho):
            if i < len(res.moves):
                print(f"{res.moves[i][0]}", end="") # Imprime apenas a primeira letra do movimento

        # Perguntar antes de imprimir os passos a passos
        print("\n\nDeseja visualizar o passo a passo?")
        print("  1 - SIM")
        print("  OUTRO - NÃO")
        choice = input("Opção: ").strip()
        if choice == "1":
            print("\n\nEstado Inicial:")
            for i, s in enumerate(res.caminho):
                print(f"Passo {i}:")
                print_matrix(s)
                # Se não for o último passo mostra o movimento
                if i < len(res.moves):
                    print(f"Movimento: {res.moves[i]}")
                    
    else:
        print("Solução não encontrada.")



# HEURÍSTICAS DO A*
def hamming(s):
    """Heurística Hamming: Conta o número de peças fora de posição."""
    misplaced_tiles = 0
    for i in range(9):
        tile = s[i]
        # Ignora o espaço vazio (0)
        if tile != 0 and tile != OBJETIVO[i]:
            misplaced_tiles += 1
    return misplaced_tiles

def manhattan(s):
    """Heurística Manhattan: Soma das distâncias de cada peça ao seu lugar objetivo."""
    dist_total = 0
    for i, tile in enumerate(s):
        if tile == 0:
            continue
        
        atual_posx, atual_posy = INDICE_PARA_XY[i]  # Posição atual (linha, coluna)
        dest_posx, dest_posy = OBJETIVO_POS[tile] # Posição objetivo (linha, coluna)
        
        # Distância Manhattan: |variaçãoX| + |variaçãoY|
        dist_total += abs(atual_posx - dest_posx) + abs(atual_posy - dest_posy)
        
    return dist_total

def count_conflicts(pairs):
    """Conta conflitos lineares em uma linha ou coluna."""
    # pairs é uma lista de tuplas: [(posicao_atual, posicao_objetivo), ...]
    cnt = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            pi, gi = pairs[i]
            pj, gj = pairs[j]
            # Conflito se a ordem relativa atual (pi < pj) é inversa à ordem objetivo (gi > gj)
            # OU se a ordem atual é inversa (pi > pj) e a ordem objetivo é direta (gi < gj)
            if (pi < pj and gi > gj) or (pi > pj and gi < gj):
                cnt += 1
    return cnt

def linear_conflicts(s):
    """Calcula o total de conflitos lineares em linhas e colunas."""
    conflicts_total = 0
    
    # Checa Linhas
    for l in range(3):
        linha = s[l * 3 : (l + 1) * 3] # Pega a linha como uma lista de 3
        colunas_objetivo = []
        for c, tile in enumerate(linha):
            # Se a peça não é zero E está na linha objetivo 'l'
            if tile != 0 and OBJETIVO_POS[tile][0] == l:
                # Adiciona o par: (coluna_atual, coluna_objetivo)
                colunas_objetivo.append((c, OBJETIVO_POS[tile][1]))
        conflicts_total += count_conflicts(colunas_objetivo)
        
    # Checa Colunas
    for c in range(3):
        # Constrói a coluna (lista dos 3 elementos da coluna 'c')
        coluna = [s[l * 3 + c] for l in range(3)] 
        linhas_objetivo = []
        for l, tile in enumerate(coluna):
            # Se a peça não é zero E está na coluna objetivo 'c'
            if tile != 0 and OBJETIVO_POS[tile][1] == c:
                # Adiciona o par: (linha_atual, linha_objetivo)
                linhas_objetivo.append((r, OBJETIVO_POS[tile][0]))
        conflicts_total += count_conflicts(linhas_objetivo)
        
    return conflicts_total

def manhattan_linear_conflict(s):
    """Heurística A* mManhattan + 2 * Conflitos Lineares."""
    # O multiplicador 2 garante que a heurística seja admissível (nunca superestima o custo, como discutido em sala)
    return manhattan(s) + 2 * linear_conflicts(s)




# Algoritmos de Busca
def bfs(start):
    """Busca em Largura iterativa"""
    t_inicio = time.perf_counter()
    # Fila 
    q = deque([start]) 
    
    # Conjunto de estados já vistos 
    visitados = {start}
    
    # Dicionário para rastrear a trajetória: {estado_filho: ParentLink(estado_pai, movimento)}
    parent = {start: ParentLink(None, None)}
    expanded = 0

    while q:
        s = q.popleft() # Pega o estado mais antigo da fila
        expanded += 1

        if s == OBJETIVO:
            t_fim = time.perf_counter()
            return build_result(parent, s, expanded, t_fim - t_inicio)

        for ns, m in vizinhos(s):
            # ns = estado vizinho, m = movimento para chegar a ele
            if ns not in visitados:
                visitados.add(ns)
                # Guarda o link para reconstruir o caminho depois
                parent[ns] = ParentLink(s, m) 
                q.append(ns)
                
    t_fim = time.perf_counter()
    return Result(False, [], [], expanded, 0, t_fim - t_inicio) # Solução não encontrada


def dfs(start, limit=200_000):
    """Busca em Profundidade (com limite de 200.000 vértices)"""
    t_inicio = time.perf_counter()
    # Pilha
    stack = [start]
    
    # Conjunto de estados já vistos
    visitados = {start}
    
    # Dicionário para rastrear a trajetória
    parent = {start: ParentLink(None, None)}
    expanded = 0

    # Adiciona um limite para evitar que DFS demore infinitamente em grafos grandes ou ciclos
    while stack and expanded < limit:
        s = stack.pop() # Pega o estado mais recente da pilha
        expanded += 1

        if s == OBJETIVO:
            t_fim = time.perf_counter()
            return build_result(parent, s, expanded, t_fim - t_inicio)

        # Adiciona vizinhos à pilha para explorar o caminho mais profundo primeiro
        for ns, m in vizinhos(s):
            if ns not in visitados:
                visitados.add(ns)
                # Guarda o link para reconstruir o caminho
                parent[ns] = ParentLink(s, m)
                stack.append(ns)
                
    t_fim = time.perf_counter()
    return Result(False, [], [], expanded, 0, t_fim - t_inicio) # Solução não encontrada


def astar(start, h):
    """Algoritmo A* (A-Star) com heurística 'h' passada no parâmetro"""
    t_inicio = time.perf_counter()

    # A tupla na fila será: (f_score, g_score, estado)
    # O heapq vai priorizar o menor f_score. Em caso de empate, o menor g_score.    
    f_start = h(start) # f = g + h, onde g=0
    open_heap = [(f_start, 0, start)]
    
    # parent: {estado_filho: ParentLink(estado_pai, movimento)}
    parent = {start: ParentLink(None, None)}
    
    # g_score: {estado: custo_do_caminho_ate_o_estado (g)}
    g_score = {start: 0}
    
    # closed: Conjunto de estados que já foram totalmente explorados
    closed = set()
    expanded = 0

    while open_heap:
        # Pega o estado com menor f_score da Fila de Prioridade
        f, g, s = heapq.heappop(open_heap)
        
        if s in closed:
            continue
        closed.add(s)
        expanded += 1

        if s == OBJETIVO:
            t_fim = time.perf_counter()
            return build_result(parent, s, expanded, t_fim - t_inicio)

        for ns, m in vizinhos(s):
            # Custo do caminho (g_score) para o vizinho 'ns'
            tentative_g = g + 1 

            # Verifica se 'ns' já está fechado E se o novo caminho é Pior ou Igual.
            # Se for pior ou igual, ignora (evita re-explorar caminhos ruins).
            # g_score.get(ns, float('inf')) retorna o g_score atual de ns, ou Infinito se não tiver sido descoberto.
            if ns in closed and tentative_g >= g_score.get(ns, float('inf')):
                continue
            
            # Se o novo caminho é melhor (tentative_g < g_score atual), atualiza o caminho
            if tentative_g < g_score.get(ns, float('inf')):
                
                # Atualiza o custo G (custo real do caminho)
                g_score[ns] = tentative_g
                
                # Atualiza o pai
                parent[ns] = ParentLink(s, m)
                
                # Adiciona/Atualiza 'ns' na Fila de Prioridade
                f_score = tentative_g + h(ns)
                heapq.heappush(open_heap, (f_score, tentative_g, ns)) # Adiciona na fila

    t_fim = time.perf_counter()
    return Result(False, [], [], expanded, 0, t_fim - t_inicio)



# MENU
def get_initial_state():
    while True:
        try:
            print("\n----------------------------------------------------")
            print("Defina o estado inicial (digite 9 números de 0 a 8, sem espaços ou vírgulas):")
            print("Exemplo: 038471256 (0 é o espaço vazio)")
            input_str = input("Estado: ").strip()
            
            # Checa o tamanho
            if len(input_str) != 9:
                raise ValueError("O estado inicial deve conter exatamente 9 dígitos.")
                
            # Converte para tupla de inteiros
            start_list = [int(d) for d in input_str]
            start = tuple(start_list)

            # Verificara se tem todos os dígitos de 0 a 8
            if sorted(start) != [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                raise ValueError("O estado deve conter os números de 0 a 8, uma única vez.")
            
            return start
            
        except ValueError as e:
            print(f"\n Erro de entrada: {e}")
            print("Tente novamente.")

def run_cli():
    # Obter o estado inicial do usuário
    start = get_initial_state()
    
    print("\n--- Estado Inicial ---")
    print_matrix(start) # Mostrar a representação do puzzle 

    if not is_solvable(start):
        print("Este estado é **NÃO SOLUCIONÁVEL**. Não é possível prosseguir com a busca.")
        return

    # Menu de seleção de algoritmo
    while True:
        print("\n----------------------------------------------------")
        print("Escolha o Algoritmo/Heurística:")
        print("  1 - Busca em Largura (BFS)")
        print("  2 - Busca em Profundidade (DFS)")
        print("  3 - A* (Heurística Hamming)")
        print("  4 - A* (Heurística Manhattan)")
        print("  5 - A* (Manhattan + Conflito Linear)")
        print("  6 - Sair")
        
        choice = input("Opção: ").strip()
        
        if choice == '6':
            print("Encerrando o programa...")
            break
            
        algorithm_map = {
            '1': (bfs, None, "Busca em Largura (BFS)"),
            '2': (dfs, None, "Busca em Profundidade (DFS)"),
            '3': (astar, hamming, "A* (Hamming)"),
            '4': (astar, manhattan, "A* (Manhattan)"),
            '5': (astar, manhattan_linear_conflict, "A* (Manhattan + Conflito)")
        }
        
        if choice in algorithm_map:
            func, heuristic, name = algorithm_map[choice]
            
            print(f"\nIniciando busca: {name}...")
            
            # Chama o algoritmo com ou sem heurística
            if heuristic:
                result = func(start, heuristic)
            else:
                result = func(start)
                
            # Exibe o resultado
            show_solution(result, name)
            
        else:
            print("Opção inválida. Digite um número de 1 a 6.")



# EXECUÇÃO
if __name__ == "__main__":
    run_cli()