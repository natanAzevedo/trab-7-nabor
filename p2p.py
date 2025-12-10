import json
import sys
import random
import time
import os
from collections import deque, defaultdict
from typing import Dict, Set, List, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class Node:
    def __init__(self, node_id: str, resources: Set[str]):
        self.id = node_id
        self.resources = set(resources)
        self.neighbors: Set[str] = set()
        # cache[resource_id] = set(node_ids que possuem o recurso)
        self.cache: Dict[str, Set[str]] = defaultdict(set)

    def add_neighbor(self, neighbor_id: str):
        if neighbor_id == self.id:
            raise ValueError(f"Aresta de loop detectada em {self.id}")
        self.neighbors.add(neighbor_id)


class P2PNetwork:
    def __init__(self, config: dict):
        self.nodes: Dict[str, Node] = {}
        self.min_neighbors = config["min_neighbors"]
        self.max_neighbors = config["max_neighbors"]

        # Cria nós
        for node_id, res_list in config["resources"].items():
            if not res_list:
                raise ValueError(f"Nó {node_id} sem recursos")
            self.nodes[node_id] = Node(node_id, set(res_list))

        # Cria arestas
        for a, b in config["edges"]:
            if a not in self.nodes or b not in self.nodes:
                raise ValueError(f"Aresta inválida: {a}-{b}")
            if a == b:
                raise ValueError(f"Aresta de loop detectada em {a}")
            self.nodes[a].add_neighbor(b)
            self.nodes[b].add_neighbor(a)

        # Valida rede
        self._validate_degrees()
        self._validate_connected()

    def _validate_degrees(self):
        for node in self.nodes.values():
            deg = len(node.neighbors)
            if deg < self.min_neighbors or deg > self.max_neighbors:
                raise ValueError(
                    f"Nó {node.id} tem {deg} vizinhos, "
                    f"fora do intervalo [{self.min_neighbors}, {self.max_neighbors}]"
                )

    def _validate_connected(self):
        # BFS a partir de um nó qualquer
        start = next(iter(self.nodes))
        visited = set()
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if u in visited:
                continue
            visited.add(u)
            for v in self.nodes[u].neighbors:
                if v not in visited:
                    queue.append(v)
        if len(visited) != len(self.nodes):
            raise ValueError("A rede está particionada (não é totalmente conectada)")

    # ---------- Utilidades comuns ----------

    def _update_cache_on_hit(self, path: List[str], resource_id: str, target_id: str):
        """
        Atualiza o cache de todos os nós no caminho com a informação
        de que 'target_id' possui 'resource_id'.
        """
        for node_id in path:
            self.nodes[node_id].cache[resource_id].add(target_id)

    # ---------- Visualização ----------

    def visualize_network(self, save_path: Optional[str] = None):
        """
        Exibe uma representação gráfica da rede P2P.
        Se save_path for fornecido, salva a imagem ao invés de exibir.
        """
        G = nx.Graph()
        
        # Adiciona nós
        for node_id, node in self.nodes.items():
            resources_str = ', '.join(sorted(node.resources))
            G.add_node(node_id, resources=resources_str)
        
        # Adiciona arestas
        for node_id, node in self.nodes.items():
            for neighbor_id in node.neighbors:
                if node_id < neighbor_id:  # Evita duplicatas
                    G.add_edge(node_id, neighbor_id)
        
        # Configuração do layout
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Desenha nós
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1500, alpha=0.9)
        
        # Desenha arestas
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
        
        # Labels dos nós com recursos
        labels = {}
        for node_id in G.nodes():
            resources = G.nodes[node_id]['resources']
            labels[node_id] = f"{node_id}\n[{resources}]"
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Rede P2P - Topologia e Recursos", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Rede visualizada e salva em: {save_path}")
            plt.close()
        else:
            plt.show()

    def visualize_search_animated(self, node_id: str, resource_id: str, 
                                 ttl: int, algo: str, seed: Optional[int] = None,
                                 save_path: Optional[str] = None):
        """
        Cria uma animação da busca em tempo real.
        Se save_path for fornecido, salva como GIF.
        """
        if seed is not None:
            random.seed(seed)
        
        # Executa a busca com rastreamento
        search_steps = self._search_with_tracking(node_id, resource_id, ttl, algo)
        
        # Cria o grafo NetworkX
        G = nx.Graph()
        for nid, node in self.nodes.items():
            G.add_node(nid)
        for nid, node in self.nodes.items():
            for neighbor_id in node.neighbors:
                if nid < neighbor_id:
                    G.add_edge(nid, neighbor_id)
        
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Configuração da figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update(frame):
            ax.clear()
            
            if frame >= len(search_steps):
                frame = len(search_steps) - 1
            
            step_data = search_steps[frame]
            current_nodes = step_data['visited']
            current_path = step_data['current_path']
            found = step_data['found']
            msg_count = step_data['msg_count']
            
            # Cores dos nós
            node_colors = []
            for n in G.nodes():
                # Protege o acesso a current_path[-1]
                if found and current_path and n == current_path[-1]:
                    node_colors.append('green')  # Nó que possui o recurso
                elif n == node_id:
                    node_colors.append('orange')  # Nó inicial
                elif n in current_nodes:
                    node_colors.append('yellow')  # Nós visitados
                else:
                    node_colors.append('lightgray')  # Nós não visitados
            
            # Desenha nós
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=1500, alpha=0.9, ax=ax)
            
            # Desenha todas as arestas em cinza
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, ax=ax)
            
            # Desenha o caminho atual em vermelho
            if len(current_path) > 1:
                path_edges = [
                    (current_path[i], current_path[i+1])
                    for i in range(len(current_path)-1)
                    if G.has_edge(current_path[i], current_path[i+1])
                ]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                      edge_color='red', width=3, ax=ax)
            
            # Labels dos nós
            labels = {n: n for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
            
            # Informações da busca
            status = "RECURSO ENCONTRADO!" if found else f"Buscando... (Passo {frame+1}/{len(search_steps)})"
            title = f"Busca: {algo}\n"
            title += f"Origem: {node_id} | Recurso: {resource_id} | TTL: {ttl}\n"
            title += f"{status}\n"
            title += f"Mensagens: {msg_count} | Nós envolvidos: {len(current_nodes)}"
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        anim = FuncAnimation(fig, update, frames=len(search_steps), 
                   interval=800, repeat=False, blit=False)
        
        if save_path:
            # Garante que o diretório de destino exista
            save_path = str(save_path)
            parent = os.path.dirname(save_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)

            # Se não tiver extensão conhecida, assume .gif
            if not save_path.lower().endswith(('.gif', '.mp4', '.avi')):
                save_path = save_path + '.gif'

            # Salva usando PillowWriter para compatibilidade no Windows
            writer = PillowWriter(fps=1)
            anim.save(save_path, writer=writer, dpi=150)
            print(f"Animação salva em: {save_path}")
            plt.close(fig)
        else:
            plt.show()
        
        return search_steps[-1]['found'], search_steps[-1]['msg_count'], \
               len(search_steps[-1]['visited']), search_steps[-1]['current_path']

    def _search_with_tracking(self, node_id: str, resource_id: str, 
                            ttl: int, algo: str) -> List[dict]:
        """
        Executa a busca e retorna uma lista de estados (steps) para animação.
        """
        algo = algo.lower()
        if algo == "flooding":
            return self._search_flooding_tracked(node_id, resource_id, ttl, informed=False)
        elif algo == "informed_flooding":
            return self._search_flooding_tracked(node_id, resource_id, ttl, informed=True)
        elif algo == "random_walk":
            return self._search_random_walk_tracked(node_id, resource_id, ttl, informed=False)
        elif algo == "informed_random_walk":
            return self._search_random_walk_tracked(node_id, resource_id, ttl, informed=True)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algo}")

    def _search_flooding_tracked(self, start_id: str, resource_id: str, 
                                ttl: int, informed: bool) -> List[dict]:
        """
        Flooding com backtracking e rastreamento para animação.
        """
        steps = []
        msg_count = 0
        visited = set()
        nodes_involved = set()
        queue = deque([(start_id, ttl, [start_id])])
        visited.add(start_id)
        nodes_involved.add(start_id)
        
        # Estado inicial
        steps.append({
            'visited': visited.copy(),
            'current_path': [start_id],
            'found': False,
            'msg_count': 0
        })
        
        while queue:
            node_id, ttl_left, path = queue.popleft()
            node = self.nodes[node_id]
            
            # Adiciona step
            steps.append({
                'visited': visited.copy(),
                'current_path': path.copy(),
                'found': False,
                'msg_count': msg_count
            })
            
            # Verifica recurso
            if resource_id in node.resources:
                self._update_cache_on_hit(path, resource_id, node_id)
                steps.append({
                    'visited': visited.copy(),
                    'current_path': path.copy(),
                    'found': True,
                    'msg_count': msg_count
                })
                return steps
            
            # Busca informada
            if informed and resource_id in node.cache and node.cache[resource_id]:
                target_id = next(iter(node.cache[resource_id]))
                msg_count += 1
                path2 = path + [target_id]
                nodes_involved.add(target_id)
                self._update_cache_on_hit(path2, resource_id, target_id)
                steps.append({
                    'visited': nodes_involved.copy(),
                    'current_path': path2.copy(),
                    'found': True,
                    'msg_count': msg_count
                })
                return steps
            
            if ttl_left <= 0:
                continue
            
            # Envia para vizinhos não visitados
            for neigh_id in node.neighbors:
                if neigh_id not in visited:
                    visited.add(neigh_id)
                    nodes_involved.add(neigh_id)
                    msg_count += 1
                    queue.append((neigh_id, ttl_left - 1, path + [neigh_id]))
        
        return steps

    def _search_random_walk_tracked(self, start_id: str, resource_id: str, 
                                   ttl: int, informed: bool) -> List[dict]:
        """
        Random Walk com backtracking e rastreamento para animação:
        - TTL representa quantos níveis de profundidade podemos explorar
        - Backtracking NÃO consome TTL
        """
        animation_steps = []
        msg_count = 0
        visited = set()
        current_id = start_id
        path = [current_id]
        visited.add(current_id)
        path_ttl = [ttl]
        
        # Estado inicial
        animation_steps.append({
            'visited': visited.copy(),
            'current_path': path.copy(),
            'found': False,
            'msg_count': 0
        })
        
        while True:
            node = self.nodes[current_id]
            current_ttl = path_ttl[-1]
            
            # Adiciona step
            animation_steps.append({
                'visited': visited.copy(),
                'current_path': path.copy(),
                'found': False,
                'msg_count': msg_count
            })
            
            # Verifica recurso
            if resource_id in node.resources:
                self._update_cache_on_hit(path, resource_id, current_id)
                animation_steps.append({
                    'visited': visited.copy(),
                    'current_path': path.copy(),
                    'found': True,
                    'msg_count': msg_count
                })
                return animation_steps
            
            # Busca informada
            if informed and resource_id in node.cache and node.cache[resource_id]:
                target_id = next(iter(node.cache[resource_id]))
                msg_count += 1
                path.append(target_id)
                visited.add(target_id)
                self._update_cache_on_hit(path, resource_id, target_id)
                animation_steps.append({
                    'visited': visited.copy(),
                    'current_path': path.copy(),
                    'found': True,
                    'msg_count': msg_count
                })
                return animation_steps
            
            # Encontra vizinhos não visitados
            unvisited_neighbors = [n for n in node.neighbors if n not in visited]
            
            if unvisited_neighbors and current_ttl > 0:
                next_id = random.choice(unvisited_neighbors)
                msg_count += 1
                current_id = next_id
                path.append(current_id)
                path_ttl.append(current_ttl - 1)
                visited.add(current_id)
            else:
                # Backtracking (mantém TTL original do nó)
                if len(path) > 1:
                    path.pop()
                    path_ttl.pop()
                    current_id = path[-1]
                else:
                    break
        
        return animation_steps

    # ---------- Algoritmos de busca ----------

    def search(
        self,
        node_id: str,
        resource_id: str,
        ttl: int,
        algo: str,
        seed: Optional[int] = None,
    ) -> Tuple[bool, int, int, List[str]]:
        """
        Retorna:
          found: bool
          msg_count: int (nº de mensagens trocadas)
          nodes_involved: int (nº distinto de nós que processaram a busca)
          path: caminho até o alvo (se encontrado)
        """
        if node_id not in self.nodes:
            raise ValueError(f"Nó de origem {node_id} não existe")

        if seed is not None:
            random.seed(seed)

        algo = algo.lower()
        if algo == "flooding":
            return self._search_flooding(node_id, resource_id, ttl, informed=False)
        elif algo == "informed_flooding":
            return self._search_flooding(node_id, resource_id, ttl, informed=True)
        elif algo == "random_walk":
            return self._search_random_walk(node_id, resource_id, ttl, informed=False)
        elif algo == "informed_random_walk":
            return self._search_random_walk(node_id, resource_id, ttl, informed=True)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algo}")

    def _search_flooding(
        self,
        start_id: str,
        resource_id: str,
        ttl: int,
        informed: bool,
    ) -> Tuple[bool, int, int, List[str]]:
        """
        Flooding com backtracking:
        - Envia para TODOS os vizinhos não visitados
        - TTL é decrementado a cada nível de profundidade
        - Usa BFS, então backtracking é implícito na estrutura
        """
        msg_count = 0
        visited = set()
        nodes_involved = set()
        # fila: (node_id, ttl_restante, path)
        queue = deque([(start_id, ttl, [start_id])])
        visited.add(start_id)
        nodes_involved.add(start_id)

        while queue:
            node_id, ttl_left, path = queue.popleft()
            node = self.nodes[node_id]

            # Verifica se o próprio nó tem o recurso
            if resource_id in node.resources:
                self._update_cache_on_hit(path, resource_id, node_id)
                return True, msg_count, len(nodes_involved), path

            # Se for "informado" e o nó souber quem tem o recurso
            if informed and resource_id in node.cache and node.cache[resource_id]:
                target_id = next(iter(node.cache[resource_id]))
                msg_count += 1
                path2 = path + [target_id]
                self._update_cache_on_hit(path2, resource_id, target_id)
                nodes_involved.add(target_id)
                return True, msg_count, len(nodes_involved), path2

            # Se TTL zerou, não propaga mais
            if ttl_left <= 0:
                continue

            # Envia para todos os vizinhos não visitados (flood)
            for neigh_id in node.neighbors:
                if neigh_id not in visited:
                    visited.add(neigh_id)
                    nodes_involved.add(neigh_id)
                    msg_count += 1
                    queue.append((neigh_id, ttl_left - 1, path + [neigh_id]))

        return False, msg_count, len(nodes_involved), []

    def _search_random_walk(
        self,
        start_id: str,
        resource_id: str,
        ttl: int,
        informed: bool,
    ) -> Tuple[bool, int, int, List[str]]:
        """
        Random Walk com backtracking:
        - A cada passo, escolhe UM vizinho aleatório não visitado
        - TTL representa quantos níveis de profundidade podemos explorar
        - Backtracking NÃO consome TTL (apenas volta no caminho)
        """
        msg_count = 0
        visited = set()  # Nós já visitados
        current_id = start_id
        path = [current_id]
        visited.add(current_id)
        # Cada nó no path tem um TTL associado
        path_ttl = [ttl]  # TTL disponível em cada nó do path

        while True:
            node = self.nodes[current_id]
            current_ttl = path_ttl[-1]

            # Verifica recurso local
            if resource_id in node.resources:
                self._update_cache_on_hit(path, resource_id, current_id)
                return True, msg_count, len(visited), path

            # Se for informado e souber alguém que possua o recurso
            if informed and resource_id in node.cache and node.cache[resource_id]:
                target_id = next(iter(node.cache[resource_id]))
                msg_count += 1
                path.append(target_id)
                visited.add(target_id)
                self._update_cache_on_hit(path, resource_id, target_id)
                return True, msg_count, len(visited), path

            # Encontra vizinhos não visitados
            unvisited_neighbors = [n for n in node.neighbors if n not in visited]

            if unvisited_neighbors and current_ttl > 0:
                # Escolhe vizinho aleatório não visitado
                next_id = random.choice(unvisited_neighbors)
                msg_count += 1
                current_id = next_id
                path.append(current_id)
                path_ttl.append(current_ttl - 1)  # Próximo nó tem TTL-1
                visited.add(current_id)
            else:
                # Backtracking: volta para o nó anterior (mantém TTL original)
                if len(path) > 1:
                    path.pop()
                    path_ttl.pop()
                    current_id = path[-1]
                    # Não incrementa msg_count - backtrack é gratuito
                else:
                    # Não há para onde voltar
                    break

        return False, msg_count, len(visited), []


def load_config(path: str) -> dict:
    """
    Espera um JSON no formato:
    {
      "num_nodes": 4,
      "min_neighbors": 1,
      "max_neighbors": 3,
      "resources": {
        "n1": ["r1", "r2"],
        "n2": ["r3"],
        "n3": ["r4"],
        "n4": ["r5"]
      },
      "edges": [
        ["n1", "n2"],
        ["n2", "n3"],
        ["n3", "n4"]
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Checagem simples de consistência
    if cfg["num_nodes"] != len(cfg["resources"]):
        raise ValueError("num_nodes diferente da quantidade de nós em resources")

    return cfg


def main():
    if len(sys.argv) < 2:
        print(
            "Uso: python p2p.py <config.json> [comando] [args...]\n"
            "\nComandos:\n"
            "  visualize                    - Exibe a topologia da rede\n"
            "  visualize <output.png>       - Salva a topologia em arquivo\n"
            "  search <node_id> <resource_id> <ttl> <algo> - Busca sem animação\n"
            "  animate <node_id> <resource_id> <ttl> <algo> - Busca com animação\n"
            "  animate <node_id> <resource_id> <ttl> <algo> <output.gif> - Salva animação\n"
            "  <node_id> <resource_id> <ttl> <algo> - Busca sem animação (atalho)\n"
            "\nAlgoritmos: flooding, informed_flooding, random_walk, informed_random_walk"
        )
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    net = P2PNetwork(config)

    if len(sys.argv) == 2 or sys.argv[2] == "visualize":
        # Visualização estática
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        net.visualize_network(save_path)
    
    elif sys.argv[2] == "search":
        # Busca sem animação
        if len(sys.argv) < 7:
            print("Uso: python p2p.py <config.json> search <node_id> <resource_id> <ttl> <algo>")
            sys.exit(1)
        
        node_id = sys.argv[3]
        resource_id = sys.argv[4]
        ttl = int(sys.argv[5])
        algo = sys.argv[6]
        
        found, msg_count, nodes_involved, path = net.search(
            node_id=node_id,
            resource_id=resource_id,
            ttl=ttl,
            algo=algo,
        )
        
        print(f"Encontrado: {found}")
        print(f"Mensagens trocadas: {msg_count}")
        print(f"Nós envolvidos: {nodes_involved}")
        if found:
            print(f"Caminho: {' -> '.join(path)}")
    
    elif sys.argv[2] == "animate":
        # Busca com animação
        if len(sys.argv) < 7:
            print("Uso: python p2p.py <config.json> animate <node_id> <resource_id> <ttl> <algo> [output.gif]")
            sys.exit(1)
        
        node_id = sys.argv[3]
        resource_id = sys.argv[4]
        ttl = int(sys.argv[5])
        algo = sys.argv[6]
        save_path = sys.argv[7] if len(sys.argv) > 7 else None
        
        found, msg_count, nodes_involved, path = net.visualize_search_animated(
            node_id=node_id,
            resource_id=resource_id,
            ttl=ttl,
            algo=algo,
            save_path=save_path
        )
        
        print(f"\nResultados:")
        print(f"Encontrado: {found}")
        print(f"Mensagens trocadas: {msg_count}")
        print(f"Nós envolvidos: {nodes_involved}")
        if found:
            print(f"Caminho: {' -> '.join(path)}")
    
    elif len(sys.argv) == 6:
        # Atalho: busca sem precisar escrever "search"
        # python p2p.py config.json <node_id> <resource_id> <ttl> <algo>
        node_id = sys.argv[2]
        resource_id = sys.argv[3]
        ttl = int(sys.argv[4])
        algo = sys.argv[5]
        
        found, msg_count, nodes_involved, path = net.search(
            node_id=node_id,
            resource_id=resource_id,
            ttl=ttl,
            algo=algo,
        )
        
        print(f"Encontrado: {found}")
        print(f"Mensagens trocadas: {msg_count}")
        print(f"Nós envolvidos: {nodes_involved}")
        if found:
            print(f"Caminho: {' -> '.join(path)}")
    
    else:
        print(f"Comando desconhecido: {sys.argv[2]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
