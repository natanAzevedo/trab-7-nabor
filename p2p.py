import json
import sys
import random
import time
from collections import deque, defaultdict
from typing import Dict, Set, List, Tuple, Optional

# Tenta importar bibliotecas gráficas (opcionais conforme requisitos)
try:
    import importlib
    nx = importlib.import_module("networkx")
    plt = importlib.import_module("matplotlib.pyplot")
    FuncAnimation = getattr(importlib.import_module("matplotlib.animation"), "FuncAnimation")
    VISUALIZATION_AVAILABLE = True
except Exception:
    # Se qualquer import falhar, desativa visualização
    nx = None
    plt = None
    FuncAnimation = None
    VISUALIZATION_AVAILABLE = False


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
        self.min_neighbors = config.get("min_neighbors", 1)
        self.max_neighbors = config.get("max_neighbors", float('inf'))

        # 1. Cria nós e atribui recursos
        for node_id, res_list in config["resources"].items():
            # Requisito: Não pode haver nós sem recursos (Requisito II.3)
            if not res_list:
                raise ValueError(f"Nó {node_id} não possui recursos.")
            self.nodes[node_id] = Node(node_id, set(res_list))

        # 2. Cria arestas
        for a, b in config["edges"]:
            if a not in self.nodes or b not in self.nodes:
                raise ValueError(f"Aresta refere-se a nó inexistente: {a}-{b}")
            if a == b:
                raise ValueError(f"Aresta de loop detectada em {a} (Requisito II.4)")
            self.nodes[a].add_neighbor(b)
            self.nodes[b].add_neighbor(a)

        # 3. Validações da Rede
        self._validate_degrees()
        self._validate_connected()

    def _validate_degrees(self):
        """Verifica se todos os nós respeitam min/max vizinhos (Requisito II.2)"""
        for node in self.nodes.values():
            deg = len(node.neighbors)
            if deg < self.min_neighbors or deg > self.max_neighbors:
                raise ValueError(
                    f"Nó {node.id} tem {deg} vizinhos. "
                    f"Permitido: [{self.min_neighbors}, {self.max_neighbors}]"
                )

    def _validate_connected(self):
        """Verifica se o grafo é conexo (Requisito II.1)"""
        if not self.nodes:
            return
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
            raise ValueError("A rede está particionada (não existe caminho entre todos os nós).")

    # ---------- Lógica de Cache (Busca Informada) ----------

    def _update_cache(self, path: List[str], resource_id: str, target_id: str):
        """
        Atualiza o cache de TODOS os nós no caminho percorrido pela resposta.
        Simula a mensagem de 'resource found' voltando pelo caminho reverso.
        """
        for node_id in path:
            self.nodes[node_id].cache[resource_id].add(target_id)

    # ---------- Algoritmos de Busca ----------

    def search(
        self,
        node_id: str,
        resource_id: str,
        ttl: int,
        algo: str,
        seed: Optional[int] = None,
    ) -> Tuple[bool, int, int, List[str]]:
        """
        Executa a busca e retorna (encontrou, total_msgs, nós_envolvidos, caminho).
        """
        if node_id not in self.nodes:
            raise ValueError(f"Nó de origem {node_id} não existe.")

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

    def _search_flooding(self, start_id: str, resource_id: str, ttl: int, informed: bool) -> Tuple[bool, int, int, List[str]]:
        msg_count = 0
        visited = set()     # Para evitar ciclos no flooding
        nodes_involved = set()
        
        # Fila: (nó_atual, ttl_restante, caminho_percorrido)
        queue = deque([(start_id, ttl, [start_id])])
        visited.add(start_id)
        nodes_involved.add(start_id)

        while queue:
            curr_id, curr_ttl, path = queue.popleft()
            node = self.nodes[curr_id]

            # 1. Verifica se o recurso está aqui
            if resource_id in node.resources:
                if informed:
                    self._update_cache(path, resource_id, curr_id)
                return True, msg_count, len(nodes_involved), path

            # 2. Verifica Cache (apenas se informed)
            if informed and resource_id in node.cache and node.cache[resource_id]:
                # Cache Hit! Sabemos onde está. 
                # Simplificação: assume envio direto ou roteamento eficiente até o alvo.
                target_id = next(iter(node.cache[resource_id]))
                msg_count += 1 # Mensagem direcionada
                final_path = path + [target_id]
                nodes_involved.add(target_id)
                # Reforça o cache no caminho
                self._update_cache(final_path, resource_id, target_id)
                return True, msg_count, len(nodes_involved), final_path

            # 3. Verifica TTL
            if curr_ttl <= 0:
                continue

            # 4. Propaga para vizinhos
            for neighbor in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    nodes_involved.add(neighbor)
                    msg_count += 1
                    queue.append((neighbor, curr_ttl - 1, path + [neighbor]))

        return False, msg_count, len(nodes_involved), []

    def _search_random_walk(self, start_id: str, resource_id: str, ttl: int, informed: bool) -> Tuple[bool, int, int, List[str]]:
        msg_count = 0
        visited_unique = set() # Apenas para métrica de "nós envolvidos"
        
        curr_id = start_id
        path = [curr_id]
        visited_unique.add(curr_id)
        
        while ttl >= 0:
            node = self.nodes[curr_id]

            # 1. Verifica recurso
            if resource_id in node.resources:
                if informed:
                    self._update_cache(path, resource_id, curr_id)
                return True, msg_count, len(visited_unique), path

            # 2. Verifica Cache
            if informed and resource_id in node.cache and node.cache[resource_id]:
                target_id = next(iter(node.cache[resource_id]))
                msg_count += 1
                path.append(target_id)
                visited_unique.add(target_id)
                self._update_cache(path, resource_id, target_id)
                return True, msg_count, len(visited_unique), path

            # 3. Verifica fim do TTL
            if ttl == 0:
                break

            # 4. Escolhe vizinho aleatório (Random Walk Puro)
            neighbors = list(node.neighbors)
            if not neighbors:
                break # Sem saída (embora a validação da rede deva impedir ilhas)

            next_id = random.choice(neighbors)
            
            msg_count += 1
            ttl -= 1
            curr_id = next_id
            path.append(curr_id)
            visited_unique.add(curr_id)

        return False, msg_count, len(visited_unique), []

    # ---------- Visualização e Animação ----------

    def visualize_network(self, save_path: Optional[str] = None):
        if not VISUALIZATION_AVAILABLE:
            print("Bibliotecas gráficas não instaladas (networkx/matplotlib).")
            return

        G = nx.Graph()
        for nid, node in self.nodes.items():
            res_str = "\n".join(node.resources)
            G.add_node(nid, label=f"{nid}\n[{res_str}]")
            for neigh in node.neighbors:
                G.add_edge(nid, neigh)

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=2000, edge_color='gray')
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Topologia da Rede P2P")
        if save_path:
            plt.savefig(save_path)
            print(f"Imagem salva em {save_path}")
        else:
            plt.show()
        plt.close()

    def animate_search(self, node_id, resource_id, ttl, algo, save_path=None):
        """
        Gera animação recriando o passo a passo da busca.
        Nota: Para simplicidade, re-executa uma lógica simplificada de rastreamento.
        """
        if not VISUALIZATION_AVAILABLE:
            print("Erro: Bibliotecas gráficas necessárias.")
            return

        # Executa uma versão modificada que guarda frames
        # Aqui, por brevidade, apenas executamos o search normal e mostramos o resultado estático final
        # (Implementar animação frame-a-frame completa requer duplicar toda lógica de busca com 'yield')
        print("Gerando animação baseada no caminho encontrado...")
        
        found, _, _, path = self.search(node_id, resource_id, ttl, algo)
        
        G = nx.Graph()
        for nid, node in self.nodes.items():
            G.add_node(nid)
            for neigh in node.neighbors:
                G.add_edge(nid, neigh)
        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(10, 8))

        def update(num):
            ax.clear()
            nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray', ax=ax)
            
            # Desenha caminho até o passo atual
            if len(path) > 0:
                current_path = path[:num+1]
                path_edges = list(zip(current_path, current_path[1:]))
                
                # Nós visitados
                nx.draw_networkx_nodes(G, pos, nodelist=current_path, node_color='yellow', ax=ax)
                # Nó atual (cabeça)
                nx.draw_networkx_nodes(G, pos, nodelist=[current_path[-1]], node_color='orange', ax=ax)
                # Arestas do caminho
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2, ax=ax)
                
                if num == len(path) - 1 and found:
                    nx.draw_networkx_nodes(G, pos, nodelist=[current_path[-1]], node_color='green', ax=ax)

            ax.set_title(f"Algoritmo: {algo} | Passo {num}/{len(path) if path else 0}")

        frames = len(path) if path else 1
        ani = FuncAnimation(fig, update, frames=frames, interval=800, repeat=False)
        
        if save_path:
            ani.save(save_path, writer='pillow')
            print(f"Animação salva em {save_path}")
        else:
            plt.show()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_result(found, msg, nodes, path):
    print(f"  -> Resultado: {'SUCESSO' if found else 'FALHA'}")
    print(f"  -> Mensagens: {msg}")
    print(f"  -> Nós envolvidos: {nodes}")
    print(f"  -> Caminho: {path}")

def run_shell(net: P2PNetwork):
    print("=== P2P Interactive Shell ===")
    print("Comandos: search <node> <res> <ttl> <algo>")
    print("          cache <node>  (para ver o cache atual do nó)")
    print("          exit")
    print("Algoritmos: flooding, informed_flooding, random_walk, informed_random_walk")
    
    while True:
        try:
            cmd_input = input("\np2p> ").strip().split()
            if not cmd_input: continue
            
            cmd = cmd_input[0].lower()
            
            if cmd == "exit":
                break
            
            elif cmd == "cache":
                if len(cmd_input) < 2:
                    print("Uso: cache <node_id>")
                    continue
                node = net.nodes.get(cmd_input[1])
                if node:
                    print(f"Cache do nó {node.id}: {dict(node.cache)}")
                else:
                    print("Nó não encontrado.")

            elif cmd == "search":
                # search n1 recurso 5 flooding
                if len(cmd_input) < 5:
                    print("Uso: search <src> <res> <ttl> <algo>")
                    continue
                
                src, res, ttl, algo = cmd_input[1], cmd_input[2], int(cmd_input[3]), cmd_input[4]
                found, msg, nodes, path = net.search(src, res, ttl, algo)
                print_result(found, msg, nodes, path)

            else:
                print("Comando desconhecido.")
        except Exception as e:
            print(f"Erro: {e}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python p2p.py <config.json> [comando]")
        print("Comandos: visualize, shell, animate ...")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        config = load_config(config_path)
        net = P2PNetwork(config)
        print("Rede carregada e validada com sucesso.")
    except Exception as e:
        print(f"Erro na configuração: {e}")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Se não passar comando, abre shell por padrão
        run_shell(net)
        return

    command = sys.argv[2]

    if command == "shell":
        run_shell(net)

    elif command == "visualize":
        save = sys.argv[3] if len(sys.argv) > 3 else None
        net.visualize_network(save)

    elif command == "search":
        # python p2p.py config.json search n1 res 5 algo
        if len(sys.argv) < 7:
            print("Argumentos insuficientes para search.")
            sys.exit(1)
        src, res, ttl, algo = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
        found, msg, nodes, path = net.search(src, res, ttl, algo)
        print_result(found, msg, nodes, path)

    elif command == "animate":
        if len(sys.argv) < 7:
            print("Argumentos insuficientes para animate.")
            sys.exit(1)
        src, res, ttl, algo = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
        save = sys.argv[7] if len(sys.argv) > 7 else None
        net.animate_search(src, res, ttl, algo, save)
        
    else:
        print(f"Comando '{command}' não reconhecido.")

if __name__ == "__main__":
    main()