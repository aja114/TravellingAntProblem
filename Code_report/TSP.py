from functions import *
import numpy as np


class Node:
    """
    Defines a node of the graph
    Each node holds two list: the pheromone and visibility matrices
    Both assign to their neighbours a "score" of moving to one of them
    """
    def __init__(self, name, location):
        self.name = name
        self.location = location  # Physical location on the euclidean plane (x, y)
        self.neighbours = {}  # Dic of Nodes linked to self with their associated distance
        self.pheromone_table = {}
        self.visibility_table = {}

    def next_node(self, deterministic=False):
        """
        determines the most attractive
        neighbouring node to visit
        """
        if len(self.pheromone_table):
            if deterministic:
                return max([(self.pheromone_table[n], n) for n in self.neighbours], key=lambda x: x[0])[1]
            neighbours = [(n, p) for n, p in self.pheromone_table.items()]
            probabilities = [p for _, p in neighbours]
            return neighbours[fortune_wheel(probabilities)]
        return None

    def create_tables(self, neighbours):
        """
        create the pheromone and visibility tables
        initialise pheromone table to 1
        initialise visibility table to 1/(distance to neighbour)
        """
        self.neighbours = neighbours
        for neighbour, dist in self.neighbours.items():
            self.pheromone_table[neighbour] = 1 / dist
            self.visibility_table[neighbour] = 1 / dist

    def __lt__(self, other):
        return self.name < other.name  # just for display

    def __str__(self):
        return self.name + str(self.location)


class Link:
    """
    Defines a link in the network:
    Main components are the Node 1, Node 2, distance
    """
    def __init__(self, node1, node2):
        self.name = (node1, node2)
        self.node1 = node1
        self.node2 = node2
        self.distance = self.compute_distance()

    def compute_distance(self):
        """
        Keep the sqare distances as this won't change the results
        """
        n1 = self.node1.location
        n2 = self.node2.location
        d = np.sqrt((n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2)
        return d

    def __str__(self):
        return self.node1.name + "-" + self.node2.name + ": " + str(self.distance)


class Network:
    """	Defines a network as a graph """

    def __init__(self, Size=100, NbNodes=0, Networkfile_name=None):
        self.Nodenames = {}
        margin = 5 if Size > 20 else 0
        self.nodes = []
        self.links = []
        self.links_map = {}

        # Start with a random graph or load if from file
        if network_parameters['random_network'] and NbNodes > 1:
            self.nodes = [
                Node('N%d' % i, (random.randint(margin, Size - margin), random.randint(margin, Size - margin))) for i in
                range(NbNodes)]
        else:
            for Line in open(Networkfile_name, 'r', 1):
                NodeDef = Line.split()
                self.nodes.append(Node(NodeDef[0], tuple(map(int, NodeDef[1:]))))
                self.Nodenames[NodeDef[0]] = self.nodes[-1]

        # Create a complete weighted graph
        for n in self.nodes:
            for on in self.nodes:
                if on != n:
                    new_link = Link(n, on)
                    self.links.append(new_link)
                    self.links_map[(n, on)] = new_link

        self.size = len(self.nodes)

        # Creating routing tables
        for n in self.nodes:
            n.create_tables(self.neighbours(n))

        # for N in self.nodes:
        #     print(N.name)
        #
        # for _, L in self.links_map.items():
        #     print(str(L))

    def nodes(self):
        return self.nodes

    def neighbours(self, Node):
        return {n: self.links_map[(n, Node)].distance for n in self.nodes if (n, Node) in self.links_map}

    def update(self, path, distance):
        Q = algorithm_parameters["Q"]
        for i in range(len(path)-1):
            node_i = path[i]
            node_j = path[i+1]
            node_i.pheromone_table[node_j] += Q/distance

    def evaporate(self):
        pho = algorithm_parameters["pho"]
        for node in self.nodes:
            for ngb in node.pheromone_table:
                node.pheromone_table[ngb] *= (1-pho)


class Ant:
    """ Defines individual agents """
    def __init__(self, IdNb, InitialNode=None):
        self.ID = IdNb
        self.origin = InitialNode  # Ants have a birth place and keep it
        self.restart_ant()
        self.position = 0  # physical location between two nodes

    def restart_ant(self):
        """ initializes the ant """
        self.location = self.origin  # location in the network
        self.visited_nodes = set()
        self.visited_nodes.add(self.origin)
        self.path = [self.origin]
        self.traveled_distance = 0

    def next_node(self):
        " Looks for the next place to go "
        # The ant makes a biased choice of its next location
        # More probable links leading from neighbouring nodes to the ant's origin are more likely to be chosen
        neighbours = [ngb for ngb in self.location.neighbours if ngb not in self.visited_nodes]
        pheromones = np.array([self.location.pheromone_table[n] for n in neighbours])
        visibility = np.array([self.location.visibility_table[n] for n in neighbours])

        proba = self.comp_proba(pheromones, visibility)

        if len(proba) > 0:
            node_rank = fortune_wheel(proba)  # chooses a neighbour based on probabilities
            return neighbours[node_rank]
        return None

    def comp_proba(self, pheromones, visibility):
        alpha = algorithm_parameters["alpha"]
        beta = algorithm_parameters["beta"]
        total = np.sum(pheromones**alpha * visibility**beta)
        prob = (pheromones ** alpha * visibility ** beta) / total
        return prob

    def travel(self, graph):
        while self.visited_nodes != set(graph.nodes):
            new_location = self.next_node()
            self.visited_nodes.add(new_location)
            self.path.append(new_location)
            self.traveled_distance += graph.links_map[(self.location, new_location)].distance
            self.location = new_location

        self.path.append(self.origin)
        self.traveled_distance += graph.links_map[(self.path[-2], self.path[-1])].distance

        # print(self.path)
        # print(self.traveled_distance)

    def __str__(self):
        return self.ID + " - " + str(self.origin)


class Message():
    """ Messages travel through the network by following routing table deterministically """
    def __init__(self, start_node, Nodes):
        self.starting_node = start_node
        self.location = self.starting_node
        self.number_node = len(Nodes)
        self.traveled_distance = 0
        self.visited_nodes = set()
        self.visited_nodes.add(self.location)
        self.path = [self.location]

    def travel(self, graph):
        while len(self.visited_nodes) < graph.size:
            new_location = self.next_node()
            self.visited_nodes.add(new_location)
            self.path.append(new_location)
            self.traveled_distance += graph.links_map[(new_location, self.location)].distance
            self.location = new_location

        self.path.append(self.starting_node)
        self.traveled_distance += graph.links_map[(self.path[-2], self.path[-1])].distance

    def next_node(self):
        " Looks for the next place to go "
        # The ant makes a biased choice of its next location
        # More probable links leading from neighbouring nodes to the ant's origin are more likely to be chosen
        neighbours = [ngb for ngb in self.location.neighbours if ngb not in self.visited_nodes]
        pheromones = np.array([self.location.pheromone_table[n] for n in neighbours])
        visibility = np.array([self.location.visibility_table[n] for n in neighbours])
        proba = self.comp_proba(pheromones, visibility)
        return neighbours[int(np.argmax(proba))]

    def comp_proba(self, pheromones, visibility):
        alpha = algorithm_parameters["alpha"]
        beta = algorithm_parameters["beta"]
        total = np.sum(pheromones**alpha * visibility**beta)
        prob = (pheromones ** alpha * visibility ** beta) / total
        return prob


class AntPopulation:
    " defines the population of agents "
    def __init__(self, Nodes, pop_size):
        """ creates a population of ant agents """
        self.pop_size = pop_size

        # Create all the Ant agents
        self.pop = []
        for ID in range(self.pop_size):
            # Assigning a random initialisation node to the ant
            node = random.choice(Nodes)
            self.pop.append(Ant('A%d' % ID, InitialNode=node))

        self.shortest_distance = np.inf
        self.shortest_path = []

    def one_decision(self, graph):
        for ant in self.pop:
            ant.travel(graph)

            if ant.traveled_distance < self.shortest_distance:
                self.shortest_distance = ant.traveled_distance
                self.shortest_path = ant.path

        graph.evaporate()

        for ant in self.pop:
            graph.update(ant.path, ant.traveled_distance)
            ant.restart_ant()


if __name__ == "__main__":

    # Parameters
    population_parameters = {
            "pop_size": 500
    }

    network_parameters = {
            "graph_size": 100,
            "num_nodes": 50,
            "random_network": 1,
            "file_name": "Network1.ntw",
    }

    algorithm_parameters = {
            "num_iteration": 1000,
            "alpha": 1,
            "beta": 3,
            "Q": 1,
            "pho": 0.3
    }

    graph = Network(NbNodes=network_parameters['num_nodes'], Networkfile_name=network_parameters['file_name'])

    # Print the network
    # print_graph(graph)

    pop = AntPopulation(Nodes=graph.nodes, pop_size=population_parameters["pop_size"])

    message_distances = []
    start_node = random.choice(graph.nodes)

    for it in range(0, algorithm_parameters['num_iteration']):
        pop.one_decision(graph)

        if it % 20 == 0:
            message = Message(start_node=start_node, Nodes=graph.nodes)
            message.travel(graph)
            message_distances.append(message.traveled_distance)

            # for n in graph.nodes:
            #     print([(a.name, b) for a, b in n.pheromone_table.items()])
            #     print([(a.name, b) for a, b in n.visibility_table.items()])
            # print()
            # print()

    print(pop.shortest_distance)

    print_graph(graph, path=message.path, start_node=message.starting_node)

    print(message_distances)

    # RUN ANOTHER SOLVER FOUND ONLINE
    # solver([n.location for n in graph.nodes])


    # PRINT FINAL PROBABILITIES
    # for n in graph.nodes:
    #     print(n.name)
    #     neighb = [_ for _ in n.pheromone_table]
    #
    #     pheromones = np.array([n.pheromone_table[ngb] for ngb in neighb])
    #     visibility = np.array([n.visibility_table[ngb] for ngb in neighb])
    #
    #     alpha = algorithm_parameters["alpha"]
    #     beta = algorithm_parameters["beta"]
    #
    #     total = np.sum(pheromones**alpha * visibility**beta)
    #     prob = (pheromones ** alpha * visibility ** beta) / total
    #     print([a.name for a in neighb])
    #     print(list(prob))
