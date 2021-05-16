import networkx as nx
import matplotlib.pyplot as plt
import random, numpy, math, copy


def fortune_wheel(proba):
    " draws one one the pie shares y picking a location uniformly "
    if proba == []:
        ValueError('Calling Fortune Wheel with no probabilities')
    Lottery = random.uniform(0, sum(proba))
    P = 0  # cumulative probability
    for p in enumerate(proba):
        P += p[1]
        if P >= Lottery:
            break
    return p[0]


def print_graph(graph, path=[], start_node=None):
    G = nx.Graph()

    # Create the graph
    for g in graph.nodes:
        G.add_node(g.name, pos=g.location)

    for l in graph.links:
        G.add_edge(l.node1.name, l.node2.name)

    # Plot
    options = {"color": "b", "node_size": 30, "alpha": 0.8}
    pos = nx.get_node_attributes(G, 'pos')

    if start_node:
        nx.draw_networkx_nodes(G, pos=pos, **options)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[start_node.name], node_color="r", **options)
    else:
        nx.draw_networkx_nodes(G, pos=pos, **options)

    nx.draw_networkx_labels(G, pos, font_size=8)

    if len(path) == 0:
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    else:
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        edgelist = []

        for i in range(len(path)-1):
            n1 = path[i].name
            n2 = path[i+1].name
            edgelist.append((n1, n2))

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=3,
            alpha=0.5,
            edge_color="r",
        )

    plt.show()


def solver(cities):
    l = len(cities)

    tour = random.sample(range(l), l);

    for temperature in numpy.logspace(0, 5, num=100000)[::-1]:
        [i, j] = sorted(random.sample(range(l), 2));
        newTour = tour[:i] + tour[j:j + 1] + tour[i + 1:j] + tour[i:i + 1] + tour[j + 1:];
        if math.exp((sum(
                [math.sqrt(sum([(cities[tour[(k + 1) % l]][d] - cities[tour[k % l]][d]) ** 2 for d in [0, 1]])) for k in
                 [j, j - 1, i, i - 1]]) - sum(
            [math.sqrt(sum([(cities[newTour[(k + 1) % l]][d] - cities[newTour[k % l]][d]) ** 2 for d in [0, 1]])) for
             k in [j, j - 1, i, i - 1]])) / temperature) > random.random():
            tour = copy.copy(newTour);

    path = [numpy.array([cities[tour[i % l]][0], cities[tour[i % l]][1]]) for i in range(l)]
    print(path)
    length = 0
    for i in range(len(path)-1):
        length += numpy.linalg.norm(path[i+1]-path[i])

    print("Solver 2 length: ", length)

    plt.plot([cities[tour[i % l]][0] for i in range(l)], [cities[tour[i % l]][1] for i in range(l)], 'xb-');
    plt.show()







