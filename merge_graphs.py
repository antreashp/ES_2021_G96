import dgl
import torch
import neat
import os,sys
import pickle


from collections import defaultdict
# print(winner)

def get_graph_from_genome(genome):
    connections = [cg.key for cg in genome.connections.values() if cg.enabled]
    in_arr = []
    out_arr = []
    weights = defaultdict(float)
    for (inc,outc) in connections:
        # print(inc,outc)
        in_arr.append(inc)
        out_arr.append(outc)
        # exit()
        # print(genome.connections.keys())
        weights[inc] = genome.connections[(inc,outc)].weight    
        # weights.append( genome.connections[(inc,outc)].weight)    
        
    return in_arr,out_arr,weights



local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
with open('enemy7/winner_genome.pkl', 'rb') as f:
    winner = pickle.load(f)
net7 = neat.nn.FeedForwardNetwork.create(winner, config)
in_arr7,out_arr7,weights7 = get_graph_from_genome(winner)
# print(winner)


with open('enemy5/winner_genome.pkl', 'rb') as f:
    winner = pickle.load(f)
net5 = neat.nn.FeedForwardNetwork.create(winner, config)
in_arr5,out_arr5,weights5 = get_graph_from_genome(winner)
# print(net.input_nodes)
# print(net.values)
# print(net.node_evals)
# print(net.output_nodes)
# print(net.__dict__)
# print(net.values)

# print(winner.nodes)
# print(connections)
g = dgl.DGLGraph()
g.add_nodes(len(net5.input_nodes)+len(net5.output_nodes)+15)
print(len(net5.input_nodes)+len(net5.output_nodes)+15)
g.add_edges(in_arr5,out_arr5)
g.ndata['h'] = torch.tensor(list(weights5.values()))
g1 = dgl.DGLGraph()
g1.add_nodes(len(net7.input_nodes)+len(net7.output_nodes)+15)
g1.add_edges(in_arr7,out_arr7)
g1.ndata['h'] = weights7
large_g = dgl.batch([g, g1])
print(large_g.ndata)
print(large_g.batch_size)

print(large_g.batch_num_nodes)

# large_g.batch_num_edges