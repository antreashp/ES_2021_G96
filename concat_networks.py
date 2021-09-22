import dgl
import torch
import neat
import os,sys
import pickle







local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


with open('enemy7/winner_genome.pkl', 'rb') as f:
    winner = pickle.load(f)
net7 = neat.nn.FeedForwardNetwork.create(winner, config)
# print(winner)


with open('enemy5/winner_genome.pkl', 'rb') as f:
    winner = pickle.load(f)
net5 = neat.nn.FeedForwardNetwork.create(winner, config)



print(net5.values)
print(net5.node_evals)
print(net5.)