
from __future__ import print_function
import os
import neat
import visualize
# imports framework

import statistics
from ast import literal_eval
import os,sys
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
import pickle
import sys
import numpy as np
sys.path.insert(0, 'evoman')
from environment import Environment
from tqdm import tqdm
from neat_feed_forward_controller import gen_skip_player_controller
# enemies = [4,7]
enemies =str( sys.argv[1])
enemies = literal_eval(enemies)
experiment_name = 'gen_experiments/enemies'+ str(enemies)+'_single'

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                config_path)
snets = []
for enemy in enemies:
    print(enemy)
    if enemy == 4 or enemy ==5:
            with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
                winner = pickle.load(f)
    if enemy == 2:
        with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
                winner = pickle.load(f)
    if enemy == 8:
        with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_87.pkl', 'rb') as f:
                winner = pickle.load(f)
    
    snets.append(neat.nn.FeedForwardNetwork.create(winner, config))
# print(winner)


# with open('enemy4/winner_genome.pkl', 'rb') as f:
#     winner = pickle.load(f)
# net4 = neat.nn.FeedForwardNetwork.create(winner, config)

# snets=[net4,net7]
print(enemies)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
env = Environment(experiment_name=experiment_name,
                  enemies=enemies,
                   multiplemode="yes",
                  playermode="ai",
                  player_controller=gen_skip_player_controller,
                  enemymode="static",
                  level=2,
                   randomini="yes" ,
                  speed="fastest")

# def evaluate(genome,config):
#     env.player_controller =gen_player_controller(  neat.nn.FeedForwardNetwork.create(genome, config))

#     f,p,e,t = env.play(pcont=genome)
#     genome.fitness = f

def eval_genomes(genomes, config):
    sum = 0
    max = -9999999
    c = 0
    
    global max_f
    global max_g
    fs  = []
    for genome_id, genome in tqdm(genomes):
        # genome.fitness = 4.0
        # print(f,p,e,t)
        env.player_controller =gen_skip_player_controller(snets ,neat.nn.FeedForwardNetwork.create(genome, config))
        c +=1
        f,p,e,t = env.play(pcont=genome)
        genome.fitness = f
        
        fs.append(f)
        sum += f
        if f>max:
            max = f
        if f > max_f:
            max_f = f
            max_g = genome
    mean = np.sum(fs)  / c
    std = statistics.stdev(fs)
    with open(experiment_name+'/'+'my_log.txt','a') as f:
        f.write('mean,'+str(mean)+',max,'+str(max)+',std,'+str(std)+'\r')
    # return f
    
    # return f


def run(gen_config_path):
    # Load configuration.

    gen_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         gen_config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(gen_config)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5,filename_prefix=experiment_name+'/enemy'+str(enemy) + str('/')))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes,int(sys.argv[2]) if len(sys.argv)>2 else 50 )

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    with open(experiment_name+str('/')+'winner_genome.pkl','wb')as f:

        pickle.dump(winner,f)
        f.close()
    with open(experiment_name+'/my_winner_genome_'+str(int(max_f))+'.pkl','wb')as f:

        pickle.dump(max_g,f)
        f.close()
    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=None,filename=experiment_name+ str('/')+'model_graph.svg')
    # visualize.plot_stats(stats, ylog=False, view=True,filename=experiment_name+ str('/')+'avg_fitness.png')
    # visualize.plot_species(stats, view=True,filename=experiment_name+str('/')+'species.png')

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    global max_f 
    max_f = -9999
    global max_g 
    max_g = None

    gen_config_path = os.path.join(local_dir, 'config-feedforward_gen_skip.txt')
    run(gen_config_path)