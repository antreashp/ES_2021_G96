
from __future__ import print_function
import os
import neat
import visualize
# imports framework
import os,sys
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from tqdm import tqdm
from neat_feed_forward_controller import player_controller

import statistics
import numpy as np


def eval_genomes(genomes, config):
    sum = 0
    max = -9999999
    c = 0
    fs  = []
    global max_f
    global max_g
    for genome_id, genome in tqdm(genomes):
        # genome.fitness = 4.0
        # print(f,p,e,t)
        env.player_controller =player_controller(  neat.nn.FeedForwardNetwork.create(genome, config))
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


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5,filename_prefix='enemy'+str(enemy) + str('/')))

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
    with open(experiment_name+'/'+'winner_genome.pkl','wb')as f:

        pickle.dump(winner,f)
        f.close()
    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=None,filename='enemy'+str(enemy) + str('/')+'model_graph.svg')
    # visualize.plot_stats(stats, ylog=False, view=True,filename='enemy'+str(enemy) + str('/')+'avg_fitness.svg')
    # visualize.plot_species(stats, view=True,filename='enemy'+str(enemy) + str('/')+'species.svg')

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    enemy = str(sys.argv[1]) 
        
    for i in range(10):
            
        experiment_name ='neat_enemy'+ str(enemy)+'multi'+str(i)
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"


        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
        env = Environment(experiment_name=experiment_name,
                        enemies=[enemy],
                        playermode="ai",
                        player_controller=player_controller,
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        randomini="yes" )

        global max_f 
        max_f = -9999
        global max_g 
        max_g = None
        if  os.path.exists(experiment_name+'/'+'my_log.txt'):
            os.remove(experiment_name+'/'+'my_log.txt')
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward.txt')
        run(config_path)