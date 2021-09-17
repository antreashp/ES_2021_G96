
from __future__ import print_function
import os
import neat
import visualize
# imports framework
import os,sys
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'

import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from tqdm import tqdm
from neat_feed_forward_controller import player_controller

enemy = str(sys.argv[1]) if len(sys.argv[1])> 1 else 6 
experiment_name = 'enemy'+ str(enemy)

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
                  speed="fastest")

def evaluate(genome,config):
    env.player_controller =player_controller(  neat.nn.FeedForwardNetwork.create(genome, config))

    f,p,e,t = env.play(pcont=genome)
    genome.fitness = f

def eval_genomes(genomes, config):
    for genome_id, genome in tqdm(genomes):
        # genome.fitness = 4.0
        # print(f,p,e,t)
        env.player_controller =player_controller(  neat.nn.FeedForwardNetwork.create(genome, config))

        f,p,e,t = env.play(pcont=genome)
        genome.fitness = f
    
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
    p.add_reporter(neat.Checkpointer(5))

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

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=None)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)