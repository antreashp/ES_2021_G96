
from __future__ import print_function
import os
import neat
import visualize

# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment

from neat_feed_forward_controller import player_controller
experiment_name = 'meh'

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest")


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # genome.fitness = 4.0
        env.player_controller =player_controller(  neat.nn.FeedForwardNetwork.create(genome, config))

        f,p,e,t = env.play(pcont=genome)
        genome.fitness = f
        # print(f,p,e,t)
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
    winner = p.run(eval_genomes,1)

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