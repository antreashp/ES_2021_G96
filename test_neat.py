import neat
import os,sys
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment

from tqdm import tqdm
from neat_feed_forward_controller import player_controller

headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

enemy = str(sys.argv[1]) if len(sys.argv)> 1 else 6 
# load the winner
experiment_name = 'test_enemy'+ str(enemy)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
env = Environment(experiment_name=experiment_name,
                  enemies=[int(enemy)],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="normal")



# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

def test(genome):
    print('hello')

    env.player_controller =player_controller(  neat.nn.FeedForwardNetwork.create(genome, config),write=True)

    f,p,e,t = env.play(pcont=genome)
    genome.fitness = f
    return f

if __name__ == '__main__':
    print('meh')
    with open('enemy'+str(enemy)+'/winner_genome.pkl', 'rb') as f:
        winner = pickle.load(f)

    print('Loaded genome:')
    
    

    fitnesses =[]
    for i in range(10):
        fitnesses.append( test(winner))
    

    


