import neat
import os,sys
import numpy as np
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
from neat_feed_forward_controller import gen_skip_player_controller

headless = False
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

enemy = str([2,5,8])
# load the winner
experiment_name = 'gen_experiments\\test_enemy'+ str(enemy)

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
# for enemy in enemies:
snets=[]
if enemy == 4 or enemy ==5:
    with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
        winner = pickle.load(f)
        snets.append(neat.nn.FeedForwardNetwork.create(winner, config))
if enemy == 2:
    with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
            winner = pickle.load(f)
            snets.append(neat.nn.FeedForwardNetwork.create(winner, config))
if enemy == 8:
    with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_87.pkl', 'rb') as f:
            winner = pickle.load(f)
            snets.append(neat.nn.FeedForwardNetwork.create(winner, config))

    



# Load the config file, which is assumed to live in
# the same directory as this script.

def make_box_plot(fitnesses):
    # pass
    # fitness = np.array(fitnesses[1])
    # spread = np.random.rand(50) * 100
    # print(spread.shape)
    # print(fitness.shape)
    # spread = fitness
    # center = np.median(fitness)
    # print(center)
    # flier_high = np.max(fitness)
    # flier_low = np.max(fitness)
    # data = np.concatenate((fitnesses[1], fitnesses[2]))
    fig1, ax1 = plt.subplots()
    ax1.set_title('Basic Plot')
    box_data = [fitnesses[2], fitnesses[5],fitnesses[8]]
    plt.title("NEAT Winner Phenotype's Performance")
    ax1.set_ylim(ymin=0,ymax=100)
    plt.ylabel("Fitness")
    ax1.boxplot(box_data,patch_artist=True,labels=['enemy 2','enemy 5','enemy 8'])
    plt.show()
def test(env,genome):
    # print('hello')

    env.player_controller =gen_skip_player_controller( snets,neat.nn.FeedForwardNetwork.create(genome, config))

    f,p,e,t = env.play(pcont=genome)
    genome.fitness = f
    print(f)
    return f

if __name__ == '__main__':
    print('meh')
    enemies = [2,5,8]
    # snets=[]
    fitnesses ={1:[],2:[],3:[],4:[],5:[],7:[],8:[]}
    gen_config_path = os.path.join(local_dir, 'config-feedforward_gen_skip.txt')
    gen_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         gen_config_path)
    print('Loaded genome:')
    with open('gen_experiments/enemies[2, 5, 8]_single'+'/my_winner_genome_88.pkl', 'rb') as f:
        winner = pickle.load(f)
    env = Environment(experiment_name=experiment_name,
                enemies=enemies,
                playermode="ai",
                multiplemode="yes",
                player_controller=gen_skip_player_controller,
                enemymode="static",
                level=2,
                speed="fastest",
                randomini='yes')    
    
    for i in range(2):
        fitnesses[i+1].append( test(env,winner))


    make_box_plot(fitnesses)


