import neat
import os,sys
import numpy as np
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
import matplotlib.pyplot as plt
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




# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
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

    env.player_controller =player_controller(  neat.nn.FeedForwardNetwork.create(genome, config),write=True)

    f,p,e,t = env.play(pcont=genome)
    genome.fitness = f
    print(f)
    return f

if __name__ == '__main__':
    print('meh')
    enemies = [2,5,8]

    fitnesses ={1:[],2:[],3:[],4:[],5:[],7:[],8:[]}
    
    for enemy in enemies:
        if enemy == 4 or enemy ==5:
            with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
                winner = pickle.load(f)
        if enemy == 2:
            with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
                    winner = pickle.load(f)
        if enemy == 8:
            with open('neat_enemy_again'+str(enemy)+'/my_winner_genome_87.pkl', 'rb') as f:
                    winner = pickle.load(f)
        
        print('Loaded genome:')
        env = Environment(experiment_name=experiment_name,
                  enemies=[int(enemy)],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini='yes')    
        
        for i in range(2):
            fitnesses[enemy].append( test(env,winner))
    

    make_box_plot(fitnesses)


