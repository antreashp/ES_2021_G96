import neat
import os,sys
import numpy as np
import pickle
import sys
import graphviz 
sys.path.insert(0, 'evoman')
from environment import Environment
import matplotlib.pyplot as plt
from tqdm import tqdm
from neat_feed_forward_controller import player_controller
from cairosvg import svg2png
sys.path.append('../pureples/')
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net,draw_es
from pureples.es_hyperneat.es_hyperneat import ESNetwork
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

config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn')


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
    plt.title("Es-HyperNEAT Winner Phenotype's Performance")
    ax1.set_ylim(ymin=-10,ymax=100)
    plt.ylabel("Fitness")
    ax1.boxplot(box_data,patch_artist=True,labels=['enemy 2','enemy 5','enemy 8'])
    plt.show()
def make_net_plot(enemy,experiment_name):
    
    earth   = plt.imread('hyper_enemy_again'+str(enemy)+'/es_meh')
    fig, ax = plt.subplots()
    im      = ax.imshow(earth)
    plt.show()



def test(env,genome):
    # print('hello')

    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        # plot = draw_net(cppn, filename=experiment_name+"/es_hyperneat_xor_medium_cppn")
        # plot.view()
    network = ESNetwork(sub, cppn, params)
    # if c %10 ==0:

    #     net = network.create_phenotype_network(filename=experiment_name+'/substrate.jpg')
    # else:
    net = network.create_phenotype_network()
        
    # draw_es(id_to_coords, network.connections, experiment_name+'/substrate')
    env.player_controller =player_controller(  net)
    f,p,e,t = env.play(pcont=genome)
    print(f)
    return f,cppn

if __name__ == '__main__':
    print('meh')
    input_coordinates  = [(-3.0 , 4.0),(-3.0 , 3.0),(-3.0 , 2.0),(-3.0 , 1.0),
                      (-3.0 , -1.0),(-3.0 , -2.0),(-3.0 , -3.0),(-3.0 , -4.0),
                      (3.0 , 4.0),(3.0 , 3.0),(3.0 , 2.0),(3.0 , 1.0),
                      (3.0 , -1.0),(3.0 , -2.0),(3.0 , -3.0),(3.0 , -4.0),
                      (0.0 , 4.0),(0.0 , 2.0),(0.0 , -4.0),(0.0 , -2.0)]

    output_coordinates = [(-2.0, 5.0),(-1.0, 5.0),(0.0, 5.0),(1.0, 5.0),(2.0, 5.0),]
    enemies = [5]
    
    sub = Substrate(input_coordinates, output_coordinates)
    
    # ES-HyperNEAT specific parameters.
    params = {"initial_depth": 1,
            "max_depth": 2,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 8.0,
            "activation": "sigmoid"}
    fitnesses ={1:[],2:[],3:[],4:[],5:[],7:[],8:[]}
        
    for enemy in enemies:
        if enemy == 4 or enemy ==5:
            with open('hyper_enemy_again'+str(enemy)+'/my_winner_genome_90.pkl', 'rb') as f:
                winner = pickle.load(f)
        if enemy == 2:
            with open('hyper_enemy_again'+str(enemy)+'/my_winner_genome_92.pkl', 'rb') as f:
                    winner = pickle.load(f)
        if enemy == 8:
            with open('hyper_enemy_again'+str(enemy)+'/my_winner_genome_86.pkl', 'rb') as f:
                    winner = pickle.load(f)
                
            # print(winner)
        env = Environment(experiment_name=experiment_name,
                  enemies=[int(enemy)],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="normal",
                  randomini='yes')
        print('Loaded genome:')
        
        
        for i in range(25):
            f,cppn = test(env,winner)
            
            fitnesses[enemy].append( f)
        # print(cppn)
        # meh = draw_net(cppn, filename='hyper_enemy_again'+str(enemy)+'/es_meh_test')
        # # print(meh)
        # node_attrs = {
        # 'shape': 'circle',
        # 'fontsize': '9',
        # 'height': '0.2',
        # 'width': '0.2'}
        # print(meh.format())
        # meh.render()
        # dot = graphviz.Digraph('svg', node_attr=node_attrs)
        # dot.render(meh)

        # svg2png(bytestring=meh,write_to='output.png')
        # plt.imshow(meh.)
        # make_net_plot(enemy,experiment_name)
    make_box_plot(fitnesses)


