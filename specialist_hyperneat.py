import neat 
import neat.nn
try:
   import cPickle as pickle
except:
   import pickle
import sys,os
sys.path.append('../pureples/')
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net,draw_es
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import numpy as np
import statistics
sys.path.insert(0, 'evoman')
from environment import Environment

from tqdm import tqdm
from neat_feed_forward_controller import player_controller
enemy = str(sys.argv[1]) 
print(enemy,'meeeh')
experiment_name = 'hyper_enemy_again'+ str(enemy)
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
input_coordinates  = [(-3.0 , 4.0),(-3.0 , 3.0),(-3.0 , 2.0),(-3.0 , 1.0),
                      (-3.0 , -1.0),(-3.0 , -2.0),(-3.0 , -3.0),(-3.0 , -4.0),
                      (3.0 , 4.0),(3.0 , 3.0),(3.0 , 2.0),(3.0 , 1.0),
                      (3.0 , -1.0),(3.0 , -2.0),(3.0 , -3.0),(3.0 , -4.0),
                      (0.0 , 4.0),(0.0 , 2.0),(0.0 , -4.0),(0.0 , -2.0)]

output_coordinates = [(-2.0, 5.0),(-1.0, 5.0),(0.0, 5.0),(1.0, 5.0),(2.0, 5.0),]

 
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest")

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

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn')


def eval_fitness(genomes, config):
    sum = 0
    max = -9999999
    c = 0
    global max_f
    global max_g
    fs  = []
    for idx, genome in tqdm(genomes):
        
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
        c +=1
        f,p,e,t = env.play(pcont=genome)
        genome.fitness = f
        fs.append(f)
        sum += f
        # exit()
        if f>max:
            max = f
        if f > max_f:
            max_f = f
            max_g = genome
    mean = np.sum(fs)  / c
    std = statistics.stdev(fs)
    with open(experiment_name+'/'+'my_log.txt','a') as f:
        f.write('mean,'+str(mean)+',max,'+str(max)+',std,'+str(std)+'\r')
# Create the population and run the XOR task by providing the above fitness function.
def run(gens):
    pop = neat.population.Population(config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(eval_fitness, gens)
    print("es_hyperneat_xor_medium done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    global max_f 
    max_f = -9999
    global max_g 
    max_g = None
    winner = run(int(sys.argv[2]) if len(sys.argv)>2 else 50)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    winner_net = network.create_phenotype_network(filename=experiment_name+'/es_hyperneat_xor_medium_winner.png')  # This will also draw winner_net.
    with open(experiment_name+'/winner_genome.pkl','wb')as f:

        pickle.dump(winner,f)
        f.close()
    with open(experiment_name+'/my_winner_genome_'+str(int(max_f))+'.pkl','wb')as f:

        pickle.dump(max_g,f)
        f.close()
    # Save CPPN if wished reused and draw it to file.
    draw_net(cppn, filename=experiment_name+"/es_hyperneat_xor_medium_cppn.png")
    with open(experiment_name+'/es_hyperneat_xor_medium_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
