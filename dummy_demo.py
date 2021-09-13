################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################
import multiprocessing as mp
# imports framework
import numpy as np
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
from demo_controller import player_controller

import threading
import multiprocessing as mp
from multiprocessing.pool import ThreadPool 
lock = threading.Lock()
    
def simulation(env,x):
    # with lock:
    f,p,e,t = env.play(pcont=x)
    return f
if __name__=='__main__':

    max_cpu = 24
    
    pool = ThreadPool(max_cpu)
    def evaluate(x):
        # with lock:
        # results = pool.starmap(simulation, [(env,row) for env,row in zip(envs,x[:20])])
        # print(results)
        # exit()
        # for i in range():
        #     pass
        # print(x.shape)
        results = []
        for batch in range(0,x.shape[0]-max_cpu,max_cpu): 
            # for i in range(max_cpu):

            results.extend( pool.starmap(simulation, [(env,x_) for env,x_ in zip(envs,x[batch:batch+max_cpu])]))
            # print(meh)
        # pool.close()
        
        # exit()
        # print(results.get(timeout=1))
        # print([res.get(timeout=1) for res in results])
        return results
        return np.array(list(map(lambda y: simulation(env,y), x)))

    experiment_name = 'dummy_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes environment with ai player using random controller, playing against static enemy
    envs =[Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(10),
                    enemymode="static",
                    level=2,
                    speed="fastest") for i in range(max_cpu)] 
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(10),
                    enemymode="static",
                    level=2,
                    speed="fastest")

    print("Number of processors: ", mp.cpu_count())

    n_vars = (envs[0].get_num_sensors()+1)*10 + (10+1)*5


    pop_size = 100
    pop = np.random.uniform(-1,1, (pop_size, n_vars))
    import time
    start = time.time()
    fit_pop = evaluate(pop)
    print(time.time()- start)

