import neat
import os,sys
import pickle
import sys
sys.path.insert(0, 'evoman')
from environment import Environment

from tqdm import tqdm
from neat_feed_forward_controller import player_controller








if __name__ == '__main__':
    print('meh')
    with open('enemy'+str(enemy)+'randomini_test_1'+'/winner_genome.pkl', 'rb') as f:
        winner = pickle.load(f)

    print('Loaded genome:')
    
    

    fitnesses =[]
    for i in range(10):
        fitnesses.append( test(winner))
    

    


