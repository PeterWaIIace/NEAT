import gymnasium as gym
import slimevolleygym
from neat import NEAT
import jax.numpy as jnp
import numpy as np
import argparse
import pickle 
import random
import time
import csv
import sys
import os

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--path','-p',type=str,default='',help="pass path to the directory")
parser.add_argument('--input_name', '-i', type=str, help='Input file path')
parser.add_argument('--epoch', '-e', type=int, help='Epoch to search for')
parser.add_argument('--id', '-id', type=int, help='net id to search for')

def load_params(path, name, epoch, id):
    csv_name = f"{path}/csv_params_{name}_e{epoch}"
    csv_reader = csv.reader(open(csv_name, 'r'))
    
    next(csv_reader)

    for row in csv_reader:
        # Convert each value to the appropriate type
        net_id, specienumber, fitness, connections, nodes, nmc, cmc, wmc, bmc, amc, C1, C2, C3, N = map(float, row)

        net_id = int(net_id)
        specienumber = int(specienumber)
        connections = int(connections)
        nodes = int(nodes)
        C1 = int(C1) 
        C2 = int(C2) 
        C3 = int(C3) 
        N = int(N)

        if net_id == id:
            break

    return {"net_id" : net_id,
            "specienumber" : specienumber, 
            "fitness" : fitness, 
            "connections" : connections, 
            "nodes" : nodes, 
            "NMC" : nmc, 
            "CMC" : cmc, 
            "WMC" : wmc, 
            "BMC" : bmc, 
            "AMC" : amc, 
            "C1" : C1, 
            "C2" : C2,
            "C3" : C3, 
            "N" : N}

def load_genome(path,name,epoch, net_id):
    # cgenome, ngenome = pickle.load(open(f"{path}/rest_{name}_{epoch}/{name}_e{epoch}_n{net_id}.neatpy",'rb'))
    # return cgenome, ngenome
    return f"{path}/rest_{name}_{epoch}/{name}_e{epoch}_n{net_id}.neatpy"

def load_session(path, name, epoch, net_id):
    params = load_params(path,name,epoch, net_id)
    genomes = load_genome(path,name,epoch, net_id)
    return genomes,params

def main():

    args = parser.parse_args()
    path = args.path
    input_file = args.input_name
    epoch = args.epoch
    id = args.id
    

    genomes, params = load_session(path,input_file,epoch,id)

    my_neat = NEAT(12,3,1,
                N = params["N"],
                nmc = params["NMC"],
                cmc = params["CMC"],
                wmc = params["WMC"],
                bmc = params["BMC"],
                amc = params["AMC"],
                δ_th = 0)

    my_neat.load_population(genomes)


    oldEnv = slimevolleygym.SlimeVolleyEnv()
    oldEnv.survival_bonus = True
    oldEnv.reset()
    env = gym.make("GymV21Environment-v0",
                    env=oldEnv,
                    apply_api_compatibility=True, 
                    render_mode="human")

    network = my_neat.evaluate()[0]

    observation = oldEnv.reset()
    observation = observation[0]

    score = 0

    done = False
    while not done:
        actions = network.activate(observation)
        actions = np.round(actions + 0.5).astype(int)

        observation, reward, done, info = oldEnv.step(actions)
        oldEnv.render()
        score += reward
        print(f"Score: {score}")

    env.close()

if __name__=="__main__":
    main()