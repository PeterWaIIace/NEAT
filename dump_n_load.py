from neat import NEAT
import pickle 

N = 20
GENERATIONS = 100
POPULATION_SIZE = 20
NMC = 0.5
CMC = 0.5
WMC = 0.5
BMC = 0.5
AMC = 0.5
δ_th = 10
MUTATE_RATE = 16
RENDER_HUMAN = True

if __name__=="__main__":
    my_neat = NEAT(12,3,POPULATION_SIZE,
            nmc = 0.5,
            cmc = 0.5,
            wmc = 0.5,
            bmc = 0.5,
            amc = 0.5,
            N = N,
            δ_th = δ_th)


    pickle.dump(my_neat,open("neat.model","bw"))

    loaded_neat = pickle.load(open("neat.model","br"))

    