''' Fast neat '''
import jax
import copy
import time
import pickle
import random
import jax.numpy as jnp
import jax.random as jrnd
import gymnasium as gym

from enum import Enum

class NodeTypes(Enum):
    NODE   = 1
    INPUT  = 2
    OUTPUT = 3

class StatefulRandomGenerator:

    def __init__(self):
        self.key = jrnd.PRNGKey(0)

    def randint_permutations(self,val_range):
        ''' generate subarray of permutation array from defined values range '''
        rnd_value = jrnd.uniform(self.key)
        rnd_value_int = int(rnd_value * val_range)
        indecies = jrnd.permutation(self.key, val_range)[:rnd_value_int]
        self.key = jrnd.split(self.key,1)[0]
        return indecies

    def randint(self,max=100,min=0):
        rnd_value = jrnd.randint(self.key, shape=(1,), minval=min, maxval=max)
        self.key = jrnd.split(self.key,1)[0]
        return rnd_value[0]

    def uniform(self,max=1.0,min=0):
        random_float = jrnd.uniform(self.key, shape=(1,), minval=min, maxval=max)
        self.key = jrnd.split(self.key,1)[0]
        return random_float[0]
        

Rnd = StatefulRandomGenerator()

NUMBER_OF_ACTIATION_FUNCTIONS = 1
# def sigmoid(x):
#     """Sigmoid activation function"""
#     

def __x(x):
    return x

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def activation_func(x,y):

    return jax.lax.cond(y == 0, lambda x : x, lambda x : sigmoid(x),(x))  # Handle other cases (NaN for example)



class Genome:

    i = 2
    o = 3
    w = 4
    enabled = 5

    n_bias = 2
    n_act  = 3

    def __init__(self):
        self.connections_length = 20
        self.nodes_length = 20
        self.max_innov = 0
        self.index = 0
        self.fitness = 1.0
        # helper indicies names
        # Connections genomes is array, rows are genomes, but cols are parameters of that genomes
        self.con_gen  = jnp.zeros((self.connections_length,6),)
        self.node_gen = jnp.zeros((self.nodes_length,4),)

    def new_fitness(self,fit : float):
        ''' Assign new fitness to this genome '''
        self.fitness = fit

    # TODO: there is problem with correct nodes
    def add_node(self,index : int, type : NodeTypes, bias : float, act : int):
        ''' Adding node '''
        
        if self.nodes_length <= index:
            self.nodes_length += 20
            new_nodes_spaces = jnp.zeros((20,4),)
            self.node_gen = jnp.concatenate((self.node_gen,new_nodes_spaces), axis=0)

        self.node_gen = self.node_gen.at[index].set(jnp.array([index,type.value,bias,act]))

    def add_r_connection(self,innov):
        active_nodes = self.node_gen[self.node_gen[:,0] != 0]
        possible_input_nodes  = active_nodes[active_nodes[:,1] != float(NodeTypes.OUTPUT.value)][:,0]
        possible_output_nodes = active_nodes[active_nodes[:,1] != float(NodeTypes.INPUT.value)][:,0]

        in_node  = possible_input_nodes[Rnd.randint(max=len(possible_input_nodes))]
        out_node = possible_output_nodes[Rnd.randint(max=len(possible_output_nodes))]
        return self.add_connection(int(innov),int(in_node),int(out_node),1.0)

    def add_r_node(self,innov):
        exisitng_connections = self.con_gen[self.con_gen[:,0] != 0]

        existing_connection = exisitng_connections[Rnd.randint(max=len(exisitng_connections))]
        index_of_connection = int(existing_connection[0]) - 1
        self.con_gen.at[index_of_connection,self.enabled].set(0.0)

        new_node = self.node_gen[self.node_gen[:,0] != 0][-1,0] + 1

        innov+=1
        self.add_connection(int(innov),
                            int(self.con_gen[index_of_connection,self.i]),
                            int(new_node),
                            self.con_gen[index_of_connection,self.w]
                        )
        innov+=1
        self.add_connection(int(innov),
                            int(new_node),
                            int(self.con_gen[index_of_connection,self.o]),
                            self.con_gen[index_of_connection,self.w]
                        )
       
        return innov

    def change_weigth(self,weigth):
        exisitng_connections = self.con_gen[self.con_gen[:,0] != 0]
        existing_connection = exisitng_connections[Rnd.randint(max=len(exisitng_connections))]
        index_of_connection = int(existing_connection[0]) - 1
        self.con_gen.at[index_of_connection,self.w].set(weigth)

    def change_bias(self,bias):
        existing_nodes = self.node_gen[self.node_gen[:,0] != 0]
        existing_node = existing_nodes[Rnd.randint(max=len(existing_nodes))]
        node_index = int(existing_node[0]) - 1
        self.con_gen.at[node_index,self.n_bias].set(bias)

    def change_activation(self,act):
        existing_nodes = self.node_gen[self.node_gen[:,0] != 0]
        existing_node = existing_nodes[Rnd.randint(max=len(existing_nodes))]
        node_index = int(existing_node[0]) - 1
        self.con_gen.at[int(node_index),self.n_act].set(act)

    def add_connection(self,innov : int,in_node : int, out_node : int, weight : float):
        ''' Adding connection '''

        # update innovation if is bigger than current innov of genome
        if self.max_innov < innov:
            self.max_innov = innov

        if self.connections_length <= innov:
            self.nodes_length += 20
            new_connections_spaces = jnp.zeros((20,6),)
            self.con_gen = jnp.concatenate((self.con_gen,new_connections_spaces), axis=0)

        if self.node_gen[int(in_node)][0] == 0:
            self.add_node(in_node,NodeTypes.NODE,0.0,1)

        if self.node_gen[int(out_node)][0] == 0:
            self.add_node(out_node,NodeTypes.NODE,0.0,1)

        self.con_gen = self.con_gen.at[innov].set(jnp.array([innov,innov,in_node,out_node,weight,1.0]))
        return innov

# assuming that population is class having con_gens of size N_POPULATION x CREATURE_GENES x All information
# For example 50 x 10 x 6 
# this could be done much faster
def δ(genome_1, genome_2, c1=1.0, c2=1.0, c3=1.0, N=1.0):
    ''' calculate compatibility between genomes '''
    D = 0  # disjoint genes
    E = 0  # excess genes

    # check smaller innovation number:
    innovation_thresh = genome_1.max_innov if genome_1.max_innov < genome_2.max_innov else genome_2.max_innov

    # Step 1: Determine the sizes of the first dimension
    size1 = genome_1.con_gen.shape[0]
    size2 = genome_2.con_gen.shape[0]

    # Step 2: Find the maximum size
    max_size = max(size1, size2)

    # Step 3: Pad the smaller array
    if size1 < max_size:
        padding = ((0, max_size - size1), (0, 0))  # Pad the first dimension
        genome_1.con_gen = jnp.pad(genome_1.con_gen, padding)
    elif size2 < max_size:
        padding = ((0, max_size - size2), (0, 0))  # Pad the first dimension
        genome_2.con_gen = jnp.pad(genome_2.con_gen, padding)

    D_tmp = jnp.subtract(
        genome_1.con_gen[:innovation_thresh,0]
        ,genome_2.con_gen[:innovation_thresh,0]
    ) # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    D = len(D_tmp[D_tmp != 0])

    E_tmp = jnp.subtract(
        genome_1.con_gen[innovation_thresh:,0]
        ,genome_2.con_gen[innovation_thresh:,0]
    ) # getting all disjoint connections - if array do not have connection then it will have 0 as innovtaion number
    E = len(E_tmp[E_tmp != 0])

    W_1 = jnp.sum(genome_1.con_gen[:innovation_thresh,:][D_tmp == 0][genome_1.w])
    W_2 = jnp.sum(genome_2.con_gen[:innovation_thresh,:][D_tmp == 0][genome_2.w])
    W_avg = jnp.subtract(W_1,W_2)/W_1.size

    d = abs((c1 * E) / N + (c2 * D) / N + c3 * W_avg)
    return d

def sh(δ,δ_t = 0.2):
    ''' sharing fitness threshold function '''
    return δ < δ_t

# population can be matrix of shapex N_SIZE_POP x 
def speciate(population) -> list:
    """function for speciation"""

    δ_th = 5
    species = [[population[0]]]

    for _,individual_2 in enumerate(population):
        if individual_2 is not population[0]:
            if sh(δ(population[0],individual_2),δ_th):
                species[len(species) - 1].append(individual_2)

    for i,individual_1 in enumerate(population):
        # if not in current species, create new specie
        if sum([individual_1 in specie for specie in species]) == 0:
            species.append([individual_1])
            for _,individual_2 in enumerate(population):
                if sum([individual_2 in specie for specie in species]) == 0:
                    if sh(δ(individual_1,individual_2),δ_th):
                        species[len(species) - 1].append(individual_2)

    print(f"[DEBUG] Number of species: {len(species)}, if same as population number then bad.")
    print(f"[DEBUG] Population number {len(population)}")
    return species

def mate(superior : Genome, inferior : Genome):
    ''' mate superior Genome with inferior Genome '''
    # check smaller innovation number:
    innovation_thresh = superior.max_innov if superior.max_innov < inferior.max_innov else inferior.max_innov

    offspring = copy.deepcopy(inferior)

    indecies = Rnd.randint_permutations(innovation_thresh)
    offspring.con_gen = offspring.con_gen.at[indecies].set(superior.con_gen[indecies])

    indecies = Rnd.randint_permutations(len(inferior.con_gen[innovation_thresh:])) + innovation_thresh
    offspring.con_gen = offspring.con_gen.at[indecies].set(superior.con_gen[indecies])
    # Lazy but working, copy all nodes not existing in inferior but exisitng in superior
    offspring.node_gen = offspring.node_gen.at[inferior.node_gen[:,0] == 0].set(superior.node_gen[inferior.node_gen[:,0] == 0])

    return offspring

# this can be done better on arrays
def cross_over(population,keep_top = 2):
    ''' cross over your population '''
    new_population = []
    species = speciate(population)
    
    for specie in species:
        sorted_specie = sorted(specie, key=lambda x: x.fitness, reverse=True)[:]
        top_species = sorted_specie[:keep_top]

        for keept in top_species:
            new_population.append(keept)

        for n in range(len(sorted_specie) - keep_top):
            n = random.randint(0,len(top_species)-1)
            m = n
            while m == n:
                m = random.randint(0,len(top_species)-1)

            offspring = mate(top_species[n],top_species[m])
            new_population.append(offspring)

    return new_population

def random_mutate(population,innov = 0):
    for n,_ in enumerate(population):

        mutation_chance = random.randint(0,10)
        if mutation_chance > 7:
            innov = population[n].add_r_node(innov)

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            innov = population[n].add_r_connection(innov)

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            population[n].change_weigth(Rnd.uniform())

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            population[n].change_activation(Rnd.randint(NUMBER_OF_ACTIATION_FUNCTIONS,0))

        mutation_chance = random.randint(0,10)
        if mutation_chance  > 7:
            population[n].change_bias(Rnd.uniform())

    return innov

def evolve(population,innov):
    population = cross_over(population)
    innov = random_mutate(population,innov)
    return (population, innov)

def compiler(ngenomes, cgenomes):
    # I need to make sure that all output neurons are at the same layer
    neurons = []
    active_nodes = ngenomes[ngenomes[:,0] != 0.0]
    for n,node in enumerate(active_nodes):
        neurons.append(
            Neuron(node)
        )
        # neurons[n].inputs  = cgenomes[cgenomes[:,Genome.o] == n][:,Genome.i]
        # neurons[n].weights = cgenomes[cgenomes[:,Genome.o] == n][:,Genome.i]

    for c in cgenomes[cgenomes[:,Genome.enabled] != 0.0]:
        print(f"neurons len: {len(neurons)}, index: {int(c[Genome.o])-1}")
        neurons[int(c[Genome.o])-1].add_input(
            int(c[Genome.i])-1,
            c[Genome.w],
            neurons[int(c[Genome.i])-1].layer)

    ff = FeedForward()
    for neuron in neurons:
        ff.add_neuron(neuron)

    ff.compile()
    return ff

class Neuron:

    def __init__(self,node_genome):
        self.index = int(node_genome[0]) - 1
        self.layer = 0
        self.type = int(node_genome[1])
        self.bias = node_genome[Genome.n_bias]
        self.act  = node_genome[Genome.n_act]
        self.input_list = []
        self.weights = []
        
    def add_input(self,in_neuron,weigth,layer):
        self.input_list.append(in_neuron)
        self.weights.append(weigth)
        if layer >= self.layer:
            self.layer = layer + 1

    def getLayer(self):
        return self.layer

    def get(self):
        return {self.layer : jnp.array(self.weights)}

    def print(self):
        print(
        f"self.index:     {self.index}\
          self.layer:     {self.layer}\
          self.input_list {self.input_list}\
          self.weights    {self.weights}"
        )

class Layer:
    # TODO: distinguish between output layers and rest

    def __init__(self,index):
        self.layer_index = index
        self.neurons = []

        self.weigths = None
        self.inputs  = None
        self.outputs = None
        self.biases = None
        self.acts = None

        self.inputs_len  = 0
        self.outputs_len = 0

    def add_neuron(self,neuron):
        self.outputs_len += 1
        if len(neuron.input_list) > 0:
            self.inputs_len = jnp.max(jnp.array(neuron.input_list))+1
        self.neurons.append(neuron)
        # neuron.

    def compile(self):

        self.weigths = jnp.zeros((self.outputs_len,self.inputs_len),dtype = jnp.float32)
        self.inputs  = jnp.zeros((self.inputs_len),dtype = jnp.int32)
        self.outputs = jnp.zeros((self.outputs_len),dtype = jnp.int32)
        self.acts   = jnp.zeros((self.outputs_len),dtype = jnp.int32)
        self.biases = jnp.zeros((self.outputs_len),dtype = jnp.float32)

        filled_in_length = 0
        for n_i,n in enumerate(self.neurons):
            if len(self.inputs) > 0:
                self.inputs = self.inputs.at[jnp.array(n.input_list)].set(jnp.array(n.input_list))
                filled_in_length += len(n.input_list)

                self.outputs = self.outputs.at[n_i].set(n.index)
                self.biases = self.biases.at[n_i].set(n.bias)
                self.acts = self.acts.at[n_i].set(n.act)

                self.weigths = self.weigths.at[n_i,n.input_list].set(jnp.array(n.weights))  
        # now layer is complied

    def forward(self,input):
        X = jnp.dot(input[self.inputs],self.weigths.T) + self.biases
        return jax.vmap(activation_func)(X,self.acts)

class FeedForward:

    def __init__(self):
        self.index = 0
        self.size = 0
        self.layers = [Layer(self.index)]
        self.output_layer = Layer(999)

    def add_neuron(self,neuron):
        self.size += 1
        if neuron.type == NodeTypes.OUTPUT.value:
            self.output_layer.add_neuron(neuron)
        else:
            layer_index = neuron.getLayer()

            while layer_index >= len(self.layers):
                self.index += 1
                self.layers.append(Layer(self.index))

            self.layers[layer_index].add_neuron(neuron)

    def compile(self):
        self.index += 1
        self.output_layer.index = self.index
        self.layers.append(self.output_layer)

        for l in self.layers:
            l.compile()

    def activate(self,x):

        output = jnp.zeros(self.size)
        output = output.at[:len(x)].set(x)

        for l in self.layers:
            if l.layer_index > 0:
                output = output.at[l.outputs].set(l.forward(output))

        return output[self.layers[-1].outputs]

    def print(self):

        for l in self.layers:
            print("====================================")
            print(f"weigths: {l.weigths}")
            print(f"inputs:  {l.inputs}")
            print(f"outputs: {l.outputs}")


class NEAT:

    def __init__(self,inputs=2, outputs=1, population_size = 10 ):
        self.innov = 0
        self.index = 1
        self.inputs = inputs
        self.outputs = outputs
        self.population_size = population_size
        self.population = []
        self.__population_maker()

    def __population_maker(self):
        for n in range(self.population_size):
            genome = Genome()
            self.population.append(genome)
            
            index =  0
            for i in range(self.inputs):
                self.population[n].add_node(index,NodeTypes.INPUT,Rnd.uniform(min=-1.0,max=1.0),Rnd.randint(max=1))
                index += 1
            
            output_offset = index
            for o in range(self.outputs):
                self.population[n].add_node(index,NodeTypes.OUTPUT,Rnd.uniform(min=-1.0,max=1.0),Rnd.randint(max=1))
                index += 1

            creation_innov = 0  
            for i in range(self.inputs):
                for o in range(self.outputs):
                    creation_innov += 1
                    self.innov = self.population[n].add_connection(creation_innov,i,output_offset+o,Rnd.uniform(min=-1.0,max=1.0))

    def evolve(self):
        self.population,self.innov = evolve(self.population,self.innov)

    def evaluate(self):
        ''' function for evaluating genomes into ff networks '''
        networks = []
        for genome in self.population:
            networks.append(compiler(genome.node_gen,genome.con_gen))
        return networks

    def update(self,fitness):
        ''' Function for updating fitness '''
        for n,_ in enumerate(self.population):
            self.population[n].fitness = fitness[n]


def run():

    # env = gym.make("Acrobot-v1", render_mode="human")
    # env = gym.make("Acrobot-v1")
    my_neat = NEAT(6,3, 20)


    epochs = 50
    prev_action = 0.0
    experiment_length = 100
    models_path = "models"
    game = "Acrobot-v1"
    for e in range(epochs):
        print(f"================ EPOCH: {e} ================")
        env = gym.make(game)
        observation, info = env.reset(seed=42)
        all_rewards = []
        my_neat.evolve()

        networks = my_neat.evaluate()
        for n,network in enumerate(networks):
            observation, info = env.reset()
            total_reward = 0

            for _ in range(experiment_length):
                actions = network.activate(jnp.array(observation))
                action = actions.argmax()
                #promote mobility
                if prev_action != action:
                    total_reward += abs(observation[4])/10000 + abs(observation[5])/10000
                    prev_action = action

                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            all_rewards.append(total_reward)
            print(f"net: {n}, fitness: {total_reward}")

        index = all_rewards.index(max(all_rewards))
        #display the best:
        network = networks[index]
        env = gym.make(game, render_mode="human")
        
        print(f"Displaying the best: {index}, max reward: {max(all_rewards)}")
        observation, info = env.reset()
        total_reward = 0

        for _ in range(experiment_length):
            actions = network.activate(jnp.array(observation))
            action = actions.argmax()
            #promote mobility
            if prev_action != action:
                total_reward += abs(observation[4])/10000 + abs(observation[5])/10000
                prev_action = action

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        pickle.dump(network,open(f"{models_path}/{game}_e{e}.neatpy","wb"))
        my_neat.update(all_rewards)

    env.close()

if __name__=="__main__":
    run()

    # population = [superior, inferior]
    # print(speciate(population))

    # print(inferior.con_gen)
    # print(inferior.node_gen)
    # inferior = mate(superior,inferior)
    # print(inferior.con_gen)
    # print(inferior.node_gen)