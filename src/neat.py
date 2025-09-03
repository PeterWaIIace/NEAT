"""Fast neat"""

import copy
import time
import random
import networkx as nx
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from tqdm import tqdm
from gymnasium.spaces import Discrete, Box
from utils.arrayPainter import display_array
from utils.random import StatefulRandomGenerator
from src.activations import activation_func, act2name, NUMBER_OF_ACTIVATION_FUNCTIONS
from src.genome import Genome, NodeTypes
from src.misc import δ, sh, speciate, mate, cross_over
from src.nn import Node, Layer, compiler
from src.graphs import graph, graphProcessor
# First networkx library is imported
# along with matplotlib

Rnd = StatefulRandomGenerator()

class NEAT:

    def __init__(
        self,
        inputs=2,
        outputs=1,
        population_size=10,
        keep_top=2,
        nmc=0.7,
        cmc=0.7,
        wmc=0.7,
        bmc=0.7,
        amc=0.7,
        C1=1.0,
        C2=1.0,
        C3=1.0,
        N=1.0,
        δ_th=5.0,
    ):
        """
        nmc - node mutation chance (0.0 - 1.0 higher number means lower chance)
        cmc - connection mutation chance (0.0 - 1.0 higher number means lower chance)
        wmc - weigth mutation chance (0.0 - 1.0 higher number means lower chance)
        bmc - bias mutation chance (0.0 - 1.0 higher number means lower chance)
        amc - activation mutation chance (0.0 - 1.0 higher number means lower chance)

        Speciation parameters:
        C1 - disjointed genese signifance modfier (default 1.0)
        C2 - extended genese signifance modfier (default 1.0)
        C3 - averge weigths difference modifier (defautl 1.0)
        N - population size signficance modifier (default 1.0)
        δ_th - threshold for making new species
        """
        self.innov = 0
        self.index = 1
        self.inputs = inputs
        self.outputs = outputs
        self.population_size = population_size
        self.population = []
        self.species = []

        self.keep_top = keep_top
        self.nmc = nmc  # node mutation chance
        self.cmc = cmc  # connection mutation chance
        self.wmc = wmc  # weight mutation chance
        self.bmc = bmc  # bias mutation chance
        self.amc = amc  # activation mutation chance

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.N = N
        self.δ_th = δ_th

        self.__population_maker()

    def __population_maker(self):
        print("[INFO] Preparing population\n")
        for n in tqdm(range(self.population_size)):
            genome = Genome()
            self.population.append(genome)

            index = 0
            for i in range(self.inputs):
                self.population[n].add_node(NodeTypes.INPUT.value, 0.0, Rnd.randint(max=NUMBER_OF_ACTIVATION_FUNCTIONS))
                index += 1

            output_offset = index
            for o in range(self.outputs):
                self.population[n].add_node(
                    NodeTypes.OUTPUT.value, 0.0, Rnd.randint(max=NUMBER_OF_ACTIVATION_FUNCTIONS)
                )
                index += 1

            creation_innov = 0
            for i in range(self.inputs):
                for o in range(self.outputs):
                    creation_innov = self.population[n].add_r_connection(creation_innov)
                    self.innov = creation_innov
            self.species.append(0)
            genome.specie = 0

    def mutate_activation(self, amc=0.7):
        for genome in self.population[self.keep_top :]:
            length = len(genome.nodes[:, genome.index])
            genome.change_activation(
                Rnd.randint(NUMBER_OF_ACTIVATION_FUNCTIONS, 0) * Rnd.binary(p=amc, shape=(length,))
            )

    def mutate_weight(self, epsylon=0.1, wmc=0.7):
        for genome in self.population[self.keep_top :]:
            length = len(genome.connections[:, genome.C_INNOV])
            genome.change_weigth(
                Rnd.uniform(max=epsylon, min=-epsylon, shape=(length,)) * Rnd.binary(p=wmc, shape=(length,))
            )

    def mutate_bias(self, epsylon=0.1, bmc=0.7):
        for genome in self.population[self.keep_top :]:
            length = len(genome.nodes[:, genome.index])
            genome.change_bias(Rnd.uniform(epsylon, -epsylon, shape=(length,)) * Rnd.binary(p=bmc, shape=(length,)))

    def mutate_nodes(self, nmc=0.7):
        for genome in self.population[self.keep_top :]:
            mutation_chance = random.randint(0, 10)
            if mutation_chance > nmc * 10:
                self.innov = genome.add_r_node(self.innov)

    def mutate_connections(self, cmc=0.7):
        for genome in self.population[self.keep_top :]:
            mutation_chance = random.randint(0, 10)
            if mutation_chance > cmc * 10:
                self.innov = genome.add_r_connection(self.innov)

    def cross_over(self, δ_th=5, c1=1.0, c2=1.0, c3=1.0, N=1.0):
        # print(f"[CROSS_OVER]{len(self.population)}")
        self.population, self.species = cross_over(
            self.population,
            keep_top=self.keep_top,
            population_size=self.population_size,
            δ_th=δ_th,
            c1=c1,
            c2=c2,
            c3=c3,
            N=N,
        )

        for n, _ in enumerate(self.population):
            self.population[n].specie = self.species[n]

    def prune(self, threshold):
        print(f"[PRUNE]")
        self.population = [genome for genome in self.population if threshold < genome.fitness]

    def evaluate(self):
        """function for evaluating genomes into ff networks"""
        networks = []
        plt.figure(figsize=(20, 20), dpi=150)
        plt.clf()
        for n, genome in enumerate(self.population):
            nodes = graphProcessor(genome)
            graph(nodes, n, int(self.population_size / 5), 5)
            networks.append(compiler(nodes, self.inputs, genome))
        plt.tight_layout()
        plt.pause(0.01)
        return networks

    def update(self, fitness):
        """Function for updating fitness"""
        for n, _ in enumerate(self.population):
            self.population[n].fitness = fitness[n]

    def get_params(self):
        params = []
        for n, genome in enumerate(self.population):
            params.append(
                {
                    "net": n,
                    "specienumber": genome.specie,
                    "fitness": genome.fitness,
                    "connections": len(genome.connections[genome.connections[:, genome.enabled] != 0]),
                    "nodes": len(genome.nodes[genome.nodes[:, genome.index] != 0]),
                    "nmc": self.nmc,
                    "cmc": self.cmc,
                    "wmc": self.wmc,
                    "bmc": self.bmc,
                    "amc": self.amc,
                    "C1": self.C1,
                    "C2": self.C2,
                    "C3": self.C3,
                    "N": self.N,
                }
            )
        return params


class PNEAT:

    def __init__(self):
        self.rewards = []
        self.env = None

        self.N = 20
        self.MATCHES = 3
        self.GENERATIONS = 100
        self.POPULATION_SIZE = 20
        self.NMC = 0.9
        self.CMC = 0.9
        self.WMC = 0.9
        self.BMC = 0.9
        self.AMC = 0.9
        self.δ_th = 5
        self.MUTATE_RATE = 1
        self.RENDER_HUMAN = True
        self.epsylon = 0.2
        self.INPUT_SIZE = 4
        self.OUTPUT_SIZE = 2

        self.neat = None

        self.num_timesteps = 0
        self.network = None

        # for stats only
        self.epochs = 0
        self.all_eps_rewards = []
        self.all_eps_times = []

    def __create(self):

        if not self.env:
            return

        # if len(self.env.observation_space.shape) == 1:
        if isinstance(self.env.action_space, Discrete):
            self.OUTPUT_SIZE = self.env.action_space.n
        else:
            print(self.env.action_space, type(self.env.action_space), isinstance(self.env.action_space, Discrete))
            self.OUTPUT_SIZE = self.env.action_space.shape[1]

        self.INPUT_SIZE = self.env.observation_space.shape[-1]
        print(f"self.INPUT_SIZE: {self.INPUT_SIZE}")
        self.neat = NEAT(
            self.INPUT_SIZE,
            self.OUTPUT_SIZE,
            self.POPULATION_SIZE,
            keep_top=4,
            nmc=0.5,
            cmc=0.5,
            wmc=0.5,
            bmc=0.5,
            amc=0.5,
            N=self.N,
            δ_th=self.δ_th,
        )

    def __mutate(self):
        # EVOLVE EVERYTHING
        self.neat.cross_over(δ_th=self.δ_th, N=self.N)
        self.neat.mutate_weight(epsylon=self.epsylon, wmc=self.WMC)
        self.neat.mutate_bias(epsylon=self.epsylon, bmc=self.BMC)
        for _ in range(self.MUTATE_RATE):
            self.neat.mutate_activation(amc=self.AMC)
            self.neat.mutate_nodes(nmc=self.NMC)
            self.neat.mutate_connections(cmc=self.CMC)
        return self.neat

    def set_env(self, env):
        self.env = env

    def print_stats(self):
        print(
            f"-----------------------\n"
            f"Rollout:\n"
            f"  steps: {self.num_timesteps}\n"
            f"  epochs: {self.epochs}\n"
            f"  mean_len_eps: {np.sum(self.all_eps_times)/len(self.all_eps_times)}\n"
            f"  mean_rew_eps: {np.sum(self.all_eps_rewards)/len(self.all_eps_rewards)}\n"
            f"NEAT additional stats:\n"
            f"  Population: {len(self.neat.population)}\n"
            f"  Species: {np.max(self.neat.species) + 1}\n"
            f"-----------------------\n"
        )

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        ready_to_print = False

        print("CHECKING ENV")
        if not self.env:
            return

        print("CHECKING NEAT")
        if not self.neat:
            self.__create()

        print("CHECKING TIMESTEPS")
        if reset_num_timesteps:
            self.num_timesteps = 0

        print("HERE LEARNING")
        while self.num_timesteps < total_timesteps:
            self.rewards = []

            self.neat = self.__mutate()
            networks = self.neat.evaluate()

            print("STARTING TQDM")
            for n, network in tqdm(enumerate(networks)):
                start_time = time.time()

                total_reward = 0
                observation, _ = self.env.reset()
                done = False
                while not done:

                    actions, _ = self.predict(observation, network = network)
                    # take biggest value index and make it action performed
                    observation, reward, trunacted, terminated, info = self.env.step(np.argmax(actions))
                    self.num_timesteps += 1

                    if 0 == (self.num_timesteps % log_interval):
                        ready_to_print = True

                    total_reward += reward
                    done = trunacted or terminated

                    # TODO: add callback exectution

                self.rewards.append(total_reward)

                self.all_eps_rewards.append(total_reward)
                self.all_eps_times.append(time.time() - start_time)

            self.neat.update(self.rewards)

            if ready_to_print:
                self.print_stats()

                # reset stats trackers
                self.all_eps_rewards = []
                self.all_eps_times = []
                ready_to_print = False

            self.epochs += 1

        self.network = self.neat.evaluate()
        self.env.close()

    def predict(self, observation, state=None, mask=None, deterministic=False , network = None):

        if network is None:
            network = self.network

        if not self.env:
            return

        if not network:
            print("[WARNING] NO NETWORK FOUND, PRODUCING NULL ACTIONS")
            return np.zeros(self.OUTPUT_SIZE), None

        n_agents = 1
        if len(observation.shape) > 1:
            n_agents = observation.shape[0]

        if n_agents == 1:
            total_actions = np.array(network.activate(observation))
        else:
            total_actions = np.zeros((n_agents, self.OUTPUT_SIZE))
            for n in range(n_agents):
                actions = np.array(network.activate(observation[n]))
                total_actions[n, :] = actions
        return total_actions, None
