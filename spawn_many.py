from mpi4py import MPI
import subprocess
import random 
import time

class MultiEnv:

    def __init__(self, processes = 10):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.processes = 10

    def spawn(self):
        # Spawn additional processes
        processes = self.comm.Spawn(
            command='python3',  # Python interpreter
            args=['spawn_many.py'],  # Script to run in spawned processes
            maxprocs=4)  # Number of processes to spawn
        return processes

    def run(self):
            
        if self.rank == 0:

            processes = self.spawn()
            # Send data to spawned processes
            for i, process in enumerate(processes):
                process.send(i, dest=i, tag=0)  # Send message to each process with their rank

            # Receive results from spawned processes
            results = [processes[i].recv(source=i, tag=0) for i in range(1, 5)]

            # Do something with the results
            print(results)
        
        else:
            data = self.comm.recv(source=0, tag=0)  # Receive data from main process
            # Process data and send results back to main process
            print("data: ", data)
            self.comm.send(data, dest=0, tag=0)

def main():
    env = MultiEnv(10)
    env.run()

if __name__ == "__main__":
    main()
