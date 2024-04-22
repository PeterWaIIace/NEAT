from mpi4py import MPI
import subprocess

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Main process
    print(rank)
    if rank == 0:
        # Spawn additional processes
        processes = comm.Spawn(
            command='python3',  # Python interpreter
            args=['text.jax.py'],  # Script to run in spawned processes
            maxprocs=4)  # Number of processes to spawn

        # Send data to spawned processes
        for i, process in enumerate(processes):
            print(f"send: {i}")
            process.send(i, dest=i, tag=0)  # Send message to each process with their rank

        # Receive results from spawned processes
        results = [process.recv(source=i, tag=0) for i in range(1, 5)]

        print(results)
        # Do something with the results

        # Terminate spawned processes
        for process in processes:
            process.send(None, dest=0, tag=1)  # Send termination signal
            process.Disconnect()

    # Spawned processes
    else:
        while True:
            data = comm.recv(source=0, tag=0)  # Receive data from main process
            if data is None:
                break  # Terminate the loop if termination signal received
            # Process data and send results back to main process
            result = process_data(data)
            comm.send(result, dest=0, tag=0)

def process_data(data):
    # Placeholder function to process data
    return data * 2

if __name__ == "__main__":
    main()
