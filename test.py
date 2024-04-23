from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    print(f"sent: {data}")
    for i in range(1,comm.Get_size()):
        comm.send(data, dest=i, tag=11)
else:
    data = comm.recv(source=0, tag=11)
    print(f"received: {data}")
