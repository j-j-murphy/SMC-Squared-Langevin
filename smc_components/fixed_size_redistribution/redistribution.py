import numpy as np
from mpi4py import MPI
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from smc_components.fixed_size_redistribution.rotational_nearly_sort import rot_nearly_sort
from smc_components.fixed_size_redistribution.rotational_split import rot_split


def sequential_redistribution(x, ncopies):
    return np.repeat(x, ncopies, axis=0)


def redistribute(x, ncopies):

    if MPI.COMM_WORLD.Get_size() > 1:
        x, ncopies = rot_nearly_sort(x, ncopies)
        x, ncopies = rot_split(x, ncopies)

    x = sequential_redistribution(x, ncopies)

    return x


def centralised_redistribution(x, ncopies):
    comm = MPI.COMM_WORLD
    N = len(ncopies) * comm.Get_size()
    rank = comm.Get_rank()

    all_ncopies = np.zeros(N, dtype=ncopies.dtype)
    all_x = np.zeros([N, x.shape[1]], dtype=x.dtype)
    ncopies_MPI_dtype = MPI._typedict[ncopies.dtype.char]
    x_MPI_dtype = MPI._typedict[x.dtype.char]

    comm.Gather(sendbuf=[ncopies, ncopies_MPI_dtype], recvbuf=[all_ncopies, ncopies_MPI_dtype], root=0)
    comm.Gather(sendbuf=[x, x_MPI_dtype], recvbuf=[all_x, x_MPI_dtype], root=0)

    if rank == 0:
        all_x = sequential_redistribution(all_x, all_ncopies)

    comm.Scatter(sendbuf=[all_x, x_MPI_dtype], recvbuf=[x, x_MPI_dtype], root=0)

    return x