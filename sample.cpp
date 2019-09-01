#include <cassert>
#include <iostream>
#include <cstdint>

#include <nccl.h>
#include <mpi.h>
#include <cuda.h>

constexpr const int BcastRoot = 0;
MPI_Comm mpi_comm;
int mpi_rank;
int mpi_size;
int mpi_local_rank;
ncclUniqueId nccl_id;
ncclComm_t nccl_comm;

void init_nccl() {
  if (mpi_rank == BcastRoot) {
    ncclGetUniqueId(&nccl_id);
  }

  MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, BcastRoot, MPI_COMM_WORLD);

  ncclCommInitRank(&nccl_comm, mpi_size, nccl_id, mpi_rank);
}


void finalize_nccl() {
  ncclCommDestroy(nccl_comm);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  mpi_comm = MPI_COMM_WORLD;

  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  assert(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  mpi_local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  assert(mpi_local_rank < 8);

  if (mpi_rank == 0) {
    DEBUG(std::cout) << "KF: MPI size = " << mpi_size << std::endl;
  }

  cudaSetDevice(mpi_local_rank);

  init_nccl();

  size_t buf_size = 1024 * 1024; // 1MiB
  if (argc > 1) {
    buf_size = atol(argv[1]);
    assert(buf_size > 0);
  }

  DEBUG(std::cout) << "buffer size = " << buf_size << std::endl;

  int *dmem = nullptr;
  cudaMalloc(&dmem, buf_size);

  auto ret = ncclAllReduce(dmem, dmem, buf_size / sizeof(int), ncclInt32, ncclSum, nccl_comm, 0);
  assert(ret == ncclSuccess);

  cudaFree(dmem);
  finalize_nccl();
  MPI_Finalize();
}
