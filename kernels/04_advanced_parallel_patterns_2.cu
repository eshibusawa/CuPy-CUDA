#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

extern "C" __global__ void parallelPrefixSum(int* output,
        const int* __restrict__ input)
{
  using BlockLoad = cub::BlockLoad<int, ADVANCED_PARALLEL_PATTERNS_2_BLOCK_SIZE, ADVANCED_PARALLEL_PATTERNS_2_ITEMS_PER_THREAD, cub::BLOCK_LOAD_DIRECT>;
  using BlockScan = cub::BlockScan<int, ADVANCED_PARALLEL_PATTERNS_2_BLOCK_SIZE>;
  using BlockStore = cub::BlockStore<int, ADVANCED_PARALLEL_PATTERNS_2_BLOCK_SIZE, ADVANCED_PARALLEL_PATTERNS_2_ITEMS_PER_THREAD, cub::BLOCK_STORE_DIRECT>;

  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockScan::TempStorage scan;
    typename BlockStore::TempStorage store;
  } tempStorage;

  int threadData[ADVANCED_PARALLEL_PATTERNS_2_ITEMS_PER_THREAD];
  BlockLoad(tempStorage.load).Load(input, threadData);
  __syncthreads();

  BlockScan(tempStorage.scan).InclusiveSum(threadData, threadData);
  __syncthreads();

  BlockStore(tempStorage.store).Store(output, threadData);
}
