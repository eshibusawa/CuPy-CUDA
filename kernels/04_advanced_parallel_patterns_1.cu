#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_shuffle.cuh>
#include <cub/block/block_store.cuh>

extern "C" __global__ void vectorNormalization(float* output,
        const float* __restrict__ input)
{
  using BlockLoad = cub::BlockLoad<float, ADVANCED_PARALLEL_PATTERNS_1_BLOCK_SIZE, ADVANCED_PARALLEL_PATTERNS_1_ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>;
  using BlockReduce = cub::BlockReduce<float, ADVANCED_PARALLEL_PATTERNS_1_BLOCK_SIZE>;
  using BlockStore = cub::BlockStore<float, ADVANCED_PARALLEL_PATTERNS_1_BLOCK_SIZE, ADVANCED_PARALLEL_PATTERNS_1_ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>;
  using BlockShuffle = cub::BlockShuffle<float, ADVANCED_PARALLEL_PATTERNS_1_BLOCK_SIZE>;
  __shared__ union
  {
    typename BlockLoad::TempStorage load;
    typename BlockReduce::TempStorage reduce;
    typename BlockShuffle::TempStorage shuffle;
    typename BlockStore::TempStorage store;
  } tempStorage;

  float threadData[ADVANCED_PARALLEL_PATTERNS_1_ITEMS_PER_THREAD];
  BlockLoad(tempStorage.load).Load(input, threadData);

  float threadDataSqr[ADVANCED_PARALLEL_PATTERNS_1_ITEMS_PER_THREAD];
  #pragma unroll
  for (int k = 0; k < ADVANCED_PARALLEL_PATTERNS_1_ITEMS_PER_THREAD; k++)
  {
    threadDataSqr[k] = (threadData[k]) * (threadData[k]);
  }
  __syncthreads();

  float sqrSum = BlockReduce(tempStorage.reduce).Sum(threadDataSqr);
  float norm, normDummy[1];
  if (threadIdx.x == 0)
  {
    norm = normDummy[0] = sqrtf(sqrSum);
  }
  else
  {
    norm = normDummy[0] = 1.f;
  }
  __syncthreads();
  BlockShuffle(tempStorage.shuffle).Down(normDummy, normDummy, norm);

  #pragma unroll
  for (int k = 0; k < ADVANCED_PARALLEL_PATTERNS_1_ITEMS_PER_THREAD; k++)
  {
    threadData[k] = (threadData[k]) / norm;
  }
  __syncthreads();

  BlockStore(tempStorage.store).Store(output, threadData);
}
