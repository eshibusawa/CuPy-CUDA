#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void movingAverage2CTAWarp(float *y,
  const float *__restrict__ x)
{
	auto cta = cg::this_thread_block();
	const int ctaIndex = cta.thread_rank();
  if (ctaIndex >= STENCIL_PATTERNS_2_LENGTH)
  {
    return;
  }
  auto tile = cg::tiled_partition<STENCIL_PATTERNS_2_WARP_SIZE>(cta);
  float val = x[ctaIndex];
  float average = (tile.shfl_down(val, 1) + val) / 2;
  tile.sync();
  if (tile.thread_rank() == STENCIL_PATTERNS_2_WARP_SIZE - 1)
  {
    average = (val + x[(ctaIndex + 1) % STENCIL_PATTERNS_2_LENGTH])/2;
  }
  cta.sync();
  y[ctaIndex] = average;
}
