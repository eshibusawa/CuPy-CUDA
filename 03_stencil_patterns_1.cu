extern "C" __global__ void movingAverage2(float *y,
  const float *__restrict__ x)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= STENCIL_PATTERNS_1_LENGTH)
  {
    return;
  }

  __shared__ float xShared[STENCIL_PATTERNS_1_BLOCK];
  xShared[threadIdx.x] = x[index];
  __syncthreads();

  if (threadIdx.x == (STENCIL_PATTERNS_1_BLOCK - 1))
  {
    y[index] = (xShared[threadIdx.x] + x[(index + 1) % STENCIL_PATTERNS_1_LENGTH])/2;
  }
  else
  {
    y[index] = (xShared[threadIdx.x] + xShared[threadIdx.x + 1])/2;
  }
}
