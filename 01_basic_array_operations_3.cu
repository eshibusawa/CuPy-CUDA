__constant__ float g_a; // constant memory
extern "C" __global__ void multConstantMemory(float *x)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= BASIC_ARRAY_OPTRATIONS_3_LENGTH)
  {
    return;
  }
  x[index] *= g_a;
}
