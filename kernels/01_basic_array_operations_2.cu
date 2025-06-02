const float g_a(BASIC_ARRAY_OPTRATIONS_2_A);
extern "C" __global__ void multConstant(float *x)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= BASIC_ARRAY_OPTRATIONS_2_LENGTH)
  {
    return;
  }
  x[index] *= g_a;
}
