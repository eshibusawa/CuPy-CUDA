#pragma pack(push, 4)
struct multAddCoefficient
{
  float a, b;
};
#pragma pack(pop)

__constant__ multAddCoefficient g_maCoef = {}; // constant memory

extern "C" __global__ void getMACoefSize(int *sz)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0)
  {
    return;
  }
  *sz = sizeof(multAddCoefficient);
}

extern "C" __global__ void multAddConstantMemory(float *x)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= BASIC_ARRAY_OPTRATIONS_4_LENGTH)
  {
    return;
  }
  x[index] = g_maCoef.a * x[index] + g_maCoef.b;
  // use FMA for faster operation
  // x[index] = fmaf(g_maCoef.a, x[index], g_maCoef.b);
}
