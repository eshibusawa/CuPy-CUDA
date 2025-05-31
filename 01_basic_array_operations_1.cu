extern "C" __global__ void mult(float *x, float a, int length)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= length)
  {
    return;
  }
  x[index] *= a;
}
