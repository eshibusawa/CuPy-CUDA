extern "C" __global__ void scale(
  unsigned char *output,
  cudaTextureObject_t tex,
  int width,
  int height)
{
  const int indexX = blockIdx.x * blockDim.x + threadIdx.x;
  const int indexY = blockIdx.y * blockDim.y + threadIdx.y;

  if ((indexX >= width) || (indexY >= height))
  {
    return;
  }
  const int index = indexX + indexY * width;
  float x = (indexX + 0.5f)/width;
  float y = (indexY + 0.5f)/height;
  // [0, 1) -> [-1.5, 1.5)
  x = 3 * (x - .5f);
  y = 3 * (y - .5f);
  output[index] =  tex2D<unsigned char>(tex, x, y);
}
