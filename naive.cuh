
__global__ void
matrixMul_naive( float* C, float* A, float* B, int wA, int wB)
{

  int bx = blockIdx.x;
  int by = blockIdx.y;


  int tx = threadIdx.x;
  int ty = threadIdx.y;


  int i = by * blockDim.y + ty;
  int j = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int k=0; k<wA; k++){
    accu = accu + A[ i * wA + k ] * B[ k * wB + j ];
  }


  C[ i * wB + j ] = accu;

}
