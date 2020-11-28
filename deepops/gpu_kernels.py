"""Cuda Kernels """
from pycuda.compiler import SourceModule

__all__ = ["add", "transpose", "mul"]


class Kernel:
    def addition_kernel(self):
        add = SourceModule(
            """
          __global__ void device_vec_add(float * __restrict__ d_c, const float * __restrict__ d_a, const float * __restrict__ d_b, const int N)
          {
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= N) return;
            d_c[tid] = d_a[tid] + d_b[tid];
          }
          """
        )
        return add

    def arithmetic_kernel(self, operator):
        operation = SourceModule(
            """
          __global__ void device_arithmetic(float * __restrict__ d_c, const float * __restrict__ d_a, const float * __restrict__ d_b, const int N)
          {
            const int tid = threadIdx.x + blockIdx.x * blockDim.x;
            if (tid >= N) return;
            d_c[tid] = d_a[tid] %s d_b[tid];
          }
          """
            % operator
        )
        return operation

    def multiply_kernel(self):
        mul = SourceModule(
            """
        __global__ void multiply_them(float *dest, float *a, float *b)
        {
          const int i = threadIdx.x;
          dest[i] = a[i] * b[i];
        }
        """
        )
        return mul

    def transpose_kernel(self):
        transpose = SourceModule(
            """
        #define BLOCK_SIZE %(block_size)d
        #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
        #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

        __global__ void transpose(float *A_t, float *A, int a_width, int a_height)
        {
            // Base indices in A and A_t
            int base_idx_a   = blockIdx.x * BLOCK_SIZE +
        blockIdx.y * A_BLOCK_STRIDE;
            int base_idx_a_t = blockIdx.y * BLOCK_SIZE +
        blockIdx.x * A_T_BLOCK_STRIDE;

            // Global indices in A and A_t
            int glob_idx_a   = base_idx_a + threadIdx.x + a_width * threadIdx.y;
            int glob_idx_a_t = base_idx_a_t + threadIdx.x + a_height * threadIdx.y;

            __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE+1];

            // Store transposed submatrix to shared memory
            A_shared[threadIdx.y][threadIdx.x] = A[glob_idx_a];

            __syncthreads();

            // Write transposed submatrix to global memory
            A_t[glob_idx_a_t] = A_shared[threadIdx.x][threadIdx.y];
        }
        """
        )
        return transpose

    def matrix_mul_kernel(self):
        matmul = """
        __global__ void MatrixMulKernel(float *a, float *b, float *c)
        {
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            float Pvalue = 0;

            for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
                float Aelement = a[ty * %(MATRIX_SIZE)s + k];
                float Belement = b[k * %(MATRIX_SIZE)s + tx];
                Pvalue += Aelement * Belement;
            }

            c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
        }
        """
        return matmul


kernel = Kernel()
add = kernel.addition_kernel()
mul = kernel.multiply_kernel()
transpose = None  # kernel.transpose_kernel()
arithmetic = kernel.arithmetic_kernel
