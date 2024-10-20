#include <stdio.h>
#include <cuda_fp16.h>
#include <sys/time.h>
#include "csr2block.h"
#include "blockspmv_cpu.h"
#include "utils.h"
#include "./biio2.0/src/biio.h"
#include "common.h"

#define NUM_THREADS 128
#define NUM_BLOCKS 16

#define THREAD_ID threadIdx.x + blockIdx.x *blockDim.x
#define THREAD_COUNT gridDim.x *blockDim.x
#define delta_x 1e-15
#define epsilon 1e-6

#define IMAX 1000
__global__ void device_convert(double *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = x[tid];
    }
}
__global__ void device_convert_half(double *x, half *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = __double2half(x[tid]);
    }
}
__global__ void device_convert_int8(double *x, int8_t *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = (int8_t)(x[tid]);
    }
}
__global__ void add_mix(double *y, float *y_float, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] += (double)(y_float[tid]);
    }
}
__global__ void add_mix_half(double *y, half *y_half, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] += (double)(y_half[tid]);
    }
}

__global__ void veczero(int n, double *vec)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        vec[i] = 0;
}

__global__ void scalardiv(double *num, double *den, double *result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *result = (*num) / (*den);
    }
}

__global__ void axpy(int n, double *a, double *x, double *y, double *r)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        r[i] = y[i] + (*a) * x[i];
}

// Computes y= y-a*x for n-length vectors x and y, and scalar a.
__global__ void ymax(int n, double *a, double *x, double *y)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        y[i] = y[i] - (*a) * x[i];
}

// Sets dest=src for scalars on the GPU.
void scalarassign(double *dest, double *src)
{
    cudaMemcpy(dest, src, sizeof(double), cudaMemcpyDeviceToDevice);
}

// Sets dest=src for n-length vectors on the GPU.
void vecassign(double *dest, double *src, int n)
{
    cudaMemcpy(dest, src, sizeof(double) * n, cudaMemcpyDeviceToDevice);
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_mixed_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                                                                     int *d_tile_ptr,
                                                                                                     int *d_tile_columnidx,
                                                                                                     unsigned char *d_csr_compressedIdx,
                                                                                                     double *d_Blockcsr_Val_d,
                                                                                                     unsigned char *d_Blockcsr_Ptr,
                                                                                                     int *d_ptroffset1,
                                                                                                     int *d_ptroffset2,
                                                                                                     int rowblkblock,
                                                                                                     unsigned int *d_blkcoostylerowidx,
                                                                                                     int *d_blkcoostylerowidx_colstart,
                                                                                                     int *d_blkcoostylerowidx_colstop,
                                                                                                     double *d_x_d,
                                                                                                     double *d_y_d,
                                                                                                     unsigned char *d_blockrowid_new,
                                                                                                     unsigned char *d_blockcsr_ptr_new,
                                                                                                     int *d_nonzero_row_new,
                                                                                                     unsigned char *d_Tile_csr_Col,
                                                                                                     int *d_block_signal,
                                                                                                     int *signal_dot,
                                                                                                     int *signal_final,
                                                                                                     int *signal_final1,
                                                                                                     int *d_ori_block_signal,
                                                                                                     double *k_alpha,
                                                                                                     double *k_snew,
                                                                                                     double *k_x,
                                                                                                     double *k_r,
                                                                                                     double *k_sold,
                                                                                                     double *k_beta,
                                                                                                     double *k_threshold,
                                                                                                     int *d_balance_tile_ptr,
                                                                                                     int *d_row_each_block,
                                                                                                     int *d_index_each_block,
                                                                                                     int balance_row,
                                                                                                     int *d_non_each_block_offset,
                                                                                                     int *d_vis,
                                                                                                     float *d_Blockcsr_Val_f,
                                                                                                     half *d_Blockcsr_Val_h,
                                                                                                     int8_t *d_Blockcsr_Val_8,
                                                                                                     float *d_x_f,
                                                                                                     half *d_x_h,
                                                                                                     int8_t *d_x_8,
                                                                                                     float *d_y_f)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int vis_spmv[WARP_PER_BLOCK];
    int *vis_spmv_val = &vis_spmv[local_warp_id];

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
        int blkj_blc;
        int blkj;
        int blki;
        int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        int index_dot;
        int offset = blki_blc * BLOCK_SIZE;

        // for(int iter=1;(iter<=100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < BLOCK_SIZE)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            if (lane_id == 0)
            {
                vis_spmv_val[lane_id] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id];
            }
            __threadfence();
            if (global_id == 0)
            {
                signal_dot[0] = tilem;
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                int colid = d_tile_columnidx[blkj];
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        if (d_vis[colid] == 0) // FP64
                        {
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                        else if (d_vis[colid] == 4) // FP32
                        {
                            sum_f = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                int csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                        else if (d_vis[colid] == 3) // FP16
                        {
                            sum_h = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                        // if (d_vis[colid] == 4) // FP8
                        else
                        {
                            sum_8 = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[x_offset + csrcol] * d_Blockcsr_Val_8[csroffset + rj];
                            }
                            // packed one thread 2 int_8
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_8 += d_x_8[csrcol + x_offset] * d_Blockcsr_Val_8[csroffset + rj];
                            //     // sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
                            //         // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
                            //     }
                            // }
                            // packed one thread 4 int_8
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                    }
                }

                if (lane_id == 0)
                {
                    atomicSub(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != 0);
                index_dot = offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
                }
                __syncthreads();
                int i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot1[threadIdx.x] += s_dot1[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    d_vis[blki_blc] = 0;
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    double d_x_last = d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_d[index_dot]);
                    double val = fabs(d_x_d[index_dot] - d_x_last);
                    if (val <= 1e-7 && val > 1e-8) // fp8
                    {
                        vis_spmv_val[0] = 2;
                    }
                    else if (val <= 1e-6 && val > 1e-7) // fp16
                    {
                        vis_spmv_val[0] = 3;
                    }
                    else if (val <= 1e-5 && val > 1e-6) // fp32
                    {
                        vis_spmv_val[0] = 4;
                    }
                    else if (val > 1e-5)
                    {
                        vis_spmv_val[0] = 1; // fp64
                    }
                }
                __syncthreads();
                if (lane_id == 0)
                {
                    if (vis_spmv_val[0] == 0) // bypass
                    {
                        d_vis[blki_blc] = 1;
                    }
                    if (vis_spmv_val[0] == 2) // fp8
                    {
                        d_vis[blki_blc] = 2;
                    }
                    if (vis_spmv_val[0] == 3) // fp16
                    {
                        d_vis[blki_blc] = 3;
                    }
                    if (vis_spmv_val[0] == 4) // fp32
                    {
                        d_vis[blki_blc] = 4;
                    }
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_shared_queue_mixed_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                                                                                  int *d_tile_ptr,
                                                                                                                  int *d_tile_columnidx,
                                                                                                                  unsigned char *d_csr_compressedIdx,
                                                                                                                  double *d_Blockcsr_Val_d,
                                                                                                                  unsigned char *d_Blockcsr_Ptr,
                                                                                                                  int *d_ptroffset1,
                                                                                                                  int *d_ptroffset2,
                                                                                                                  int rowblkblock,
                                                                                                                  unsigned int *d_blkcoostylerowidx,
                                                                                                                  int *d_blkcoostylerowidx_colstart,
                                                                                                                  int *d_blkcoostylerowidx_colstop,
                                                                                                                  double *d_x_d,
                                                                                                                  double *d_y_d,
                                                                                                                  unsigned char *d_blockrowid_new,
                                                                                                                  unsigned char *d_blockcsr_ptr_new,
                                                                                                                  int *d_nonzero_row_new,
                                                                                                                  unsigned char *d_Tile_csr_Col,
                                                                                                                  int *d_block_signal,
                                                                                                                  int *signal_dot,
                                                                                                                  int *signal_final,
                                                                                                                  int *signal_final1,
                                                                                                                  int *d_ori_block_signal,
                                                                                                                  double *k_alpha,
                                                                                                                  double *k_snew,
                                                                                                                  double *k_x,
                                                                                                                  double *k_r,
                                                                                                                  double *k_sold,
                                                                                                                  double *k_beta,
                                                                                                                  double *k_threshold,
                                                                                                                  int *d_balance_tile_ptr,
                                                                                                                  int *d_row_each_block,
                                                                                                                  int *d_index_each_block,
                                                                                                                  int balance_row,
                                                                                                                  int *d_non_each_block_offset,
                                                                                                                  int *d_balance_tile_ptr_shared_end,
                                                                                                                  int *d_vis,
                                                                                                                  float *d_Blockcsr_Val_f,
                                                                                                                  half *d_Blockcsr_Val_h,
                                                                                                                  int8_t *d_Blockcsr_Val_8,
                                                                                                                  float *d_x_f,
                                                                                                                  half *d_x_h,
                                                                                                                  int8_t *d_x_8,
                                                                                                                  float *d_y_f,
                                                                                                                  int shared_num)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int vis_spmv[WARP_PER_BLOCK];
    int *vis_spmv_val = &vis_spmv[local_warp_id];

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjshared_end = d_balance_tile_ptr_shared_end[blki_blc + 1];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
        int blkj_blc;
        int blkj;
        int blki;
        int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        int index_dot;
        int offset = blki_blc * BLOCK_SIZE;

        const int nnz_per_warp = 256;
        __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
        double *s_data_val = &s_data[local_warp_id * nnz_per_warp];
        __shared__ float s_data_float[nnz_per_warp * WARP_PER_BLOCK];
        float *s_data_float_val = &s_data_float[local_warp_id * nnz_per_warp];
        __shared__ half s_data_half[nnz_per_warp * WARP_PER_BLOCK];
        half *s_data_half_val = &s_data_half[local_warp_id * nnz_per_warp];
        __shared__ int8_t s_data_int8[nnz_per_warp * WARP_PER_BLOCK];
        int8_t *s_data_int8_val = &s_data_int8[local_warp_id * nnz_per_warp];
        for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
        {
            blkj = d_index_each_block[blkj_blc];
            shared_offset = d_non_each_block_offset[blkj_blc];
            csroffset = d_ptroffset1[blkj];
            s1 = d_nonzero_row_new[blkj];
            s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    index_s = rj + shared_offset;
                    s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
                    s_data_float_val[index_s] = (float)(s_data_val[index_s]);
                    s_data_half_val[index_s] = (half)(s_data_val[index_s]);
                    s_data_int8_val[index_s] = (int8_t)(s_data_val[index_s]);
                }
            }
        }

        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < BLOCK_SIZE)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            if (lane_id == 0)
            {
                vis_spmv_val[lane_id] = 0;
            }
            __syncthreads();
            __threadfence();

            if (global_id == 0)
            {
                signal_dot[0] = tilem;
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        shared_offset = d_non_each_block_offset[blkj_blc];
                        if (d_vis[colid] == 0) // FP64
                        {
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * s_data_val[index_s]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                        else if (d_vis[colid] == 4) // FP32
                        {
                            sum_f = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_f += d_x_f[x_offset + csrcol] * s_data_float_val[index_s];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                        else if (d_vis[colid] == 3) // FP16
                        {
                            sum_h = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], s_data_half_val[index_s], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                                // sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            }
                            // packed one thread 2 half
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     index_s = rj + shared_offset;
                            //     sum_h = __hfma(d_x_h[x_offset + csrcol], s_data_half_val[index_s], sum_h);
                            //     // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)s_data_val[index_s], sum_h);
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], s_data_half_val[index_s + 1], sum_h);
                            //         // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)s_data_val[index_s+1], sum_h);
                            //     }
                            // }
                            // sum_d = (double)(sum_h);
                            // atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                        }
                        else
                        {
                            sum_8 = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                                sum_8 += d_x_8[x_offset + csrcol] * s_data_int8_val[index_s];
                            }
                            // packed one thread 2 int_8
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     index_s = rj + shared_offset;
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_8 += d_x_8[csrcol + x_offset] * s_data_int8_val[index_s];
                            //     // sum_8 += d_x_8[csrcol+x_offset] * (int8_t)s_data_val[index_s];
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * s_data_int8_val[index_s + 1];
                            //         // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1]+x_offset] * (int8_t)s_data_val[index_s+1];
                            //     }
                            // }
                            // packed one thread 4 int_8
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = (float)sum_8;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            // 如果能都放在shared memory中 则不需要下面的循环
            //  for (blkj_blc = rowblkjshared_end; blkj_blc < rowblkjstop; blkj_blc++)
            //  {
            //      blkj = d_index_each_block[blkj_blc];
            //      blki = d_row_each_block[blkj_blc];
            //      int colid = d_tile_columnidx[blkj];
            //      if ((d_vis[colid] != 1))
            //      {
            //          x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
            //          csroffset = d_ptroffset1[blkj];
            //          s1 = d_nonzero_row_new[blkj];
            //          s2 = d_nonzero_row_new[blkj + 1];
            //          if (ri < s2 - s1)
            //          {
            //              ro = d_blockrowid_new[s1 + ri + 1];
            //              if (d_vis[colid] == 0) // FP64
            //              {
            //                  sum_d = 0.0;
            //                  for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
            //                  {
            //                      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
            //                  }
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d); // 这个原子加占了25%的寄存器 其他的原子加占了12.5%
            //              }
            //              if (d_vis[colid] == 4) // FP32
            //              {
            //                  sum_f = 0.0;
            //                  // sum_d = 0.0;
            //                  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
            //                  {
            //                      int csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      // sum_f += d_x_f[x_offset + csrcol] * (float)(d_Blockcsr_Val_d[csroffset + rj]);
            //                      sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
            //                  }
            //                  sum_d = (double)(sum_f);
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            //                  // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
            //              }
            //              if (d_vis[colid] == 3) // FP16
            //              {
            //                  sum_h = 0.0;
            //                  // sum_f = 0.0;
            //                  int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
            //                  // unpacked
            //                  //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
            //                  //  {
            //                  //      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                  //      sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
            //                  //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
            //                  //  }
            //                  // packed one thread 2 half
            //                  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
            //                  {
            //                      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
            //                      // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
            //                      if (rj + 1 < row_end)
            //                      {
            //                          sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
            //                          // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
            //                      }
            //                  }
            //                  sum_d = (double)(sum_h);
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            //                  // sum_f = __half2float(sum_h);
            //                  // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
            //              }
            //              if (d_vis[colid] == 4) // FP8
            //              {
            //                  //sum_f = 0.0;
            //                  sum_8 = 0.0;
            //                  int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
            //                  // unpacked 结果正确
            //                  //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
            //                  //  {
            //                  //      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                  //      sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
            //                  //  }
            //                  // packed one thread 2 int_8
            //                  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
            //                  {
            //                      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      sum_8 += d_x_8[csrcol+x_offset] * d_Blockcsr_Val_8[csroffset + rj];
            //                      //sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
            //                      if (rj + 1 < row_end)
            //                      {
            //                          sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1]+x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
            //                          //sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
            //                      }
            //                  }
            //                  // packed one thread 4 int_8
            //                  //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
            //                  //  {
            //                  //      int csrcol = d_Tile_csr_Col[csroffset + rj];
            //                  //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
            //                  //      if(rj+1<row_end)
            //                  //      {
            //                  //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
            //                  //      }
            //                  //      if(rj+2<row_end)
            //                  //      {
            //                  //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
            //                  //      }
            //                  //  }
            //                  sum_d = (double)(sum_8);
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            //                  // sum_f = (float)sum_8;
            //                  // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
            //              }
            //          }
            //      }
            //      if (lane_id == 0)
            //      {
            //          atomicAdd(&d_block_signal[blki], 1);
            //      }
            //  }

            if (blki_blc < tilem)
            {
                index_dot = iter * d_ori_block_signal[blki_blc];
                do
                {
                    __threadfence_system();
                } while (d_block_signal[blki_blc] != index_dot);

                index_dot = offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
                }
                __syncthreads();
                int i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot1[threadIdx.x] += s_dot1[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    double d_x_last = d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_d[index_dot]);
                    d_y_d[index_dot] = 0.0;
                    {
                        double val = fabs(d_x_d[index_dot] - d_x_last);
                        if (val <= 1e-7 && val > 1e-8) // fp8
                        {
                            vis_spmv_val[0] = 2;
                        }
                        else if (val <= 1e-6 && val > 1e-7) // fp16
                        {
                            vis_spmv_val[0] = 3;
                        }
                        else if (val <= 1e-5 && val > 1e-6) // fp32
                        {
                            vis_spmv_val[0] = 4;
                        }
                        else if (val > 1e-5)
                        {
                            vis_spmv_val[0] = 1; // fp64
                        }
                    }
                }
                __syncthreads();
                if (lane_id == 0)
                {
                    d_vis[blki_blc] = 0;
                    if (vis_spmv_val[0] == 0) // bypass
                    {
                        d_vis[blki_blc] = 1;
                    }
                    if (vis_spmv_val[0] == 2) // fp8
                    {
                        d_vis[blki_blc] = 2;
                    }
                    if (vis_spmv_val[0] == 3) // fp16
                    {
                        d_vis[blki_blc] = 3;
                    }
                    if (vis_spmv_val[0] == 4) // fp32
                    {
                        d_vis[blki_blc] = 4;
                    }
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_mix_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                                                                                   int *d_tile_ptr,
                                                                                                                   int *d_tile_columnidx,
                                                                                                                   unsigned char *d_csr_compressedIdx,
                                                                                                                   double *d_Blockcsr_Val_d,
                                                                                                                   unsigned char *d_Blockcsr_Ptr,
                                                                                                                   int *d_ptroffset1,
                                                                                                                   int *d_ptroffset2,
                                                                                                                   int rowblkblock,
                                                                                                                   unsigned int *d_blkcoostylerowidx,
                                                                                                                   int *d_blkcoostylerowidx_colstart,
                                                                                                                   int *d_blkcoostylerowidx_colstop,
                                                                                                                   double *d_x_d,
                                                                                                                   double *d_y_d,
                                                                                                                   unsigned char *d_blockrowid_new,
                                                                                                                   unsigned char *d_blockcsr_ptr_new,
                                                                                                                   int *d_nonzero_row_new,
                                                                                                                   unsigned char *d_Tile_csr_Col,
                                                                                                                   int *d_block_signal,
                                                                                                                   int *signal_dot,
                                                                                                                   int *signal_final,
                                                                                                                   int *signal_final1,
                                                                                                                   int *d_ori_block_signal,
                                                                                                                   double *k_alpha,
                                                                                                                   double *k_snew,
                                                                                                                   double *k_x,
                                                                                                                   double *k_r,
                                                                                                                   double *k_sold,
                                                                                                                   double *k_beta,
                                                                                                                   double *k_threshold,
                                                                                                                   int *d_balance_tile_ptr,
                                                                                                                   int *d_row_each_block,
                                                                                                                   int *d_index_each_block,
                                                                                                                   int balance_row,
                                                                                                                   int *d_non_each_block_offset,
                                                                                                                   int vector_each_warp,
                                                                                                                   int vector_total,
                                                                                                                   int *d_vis,
                                                                                                                   double *d_x_d_last,
                                                                                                                   float *d_Blockcsr_Val_f,
                                                                                                                   half *d_Blockcsr_Val_h,
                                                                                                                   int8_t *d_Blockcsr_Val_8,
                                                                                                                   float *d_x_f,
                                                                                                                   half *d_x_h,
                                                                                                                   int8_t *d_x_8,
                                                                                                                   float *d_y_f)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    __shared__ int row_begin[WARP_PER_BLOCK];
    __shared__ int row_end[WARP_PER_BLOCK];
    __shared__ int vis_spmv[WARP_PER_BLOCK * 2];
    int *vis_spmv_val = &vis_spmv[local_warp_id * 2];

    if (blki_blc < balance_row)
    {
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];

        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
        int blkj_blc;
        int blkj;
        int blki;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        int index_dot;
        int offset = blki_blc * vector_each_warp;
        int iter;
        int u;
        int row_end;

        for (iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            __syncthreads();
            if (lane_id < 2)
            {
                vis_spmv_val[lane_id] = 0;
            }
            __syncthreads();
            __threadfence();
            if (global_id == 0)
            {
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        switch (d_vis[colid])
                        {
                        case 0:
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        case 4:
                            sum_f = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        case 3:
                            sum_h = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                            }
                            // packed one thread 2 half
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                            //     // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
                            //         // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                            //     }
                            // }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        case 2:
                            sum_8 = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[x_offset + csrcol] * d_Blockcsr_Val_8[csroffset + rj];
                            }
                            // packed one thread 2 int_8
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_8 += d_x_8[csrcol + x_offset] * d_Blockcsr_Val_8[csroffset + rj];
                            //     // sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
                            //     }
                            // }
                            // packed one thread 4 int_8
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += d_x_8[csrcol+x_offset]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj+1]+x_offset]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj+2]+x_offset]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        default:
                            break;
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < vector_total)
            {
                for (u = 0; u < vector_each_warp * 2; u++)
                {
                    int off = blki_blc * vector_each_warp * 2;
                    index_dot = iter * d_ori_block_signal[(off + u)];
                    do
                    {
                        __threadfence();
                    } while (d_block_signal[(off + u)] != index_dot);
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
                }
                __syncthreads();
                int i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot1[threadIdx.x] += s_dot1[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }

                if (threadIdx.x == 0)
                {
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();

                if ((lane_id == 0))
                {
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }

                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    double d_x_last = d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_y_d[index_dot] = 0.0;
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_8[index_dot]);
                    d_vis[(index_dot / 16)] = 0;
                    double val = fabs(d_x_d[index_dot] - d_x_last);
                    if (val <= 1e-7 && val > 1e-8) // fp8
                    {
                        vis_spmv_val[(lane_id / 16)] = 2;
                    }
                    else if (val <= 1e-6 && val > 1e-7) // fp16
                    {
                        vis_spmv_val[(lane_id / 16)] = 3;
                    }
                    else if (val <= 1e-5 && val > 1e-6) // fp32
                    {
                        vis_spmv_val[(lane_id / 16)] = 4;
                    }
                    else if (val > 1e-5)
                    {
                        vis_spmv_val[(lane_id / 16)] = 1; // fp64
                    }
                    __syncthreads();
                    if (lane_id % 16 == 0)
                    {
                        switch (vis_spmv_val[lane_id / 16])
                        {
                        case 0:
                            d_vis[(index_dot / 16)] = 1;
                            break;
                        case 2:
                            d_vis[(index_dot / 16)] = 2;
                            break;
                        case 3:
                            d_vis[(index_dot / 16)] = 3;
                            break;
                        case 4:
                            d_vis[(index_dot / 16)] = 4;
                            break;
                        default:
                            break;
                        }
                    }
                    __syncthreads();
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_shared_queue_mix_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                                                                                                int *d_tile_ptr,
                                                                                                                                int *d_tile_columnidx,
                                                                                                                                unsigned char *d_csr_compressedIdx,
                                                                                                                                double *d_Blockcsr_Val_d,
                                                                                                                                unsigned char *d_Blockcsr_Ptr,
                                                                                                                                int *d_ptroffset1,
                                                                                                                                int *d_ptroffset2,
                                                                                                                                int rowblkblock,
                                                                                                                                unsigned int *d_blkcoostylerowidx,
                                                                                                                                int *d_blkcoostylerowidx_colstart,
                                                                                                                                int *d_blkcoostylerowidx_colstop,
                                                                                                                                double *d_x_d,
                                                                                                                                double *d_y_d,
                                                                                                                                unsigned char *d_blockrowid_new,
                                                                                                                                unsigned char *d_blockcsr_ptr_new,
                                                                                                                                int *d_nonzero_row_new,
                                                                                                                                unsigned char *d_Tile_csr_Col,
                                                                                                                                int *d_block_signal,
                                                                                                                                int *signal_dot,
                                                                                                                                int *signal_final,
                                                                                                                                int *signal_final1,
                                                                                                                                int *d_ori_block_signal,
                                                                                                                                double *k_alpha,
                                                                                                                                double *k_snew,
                                                                                                                                double *k_x,
                                                                                                                                double *k_r,
                                                                                                                                double *k_sold,
                                                                                                                                double *k_beta,
                                                                                                                                double *k_threshold,
                                                                                                                                int *d_balance_tile_ptr,
                                                                                                                                int *d_row_each_block,
                                                                                                                                int *d_index_each_block,
                                                                                                                                int balance_row,
                                                                                                                                int *d_non_each_block_offset,
                                                                                                                                int vector_each_warp,
                                                                                                                                int vector_total,
                                                                                                                                int *d_vis,
                                                                                                                                double *d_x_d_last,
                                                                                                                                float *d_Blockcsr_Val_f,
                                                                                                                                half *d_Blockcsr_Val_h,
                                                                                                                                int8_t *d_Blockcsr_Val_8,
                                                                                                                                float *d_x_f,
                                                                                                                                half *d_x_h,
                                                                                                                                int8_t *d_x_8,
                                                                                                                                float *d_y_f,
                                                                                                                                int *d_balance_tile_ptr_shared_end,
                                                                                                                                int shared_num)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    // __shared__ int row_begin[WARP_PER_BLOCK];
    // __shared__ int row_end[WARP_PER_BLOCK];
    // int *row_begin_val = &row_begin[local_warp_id];
    // int *row_end_val = &row_end[local_warp_id];
    __shared__ int vis_spmv[WARP_PER_BLOCK * 2];
    int *vis_spmv_val = &vis_spmv[local_warp_id * 2];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjshared_end = d_balance_tile_ptr_shared_end[blki_blc + 1];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        // row_begin_val[0] = d_balance_tile_ptr[blki_blc];
        // row_end_val[0] = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
        int blkj_blc;
        int blkj;
        int blki;
        int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        int index_dot;
        int offset = blki_blc * vector_each_warp;
        int iter;
        int u;
        int row_end;

        const int nnz_per_warp = 256;
        __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
        double *s_data_val = &s_data[local_warp_id * nnz_per_warp];
        __shared__ float s_data_float[nnz_per_warp * WARP_PER_BLOCK];
        float *s_data_float_val = &s_data_float[local_warp_id * nnz_per_warp];
        __shared__ half s_data_half[nnz_per_warp * WARP_PER_BLOCK];
        half *s_data_half_val = &s_data_half[local_warp_id * nnz_per_warp];
        __shared__ int8_t s_data_int8[nnz_per_warp * WARP_PER_BLOCK];
        int8_t *s_data_int8_val = &s_data_int8[local_warp_id * nnz_per_warp];
        for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
        {
            blkj = d_index_each_block[blkj_blc];
            shared_offset = d_non_each_block_offset[blkj_blc];
            csroffset = d_ptroffset1[blkj];
            s1 = d_nonzero_row_new[blkj];
            s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    index_s = rj + shared_offset;
                    s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
                    s_data_float_val[index_s] = (float)(s_data_val[index_s]);
                    s_data_half_val[index_s] = (half)(s_data_val[index_s]);
                    s_data_int8_val[index_s] = (int8_t)(s_data_val[index_s]);
                }
            }
            if (lane_id == 0)
            {
                atomicAdd(signal_final1, 1);
            }
        }
        do
        {
            __threadfence();
        } while (signal_final1[0] != shared_num);

        // for(int iter=1;(iter<=100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            __syncthreads();
            if (lane_id < 2)
            {
                vis_spmv_val[lane_id] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0.0;
            }
            __threadfence();

            if (global_id == 0)
            {
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        shared_offset = d_non_each_block_offset[blkj_blc];
                        switch (d_vis[colid])
                        {
                        case 0:
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * s_data_val[index_s]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;

                        case 4:
                            sum_f = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_f += d_x_f[x_offset + csrcol] * s_data_float_val[index_s];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        case 3:
                            sum_h = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      index_s = rj + shared_offset;
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],s_data_half_val[index_s],sum_h);
                            //      //sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                index_s = rj + shared_offset;
                                sum_h = __hfma(d_x_h[x_offset + csrcol], s_data_half_val[index_s], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)s_data_val[index_s], sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], s_data_half_val[index_s + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)s_data_val[index_s+1], sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        case 2:
                            sum_8 = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                                sum_8 += d_x_8[x_offset + csrcol] * s_data_int8_val[index_s];
                            }
                            // packed one thread 2 int_8
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     index_s = rj + shared_offset;
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_8 += d_x_8[csrcol + x_offset] * s_data_int8_val[index_s];
                            //     // sum_8 += d_x_8[csrcol+x_offset] * (int8_t)s_data_val[index_s];
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * s_data_int8_val[index_s + 1];
                            //         // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1]+x_offset] * (int8_t)s_data_val[index_s+1];
                            //     }
                            // }
                            // packed one thread 4 int_8
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        default:
                            break;
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            // 如果能都放在shared memory中 则不需要下面的循环
            for (blkj_blc = rowblkjshared_end; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                // if ((d_vis[colid] != 1))
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        switch (d_vis[colid])
                        {
                        // if (d_vis[colid] == 0) // FP64
                        //{
                        case 0:
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        // else if (d_vis[colid] == 4) // FP32
                        //{
                        case 4:
                            sum_f = 0.0;
                            // sum_d = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                int csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_f += d_x_f[x_offset + csrcol] * (float)(d_Blockcsr_Val_d[csroffset + rj]);
                                sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // else if (d_vis[colid] == 3) // FP16
                        //{
                        case 3:
                            sum_h = 0.0;
                            // sum_f = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // if (d_vis[colid] == 4) // FP8
                        // else
                        case 2:
                            //{
                            // sum_f = 0.0;
                            sum_8 = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //  }
                            // packed one thread 2 int_8
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[csrcol + x_offset] * d_Blockcsr_Val_8[csroffset + rj];
                                // sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                                if (rj + 1 < row_end)
                                {
                                    sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
                                    // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
                                }
                            }
                            // packed one thread 4 int_8
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        default:
                            break;
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < vector_total)
            {
                for (u = 0; u < vector_each_warp * 2; u++)
                {
                    int off = blki_blc * vector_each_warp * 2;
                    index_dot = iter * d_ori_block_signal[(off + u)];
                    do
                    {
                        __threadfence();
                    } while (d_block_signal[(off + u)] != index_dot);
                }

                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
                }
                __syncthreads();
                int i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot1[threadIdx.x] += s_dot1[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }

                if (threadIdx.x == 0)
                {
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();

                if ((lane_id == 0))
                {
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }

                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    double d_x_last = d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_y_d[index_dot] = 0.0;
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_8[index_dot]);
                    d_vis[(index_dot / 16)] = 0;
                    double val = fabs(d_x_d[index_dot] - d_x_last);
                    if (val <= 1e-7 && val > 1e-8) // fp8
                    {
                        vis_spmv_val[(lane_id / 16)] = 2;
                    }
                    else if (val <= 1e-6 && val > 1e-7) // fp16
                    {
                        vis_spmv_val[(lane_id / 16)] = 3;
                    }
                    else if (val <= 1e-5 && val > 1e-6) // fp32
                    {
                        vis_spmv_val[(lane_id / 16)] = 4;
                    }
                    else if (val > 1e-5)
                    {
                        vis_spmv_val[(lane_id / 16)] = 1; // fp64
                    }
                    __syncthreads();
                    if (lane_id == 0)
                    {
                        if (vis_spmv_val[0] == 0) // bypass
                        {
                            d_vis[(index_dot / 16)] = 1;
                        }
                        if (vis_spmv_val[0] == 2) // fp8
                        {
                            d_vis[(index_dot / 16)] = 2;
                        }
                        if (vis_spmv_val[0] == 3) // fp16
                        {
                            d_vis[(index_dot / 16)] = 3;
                        }
                        if (vis_spmv_val[0] == 4) // fp32
                        {
                            d_vis[(index_dot / 16)] = 4;
                        }
                    }
                    if (lane_id == 16)
                    {
                        if (vis_spmv_val[1] == 0) // bypass
                        {
                            d_vis[(index_dot / 16)] = 1;
                        }
                        if (vis_spmv_val[1] == 2) // fp8
                        {
                            d_vis[(index_dot / 16)] = 2;
                        }
                        if (vis_spmv_val[1] == 3) // fp16
                        {
                            d_vis[(index_dot / 16)] = 3;
                        }
                        if (vis_spmv_val[1] == 4) // fp32
                        {
                            d_vis[(index_dot / 16)] = 4;
                        }
                    }
                    __syncthreads();
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
        }
    }
}

__global__ void sdot2_2(double *a, double *b, double *c, int n)
{

    // Define variables.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double temp;
    temp = 0;
    // Define shared memories.
    __shared__ double s_data[1024];
    unsigned int tid = threadIdx.x;
    // Multiplication of data in the index.
    for (int i = index; i < n; i += stride)
    {
        temp += (a[i] * b[i]);
    }
    // Assign value to shared memory.
    s_data[tid] = temp;
    __syncthreads();
    // Add up products.
    for (int s = blockDim.x / 4; s > 0; s >>= 2)
    {
        if ((tid < s))
        {
            temp = s_data[tid];
            temp += s_data[tid + s];
            temp += s_data[tid + (s << 1)];
            temp += s_data[tid + (3 * s)];
            s_data[tid] = temp;
        }
        __syncthreads();
    }
    s_data[0] += s_data[1];
    if (tid == 0)
    {
        atomicAdd(c, s_data[0]);
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr(int tilem, int tilen, int rowA, int colA, int nnzA,
                                             int *d_tile_ptr,
                                             int *d_tile_columnidx,
                                             char *d_Format,
                                             int *d_blknnz,
                                             unsigned char *d_blknnznnz,
                                             unsigned char *d_csr_compressedIdx,
                                             double *d_Blockcsr_Val_d,
                                             float *d_Blockcsr_Val_f,
                                             half *d_Blockcsr_Val_h,
                                             unsigned char *d_Blockcsr_Ptr,
                                             int *d_ptroffset1,
                                             int *d_ptroffset2,
                                             int rowblkblock,
                                             unsigned int *d_blkcoostylerowidx,
                                             int *d_blkcoostylerowidx_colstart,
                                             int *d_blkcoostylerowidx_colstop,
                                             double *d_x_d,
                                             double *d_y_d,
                                             unsigned char *d_blockrowid_new,
                                             unsigned char *d_blockcsr_ptr_new,
                                             int *d_nonzero_row_new,
                                             unsigned char *d_Tile_csr_Col)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_x_d[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_x_warp_d = &s_x_d[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    if (blki_blc < rowblkblock)
    {
        double sum_d = 0.0;
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;
        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_tile_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_tile_ptr[blki + 1];
        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
        }
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {
            int colid = s_columnid_local[blkj - rowblkjstart];
            int x_offset = colid * BLOCK_SIZE;
            int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
            int ri = lane_id >> 1;
            int virtual_lane_id = lane_id & 0x1;
            int s1 = d_nonzero_row_new[blkj];
            int s2 = d_nonzero_row_new[blkj + 1];
            sum_d = 0.0;
            if (lane_id < BLOCK_SIZE)
            {
                s_x_warp_d[lane_id] = d_x_d[x_offset + lane_id];
            }
            if (ri < s2 - s1)
            {
                int ro = d_blockrowid_new[s1 + ri + 1];
                for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    int csrcol = d_Tile_csr_Col[csroffset + rj];
                    sum_d += s_x_warp_d[csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                }
                atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            }
        }
    }
}
__global__ void te1_four_precision(double *p, double *last_val, int *vis_new, unsigned int *vis_mix_64, unsigned int *vis_mix_32, unsigned int *vis_mix_16, unsigned int *vis_mix_8)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    double val = fabs(p[global_id] - last_val[global_id]);
    if (val >= delta_x)
    {
        vis_new[blockIdx.x] = 1;
    }
    if (val > delta_x * 1e3)
    {
        atomicOr(&(vis_mix_64[blockIdx.x / 32]), (1 << (blockIdx.x % 32)));
    }
    if (val <= delta_x * 1e3 && val > delta_x * 1e2)
    {
        atomicOr(&(vis_mix_32[blockIdx.x / 32]), (1 << (blockIdx.x % 32)));
    }
    if (val <= delta_x * 1e2 && val > delta_x * 1e1)
    {
        atomicOr(&(vis_mix_16[blockIdx.x / 32]), (1 << (blockIdx.x % 32)));
    }
    if (val <= delta_x * 1e1 && val > delta_x)
    {
        atomicOr(&(vis_mix_8[blockIdx.x / 32]), (1 << (blockIdx.x % 32)));
    }
}
__global__ void stir_spmv_cuda_kernel_newcsr_balance_inc_balance_mix(
    MAT_PTR_TYPE *d_tile_ptr,
    int *d_tile_columnidx,
    double *d_Blockcsr_Val_d,
    float *d_Blockcsr_Val_f,
    half *d_Blockcsr_Val_h,
    int8_t *d_Blockcsr_Val_8,
    int *d_ptroffset1,
    int rowblkblock,
    double *d_x_d,
    float *d_x_f,
    half *d_x_h,
    int8_t *d_x_8,
    double *d_y_d,
    float *d_y_f,
    unsigned char *d_blockrowid_new,
    unsigned char *d_blockcsr_ptr_new,
    int *d_nonzero_row_new,
    unsigned char *d_Tile_csr_Col,
    int *d_vis,
    int *d_b_start,
    int *d_tile_rowidx,
    int *d_b_map,
    unsigned int *d_vis_mix_32,
    unsigned int *d_vis_mix_16,
    unsigned int *d_vis_mix_8,
    unsigned int *d_vis_mix_64)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    if (blki_blc < rowblkblock)
    {
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
        int rowblkjstart = d_b_start[blki_blc];
        int rowblkjstop = d_b_start[blki_blc + 1];
        for (int blkj_n = rowblkjstart; blkj_n < rowblkjstop; blkj_n++)
        {
            int blkj = d_b_map[blkj_n];
            int colid = d_tile_columnidx[blkj];
            int x_offset = colid * BLOCK_SIZE;
            int csroffset = d_ptroffset1[blkj];
            int ri = lane_id >> 1;
            int virtual_lane_id = lane_id & 0x1;
            int s1 = d_nonzero_row_new[blkj];
            int s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                int blki = d_tile_rowidx[blkj];
                int ro = d_blockrowid_new[s1 + ri + 1];
                if (((d_vis_mix_64[colid / 32] >> (colid % 32)) & 1))
                {
                    sum_d = 0;
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                else
                {
                    if (((d_vis_mix_32[colid / 32] >> (colid % 32)) & 1))
                    {
                        sum_f = 0.0;
                        for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                        {
                            int csrcol = d_Tile_csr_Col[csroffset + rj];
                            sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                        }
                        atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                    }
                    else if (((d_vis_mix_8[colid / 32] >> (colid % 32)) & 1))
                    {
                        sum_f = 0.0;
                        sum_8 = 0.0;
                        if (ri < s2 - s1)
                        {
                            int ro = d_blockrowid_new[s1 + ri + 1];
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                int csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[x_offset + csrcol] * d_Blockcsr_Val_8[csroffset + rj];
                            }
                            sum_f = (float)sum_8;
                            atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                    }
                    else
                    {
                        sum_h = 0.0;
                        sum_f = 0.0;
                        if (ri < s2 - s1)
                        {
                            int ro = d_blockrowid_new[s1 + ri + 1];
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(s_x_warp_h[csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                int csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                                }
                            }
                            sum_f = __half2float(sum_h);
                            atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                    }
                }
            }
        }
    }
}

extern "C" void cg_solve_inc(int *RowPtr, int *ColIdx, MAT_VAL_TYPE *Val, MAT_VAL_LOW_TYPE *Val_Low, double *x, double *b, int n, int *iter, int maxiter, double threshold, char *filename, int nnzR, int ori)
{
    struct timeval t1, t2, t3, t4, t5, t6, t7, t8;
    int rowA = n;
    int colA = ori;
    rowA = (rowA / BLOCK_SIZE) * BLOCK_SIZE;
    Tile_matrix *matrix = (Tile_matrix *)malloc(sizeof(Tile_matrix));
    Tile_create(matrix,
                rowA, colA, nnzR,
                RowPtr,
                ColIdx,
                Val,
                Val_Low);
    int num_seg = ceil((double)rowA / BLOCK_SIZE);
    int tilenum = matrix->tilenum;
    int *ptroffset1 = (int *)malloc(sizeof(int) * tilenum);
    int *ptroffset2 = (int *)malloc(sizeof(int) * tilenum);
    memset(ptroffset1, 0, sizeof(int) * tilenum);
    memset(ptroffset2, 0, sizeof(int) * tilenum);
    MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * n);
    memset(x, 0, sizeof(double) * n);
    memset(y, 0, sizeof(MAT_VAL_TYPE) * n);
    int rowblkblock = 0;
    unsigned int *blkcoostylerowidx;
    int *blkcoostylerowidx_colstart;
    int *blkcoostylerowidx_colstop;
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    blockspmv_cpu(matrix,
                  ptroffset1,
                  ptroffset2,
                  &rowblkblock,
                  &blkcoostylerowidx,
                  &blkcoostylerowidx_colstart,
                  &blkcoostylerowidx_colstop,
                  rowA, colA, nnzR,
                  RowPtr,
                  ColIdx,
                  Val,
                  x,
                  y,
                  y_golden);
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    MAT_VAL_LOW_TYPE *Blockcsr_Val_Low = matrix->Blockcsr_Val_Low;
    unsigned char *Tile_csr_Col = matrix->Tile_csr_Col;
    unsigned char *csr_compressedIdx = matrix->csr_compressedIdx;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    int csrsize = matrix->csrsize;
    int csrptrlen = matrix->csrptrlen;
    int csr_csize = csrsize % 2 == 0 ? csrsize / 2 : csrsize / 2 + 1;

    MAT_PTR_TYPE *d_tile_ptr;
    int *d_tile_columnidx;
    int *tile_rowidx = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_rowidx, 0, sizeof(int) * tilenum);
    int *d_tile_rowidx;
    cudaMalloc((void **)&d_tile_rowidx, tilenum * sizeof(int));
    cudaMalloc((void **)&d_tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_tile_columnidx, tilenum * sizeof(int));

    cudaMemcpy(d_tile_ptr, tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_columnidx, tile_columnidx, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    int *tile_columnidx_new = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_columnidx_new, 0, sizeof(int) * tilenum);
    // CSR
    unsigned char *d_csr_compressedIdx = (unsigned char *)malloc((csr_csize) * sizeof(unsigned char));
    MAT_VAL_TYPE *d_Blockcsr_Val;
    unsigned char *d_Blockcsr_Ptr;

    cudaMalloc((void **)&d_csr_compressedIdx, (csr_csize) * sizeof(unsigned char));
    cudaMalloc((void **)&d_Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char));

    cudaMemcpy(d_csr_compressedIdx, csr_compressedIdx, (csr_csize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Val, Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Ptr, Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    float *d_Blockcsr_Val_float;
    half *d_Blockcsr_Val_half;
    half *Blockcsr_Val_half = (half *)malloc(sizeof(half) * csrsize);
    int8_t *Blockcsr_Val_int8 = (int8_t *)malloc(sizeof(int8_t) * csrsize);
    int8_t *d_Blockcsr_Val_int8;
    cudaMalloc((void **)&d_Blockcsr_Val_float, (csrsize) * sizeof(float));
    cudaMalloc((void **)&d_Blockcsr_Val_half, (csrsize) * sizeof(half));
    cudaMalloc((void **)&d_Blockcsr_Val_int8, (csrsize) * sizeof(int8_t));
    cudaMemcpy(d_Blockcsr_Val_float, Blockcsr_Val_Low, (csrsize) * sizeof(float), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < csrsize; i++)
    {
        Blockcsr_Val_half[i] = (half)(Blockcsr_Val_Low[i]);
        Blockcsr_Val_int8[i] = (int8_t)Blockcsr_Val[i];
    }
    cudaMemcpy(d_Blockcsr_Val_half, Blockcsr_Val_half, (csrsize) * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Val_int8, Blockcsr_Val_int8, (csrsize) * sizeof(int8_t), cudaMemcpyHostToDevice);
    free(Blockcsr_Val_half);
    free(Blockcsr_Val_int8);

    unsigned int *d_blkcoostylerowidx;
    int *d_blkcoostylerowidx_colstart;
    int *d_blkcoostylerowidx_colstop;

    cudaMalloc((void **)&d_blkcoostylerowidx, rowblkblock * sizeof(unsigned int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstart, rowblkblock * sizeof(int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstop, rowblkblock * sizeof(int));

    cudaMemcpy(d_blkcoostylerowidx, blkcoostylerowidx, rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstart, blkcoostylerowidx_colstart, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstop, blkcoostylerowidx_colstop, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);

    int *d_ptroffset1;
    int *d_ptroffset2;

    cudaMalloc((void **)&d_ptroffset1, tilenum * sizeof(int));
    cudaMalloc((void **)&d_ptroffset2, tilenum * sizeof(int));
    cudaMemcpy(d_ptroffset1, ptroffset1, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptroffset2, ptroffset2, tilenum * sizeof(int), cudaMemcpyHostToDevice);

    // x and y
    MAT_VAL_TYPE *d_x;
    MAT_VAL_TYPE *d_y;

    cudaMalloc((void **)&d_x, rowA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_y, rowA * sizeof(MAT_VAL_TYPE));
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
    int num_blocks_new = ceil((double)(tilem) / (double)(num_threads / WARP_SIZE));

    float *k_d_float, *k_q_float;
    half *k_d_half;
    int8_t *k_d_int8;
    cudaMalloc((void **)&k_d_float, sizeof(float) * (n));
    cudaMalloc((void **)&k_d_half, sizeof(half) * (n));
    cudaMalloc((void **)&k_d_int8, sizeof(int8_t) * (n));
    cudaMalloc((void **)&k_q_float, sizeof(float) * (n));
    cudaMemset(k_q_float, 0, n * sizeof(float));

    double *k_b, *k_x, *k_r, *k_d, *k_q, *k_s;
    double *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;
    double t, s0, snew;
    double alpha;
    double *k_val;
    int iterations = 0;

    cudaMalloc((void **)&k_b, sizeof(double) * (n));
    cudaMemcpy(k_b, b, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_val, sizeof(double) * (nnzR));
    cudaMemcpy(k_val, Val, sizeof(double) * (nnzR), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&k_x, sizeof(double) * (n));
    cudaMalloc((void **)&k_r, sizeof(double) * (n + 1));
    cudaMalloc((void **)&k_d, sizeof(double) * (n + 1));
    double *d_last = (double *)malloc(sizeof(double) * (n + 1));
    memset(d_last, 0, sizeof(double) * (n + 1));
    double *d = (double *)malloc(sizeof(double) * (n + 1));
    memset(d, 0, sizeof(double) * (n + 1));
    int *vis_new = (int *)malloc(sizeof(int) * num_seg);
    memset(vis_new, 0, sizeof(int) * num_seg);
    cudaMalloc((void **)&k_q, sizeof(double) * (n));
    cudaMalloc((void **)&k_s, sizeof(double) * (n));
    cudaMalloc((void **)&k_alpha, sizeof(double));
    cudaMalloc((void **)&k_snew, sizeof(double));
    cudaMalloc((void **)&k_sold, sizeof(double));
    cudaMalloc((void **)&k_beta, sizeof(double));
    cudaMalloc((void **)&k_s0, sizeof(double));
    double *r = (double *)malloc(sizeof(double) * (n + 1));
    memset(r, 0, sizeof(double) * (n + 1));

    dim3 BlockDim(128);
    dim3 GridDim((n / 128 + 1));

    veczero<<<1, BlockDim>>>(n, k_x);
    // r=b-Ax (r=b since x=0), and d=M^(-1)r
    cudaMemcpy(k_r, k_b, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
    cudaMemset(k_s0, 0, sizeof(double));

    sdot2_2<<<GridDim, BlockDim>>>(k_r, k_r, k_s0, n);
    cudaMemcpy(k_d, k_r, sizeof(double) * (n + 1), cudaMemcpyDeviceToDevice);
    device_convert<<<num_seg, BLOCK_SIZE>>>(k_d, k_d_float, n);
    device_convert_half<<<num_seg, BLOCK_SIZE>>>(k_d, k_d_half, n);
    device_convert_int8<<<num_seg, BLOCK_SIZE>>>(k_d, k_d_int8, n);
    // r[n] = 1.2;
    //  snew = s0
    scalarassign(k_snew, k_s0);
    // Copy snew and s0 back to host so that host can evaluate stopping condition
    cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s0, k_s0, sizeof(double), cudaMemcpyDeviceToHost);
    double time_spmv = 0;

    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
    gettimeofday(&t5, NULL);
#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            csrcount = ptroffset2[blkj];
            tile_rowidx[blkj] = blki;
            for (int ri = 0; ri < rowlength; ri++)
            {
                int stop = ri == rowlength - 1 ? (matrix->blknnz[blkj + 1] - matrix->blknnz[blkj]) : matrix->Blockcsr_Ptr[ri + 1 + csrcount];
                if (stop != matrix->Blockcsr_Ptr[csrcount + ri])
                {
                    nonzero_row_new[blkj] += 1;
                }
            }
            nonzero_row_new[blkj] += 1;
        }
    }
    exclusive_scan(nonzero_row_new, tilenum + 1);
    int cnt_non_new = nonzero_row_new[tilenum];
    unsigned char *blockrowid_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockrowid_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    unsigned char *blockcsr_ptr_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockcsr_ptr_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    int csrcount_new1 = 0;
    int *block_signal = (int *)malloc(sizeof(int) * (tilem + 4));
    memset(block_signal, 0, sizeof(int) * (tilem + 4)); // 记录块数
    // #pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        block_signal[blki] = matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki];
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            csrcount = ptroffset2[blkj];
            csrcount_new1 = nonzero_row_new[blkj];
            int fl = 0;
            for (int ri = 0; ri < rowlength; ri++)
            {
                int stop = ri == rowlength - 1 ? (matrix->blknnz[blkj + 1] - matrix->blknnz[blkj]) : matrix->Blockcsr_Ptr[ri + 1 + csrcount];
                if (ri == 0)
                {
                    blockrowid_new[csrcount_new1 + fl] = ri;
                    blockcsr_ptr_new[csrcount_new1 + fl] = 0;
                    fl++;
                }
                if (stop != matrix->Blockcsr_Ptr[csrcount + ri])
                {
                    blockrowid_new[csrcount_new1 + fl] = ri;
                    blockcsr_ptr_new[csrcount_new1 + fl] = stop;
                    fl++;
                }
            }
        }
    }

    // 负载均衡部分 重新分配每个warp需要计算的非零元
    int *non_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        // 记录每个块的非零元个数
    int *non_each_block_offset = (int *)malloc(sizeof(int) * (tilenum + 1)); // 用于shared memory的索引
    int *row_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        // 记录每个块的行号
    int *index_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));      // 排序前每个块的索引
    memset(non_each_block, 0, sizeof(int) * (tilenum + 1));
    memset(non_each_block_offset, 0, sizeof(int) * (tilenum + 1));
    memset(row_each_block, 0, sizeof(int) * (tilenum + 1));
    memset(index_each_block, 0, sizeof(int) * (tilenum + 1));
    int nnz_total = 0;
    for (int blki = 0; blki < tilem; blki++)
    {
        for (int blkj = tile_ptr[blki]; blkj < tile_ptr[blki + 1]; blkj++)
        {
            non_each_block[blkj] = matrix->blknnz[blkj + 1] - matrix->blknnz[blkj];
            nnz_total += non_each_block[blkj];
            row_each_block[blkj] = blki;
            index_each_block[blkj] = blkj;
        }
    }

    int *row_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));   // 记录每个块的行号
    int *index_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1)); // 排序前每个块的索引
    int *non_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(row_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(index_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(non_each_block_new, 0, sizeof(int) * (tilenum + 1));
    int each_block_nnz = 16;
    int cnt = 0;
    int balance_row = 0;
    int index = 1;
    int i = 0;
    int j = tilenum - 1;
    cnt = 0;
    index = 1;
    int step = 0;
    int block_per_warp = 180;
    int cnt_block1 = 0;
    int nnz_list[12] = {16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, nnzR / 6912};
    while (1)
    {
        for (int k = 0; k < 12; k++)
        {
            each_block_nnz = nnz_list[k];
            i = 0;
            j = tilenum - 1;
            cnt = 0;
            index = 1;
            step = 0;
            cnt_block1 = 0;
            while (i < j)
            {
                if (((non_each_block[i] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
                {
                    cnt += non_each_block[i];
                    i++;
                    cnt_block1++;
                }
                else if (((non_each_block[i] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
                {
                    i++;
                    index++;
                    cnt = 0;
                    cnt_block1 = 0;
                }
                if (((non_each_block[j] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
                {
                    cnt += non_each_block[j];
                    j--;
                    cnt_block1++;
                }
                else if (((non_each_block[j] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
                {
                    j--;
                    index++;
                    cnt = 0;
                    cnt_block1 = 0;
                }
            }
            if (index < 6912)
                break;
        }
        if (index < 6912)
            break;
        block_per_warp = block_per_warp * 2;
    }
    int vector_each_warp_16;
    int vector_total_16;
    int vector_each_warp_32;
    int vector_total_32;
    if (index < tilem)
    {
        vector_each_warp_16 = ceil((double)(tilem) / (double)(index));
        vector_total_16 = tilem / vector_each_warp_16;
        int tilem_32 = ceil((double)tilem / 2);
        vector_each_warp_32 = vector_each_warp_16 * 2;
        vector_total_32 = tilem_32 / vector_each_warp_32;
        vector_total_32 = (vector_total_32 / WARP_PER_BLOCK + 1) * WARP_PER_BLOCK;
    }
    if (index > 6912 || index == 0 || tilem == 0)
        return;
    int *balance_tile_ptr_new = (int *)malloc(sizeof(int) * (index + 1));
    memset(balance_tile_ptr_new, 0, sizeof(int) * (index + 1));
    int *balance_tile_ptr_shared_end = (int *)malloc(sizeof(int) * (index + 1));
    memset(balance_tile_ptr_shared_end, 0, sizeof(int) * (index + 1));
    i = 0;
    j = tilenum - 1;
    cnt = 0;
    index = 1;
    step = 0;
    cnt_block1 = 0;
    while (i < j)
    {
        if (((non_each_block[i] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
        {
            cnt += non_each_block[i];
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[i] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
        {
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1 = 0;
        }
        if (((non_each_block[j] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
        {
            cnt += non_each_block[j];
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[j] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1 = 0;
        }
        if (i == j)
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            step++;
            balance_tile_ptr_new[index] = step;
        }
        if (i > j)
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            balance_tile_ptr_new[index] = step;
        }
    }

    int *d_balance_tile_ptr_new;
    cudaMalloc((void **)&d_balance_tile_ptr_new, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_new, balance_tile_ptr_new, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_row_each_block;
    int *d_index_each_block;
    cudaMalloc((void **)&d_row_each_block, sizeof(int) * (tilenum + 1));
    cudaMalloc((void **)&d_index_each_block, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_row_each_block, row_each_block_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_each_block, index_each_block_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    int cnt_block = 0;
    int cnt_nnz = 0;
    for (int i = 0; i <= index; i++)
    {
        balance_tile_ptr_shared_end[i] = balance_tile_ptr_new[i];
    }
    int shared_nnz_each_block = 256;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            int blkj = index_each_block_new[j];
            if (j == balance_tile_ptr_new[i])
                non_each_block_offset[j] = 0;
            cnt_nnz += non_each_block_new[j];
            cnt_block++;
            if (j != balance_tile_ptr_new[i] && cnt_nnz <= shared_nnz_each_block)
            {
                non_each_block_offset[j] = non_each_block_new[j - 1];
                non_each_block_offset[j] += non_each_block_offset[j - 1];
            }
            if (cnt_nnz > shared_nnz_each_block)
            {
                balance_tile_ptr_shared_end[i + 1] = j;
                break;
            }
        }
    }

    int *d_non_each_block_offset;
    cudaMalloc((void **)&d_non_each_block_offset, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_non_each_block_offset, non_each_block_offset, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);

    int *d_balance_tile_ptr_shared_end;
    cudaMalloc((void **)&d_balance_tile_ptr_shared_end, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_shared_end, balance_tile_ptr_shared_end, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_block_signal;
    cudaMalloc((void **)&d_block_signal, sizeof(int) * (tilem + 4));
    int *signal_dot;
    cudaMalloc((void **)&signal_dot, sizeof(int));
    int *signal_final;
    cudaMalloc((void **)&signal_final, sizeof(int));
    int *signal_final1;
    cudaMalloc((void **)&signal_final1, sizeof(int));
    cudaMemset(signal_final1, 0, sizeof(int));
    double *k_threshold;
    cudaMalloc((void **)&k_threshold, sizeof(double));
    int *d_ori_block_signal;
    cudaMalloc((void **)&d_ori_block_signal, sizeof(int) * (tilem + 4));

    cudaMemcpy(d_block_signal, block_signal, sizeof(int) * (tilem + 4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ori_block_signal, block_signal, sizeof(int) * (tilem + 4), cudaMemcpyHostToDevice);
    gettimeofday(&t6, NULL);
    double time_format = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
    double pro_cnt = 0.0;
    unsigned char *d_blockrowid_new;
    unsigned char *d_blockcsr_ptr_new;
    int *d_nonzero_row_new;
    unsigned char *d_Tile_csr_Col;
    cudaMalloc((void **)&d_blockrowid_new, sizeof(unsigned char) * (cnt_non_new + 1));
    cudaMalloc((void **)&d_blockcsr_ptr_new, sizeof(unsigned char) * (cnt_non_new + 1));
    cudaMalloc((void **)&d_nonzero_row_new, sizeof(int) * (tilenum + 1));
    cudaMalloc((void **)&d_Tile_csr_Col, sizeof(unsigned char) * (matrix->csrsize));
    cudaMemcpy(d_blockrowid_new, blockrowid_new, sizeof(unsigned char) * (cnt_non_new + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockcsr_ptr_new, blockcsr_ptr_new, sizeof(unsigned char) * (cnt_non_new + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonzero_row_new, nonzero_row_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tile_csr_Col, Tile_csr_Col, sizeof(unsigned char) * (matrix->csrsize), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_rowidx, tile_rowidx, sizeof(int) * (tilenum), cudaMemcpyHostToDevice);
    threshold = epsilon * epsilon * s0;
    double *k_d_last;
    cudaMalloc((void **)&k_d_last, sizeof(double) * (n + 1));
    cudaMemset(k_d_last, 0, sizeof(double) * (n + 1));
    double *k_x_new;
    cudaMemcpy(k_threshold, &threshold, sizeof(double), cudaMemcpyHostToDevice);
    gettimeofday(&t1, NULL);
    {
        if (index < tilem)
        {
            int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
            cudaMemset(d_block_signal, 0, sizeof(int) * (tilem + 1));
            if (vector_each_warp_32 * vector_total_32 * 32 > rowA)
            {
                rowA = vector_each_warp_32 * vector_total_32 * 32;
            }
            int tilem_new = rowA / BLOCK_SIZE;
            int *d_block_signal_new;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            double *k_q_new;
            cudaMalloc((void **)&k_q_new, sizeof(double) * (rowA + 1));
            double *k_d_new;
            cudaMalloc((void **)&k_d_new, sizeof(double) * (rowA + 1));
            cudaMemset(k_d_new, 0, (rowA + 1) * sizeof(double));
            cudaMemcpy(k_d_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_r_new;
            cudaMalloc((void **)&k_r_new, sizeof(double) * (rowA + 1));
            cudaMemset(k_r_new, 0, (rowA + 1) * sizeof(double));
            cudaMemcpy(k_r_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_x_new, sizeof(double) * (rowA + 1));
            cudaMemset(k_x_new, 0, (rowA + 1) * sizeof(double));
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            int *d_vis;
            cudaMalloc((void **)&d_vis, (rowA + 1) * sizeof(int));
            cudaMemset(d_vis, 0, (rowA + 1) * sizeof(int));
            float *k_d_float_new, *k_q_float_new;
            half *k_d_half_new;
            int8_t *k_d_int8_new;
            cudaMalloc((void **)&k_d_float_new, sizeof(float) * (rowA + 1));
            cudaMemcpy(k_d_float_new, k_d_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_half_new, sizeof(half) * (rowA + 1));
            cudaMemcpy(k_d_half_new, k_d_half, sizeof(half) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_int8_new, sizeof(int8_t) * (rowA + 1));
            cudaMemcpy(k_d_int8_new, &k_d_int8, sizeof(int8_t) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_q_float_new, sizeof(float) * (rowA + 1));
            cudaMemcpy(k_q_float_new, &k_q_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_mix_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                                                                        d_tile_ptr, d_tile_columnidx,
                                                                                                                                        d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                                                                        d_ptroffset1, d_ptroffset2,
                                                                                                                                        rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                                                                        k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                                                                        signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                                                                        k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
                                                                                                                                        d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,
                                                                                                                                        vector_each_warp_32, vector_total_32, d_vis, k_d_last, d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
                                                                                                                                        k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new);

            // stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_shared_queue_mix_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
            //                                                                                                                                          d_tile_ptr, d_tile_columnidx,
            //                                                                                                                                          d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                                                                          d_ptroffset1, d_ptroffset2,
            //                                                                                                                                          rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                                                                          k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
            //                                                                                                                                          signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
            //                                                                                                                                          k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
            //                                                                                                                                          d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,
            //                                                                                                                                          vector_each_warp_32, vector_total_32, d_vis, k_d_last, d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
            //                                                                                                                                          k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new, d_balance_tile_ptr_shared_end, shared_num);

            cudaDeviceSynchronize();
        }
        else
        {
            if (index == tilem)
                index = tilem + 1;
            cudaMemset(d_block_signal, 0, sizeof(int) * (tilem + 1));
            int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
            // 扩大容量
            int tilem_new = (tilem / WARP_PER_BLOCK + 2) * WARP_PER_BLOCK;
            int re_size = (tilem_new)*BLOCK_SIZE;
            int *d_block_signal_new;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            double *k_q_new;
            cudaMalloc((void **)&k_q_new, sizeof(double) * re_size);
            double *k_d_new;
            cudaMalloc((void **)&k_d_new, sizeof(double) * re_size);
            cudaMemset(k_d_new, 0, re_size * sizeof(double));
            cudaMemcpy(k_d_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_r_new;
            cudaMalloc((void **)&k_r_new, sizeof(double) * re_size);
            cudaMemset(k_r_new, 0, re_size * sizeof(double));
            cudaMemcpy(k_r_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            // double *k_x_new;
            cudaMalloc((void **)&k_x_new, sizeof(double) * re_size);
            cudaMemset(k_x_new, 0, re_size * sizeof(double));
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            int *d_vis;
            cudaMalloc((void **)&d_vis, tilem_new * sizeof(int));
            cudaMemset(d_vis, 0, tilem_new * sizeof(int));
            float *k_d_float_new, *k_q_float_new;
            half *k_d_half_new;
            int8_t *k_d_int8_new;
            cudaMalloc((void **)&k_d_float_new, sizeof(float) * (re_size));
            cudaMemcpy(k_d_float_new, k_d_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_half_new, sizeof(half) * (re_size));
            cudaMemcpy(k_d_half_new, k_d_half, sizeof(half) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_int8_new, sizeof(int8_t) * (re_size));
            cudaMemcpy(k_d_int8_new, &k_d_int8, sizeof(int8_t) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_q_float_new, sizeof(float) * (re_size));
            cudaMemcpy(k_q_float_new, &k_q_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            // 都放在global memory上
            stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_mixed_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
                                                                                                                          d_tile_ptr, d_tile_columnidx,
                                                                                                                          d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                                                          d_ptroffset1, d_ptroffset2,
                                                                                                                          rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                                                          k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                                                          signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                                                          k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
                                                                                                                          d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset, d_vis,
                                                                                                                          d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
                                                                                                                          k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new);
            // 放在shared memory上
            //  stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_shared_queue_mixed_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
            //                                                                                    d_tile_ptr, d_tile_columnidx,
            //                                                                                    d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                    d_ptroffset1, d_ptroffset2,
            //                                                                                    rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                    k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
            //                                                                                    signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
            //                                                                                    k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
            //                                                                                    d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,d_balance_tile_ptr_shared_end,d_vis,
            //                                                                                    d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
            //                                                                                    k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new,shared_num);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t4, NULL);
        time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    cudaMemcpy(x, k_x_new, sizeof(double) * (n), cudaMemcpyDeviceToHost);
    double time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    double *b_new = (double *)malloc(sizeof(double) * n);
    memset(b_new, 0, sizeof(double) * n);
    for (int blki = 0; blki < tilem; blki++)
    {
        for (int ri = 0; ri < BLOCK_SIZE; ri++)
        {
            b_new[blki * BLOCK_SIZE + ri] = 0;
        }
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            int csrcolidx = tile_columnidx[blkj];
            int x_offset = csrcolidx * BLOCK_SIZE;
            csroffset = matrix->csr_offset[blkj];
            int cnt = 0;
            for (int ri = nonzero_row_new[blkj]; ri < nonzero_row_new[blkj + 1]; ri++)
            {
                double sum_new = 0;
                int ro = blockrowid_new[ri + 1];
                if (blockcsr_ptr_new[ri + 1] > blockcsr_ptr_new[ri])
                {
                    cnt = cnt + (blockcsr_ptr_new[ri + 1] - blockcsr_ptr_new[ri]);
                }
                for (int rj = blockcsr_ptr_new[ri]; rj < blockcsr_ptr_new[ri + 1]; rj++)
                {
                    int csrcol = Tile_csr_Col[csroffset + rj];
                    sum_new += x[x_offset + csrcol] * matrix->Blockcsr_Val[csroffset + rj];
                }
                b_new[blki * BLOCK_SIZE + ro] += sum_new;
            }
        }
    }
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        double r = b_new[i] - b[i];
        sum = sum + (r * r);
    }
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (b[i] * b[i]);
    }
    double l2_norm = sqrt(sum) / sqrt(sum_ori);
    char *s = (char *)malloc(sizeof(char) * 200);
    sprintf(s, "%d,%.3f,%d,%e,%e\n", 100, time_cg, nnzR, l2_norm,sqrt(snew));
    FILE *file1 = fopen("data/cg_performance.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
    cudaFree(k_val);
    cudaFree(k_b);
    cudaFree(k_x);
    cudaFree(k_r);
    cudaFree(k_d);
    cudaFree(k_q);
    cudaFree(k_alpha);
    cudaFree(k_snew);
    cudaFree(k_sold);
    cudaFree(k_beta);
    cudaFree(k_s0);
    cudaFree(d_tile_ptr);
    cudaFree(d_tile_columnidx);
    cudaFree(d_csr_compressedIdx);
    cudaFree(d_Blockcsr_Val);
    cudaFree(d_Blockcsr_Ptr);
    cudaFree(d_blkcoostylerowidx);
    cudaFree(d_blkcoostylerowidx_colstart);
    cudaFree(d_blkcoostylerowidx_colstop);
    cudaFree(d_ptroffset1);
    cudaFree(d_ptroffset2);
    cudaFree(d_x);
    cudaFree(d_y);
    free(matrix);
    free(ptroffset1);
    free(ptroffset2);
    free(y_golden);
    free(y);
    free(blkcoostylerowidx);
    free(blkcoostylerowidx_colstart);
    free(blkcoostylerowidx_colstop);
    free(tile_ptr);
    free(tile_columnidx);
    free(tile_nnz);
    free(csr_offset);
    free(csrptr_offset);
    free(Blockcsr_Val);
    free(Blockcsr_Val_Low);
    free(csr_compressedIdx);
    free(Blockcsr_Ptr);
}

int main(int argc, char **argv)
{
    char *filename = argv[1];
    int m, n, nnzR, isSymmetric;
    int *RowPtr;
    int *ColIdx;
    MAT_VAL_TYPE *Val;
    read_Dmatrix_32(&m, &n, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
    if (m != n)
        return 0;
    MAT_VAL_LOW_TYPE *Val_Low = (MAT_VAL_LOW_TYPE *)malloc(sizeof(MAT_VAL_LOW_TYPE) * nnzR);
    for (int i = 0; i < nnzR; i++)
    {
        Val_Low[i] = Val[i];
    }
    int ori = n;
    n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    m = (m / BLOCK_SIZE) * BLOCK_SIZE;
    MAT_VAL_TYPE *X = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (n));
    MAT_VAL_TYPE *Y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (m));
    MAT_VAL_TYPE *Y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (m));
    memset(X, 0, sizeof(MAT_VAL_TYPE) * (n));
    memset(Y, 0, sizeof(MAT_VAL_TYPE) * (n));
    memset(Y_golden, 0, sizeof(MAT_VAL_TYPE) * (n));

    for (int i = 0; i < n; i++)
    {
        X[i] = 1;
    }
    int iter = 0;
    for (int i = 0; i < n; i++)
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            Y_golden[i] += Val[j] * X[ColIdx[j]];
        }
    cg_solve_inc(RowPtr, ColIdx, Val, Val_Low, X, Y_golden, n, &iter, 10, 1e-5, filename, nnzR, ori);
}
