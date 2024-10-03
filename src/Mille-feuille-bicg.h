#include <omp.h>
#include "utils.h"
#include "common.h"

#define min(a, b) ((a < b) ? (a) : (b))
#define NUM_THREADS 128
#define NUM_BLOCKS 16
#define THREAD_ID threadIdx.x + blockIdx.x *blockDim.x
#define THREAD_COUNT gridDim.x *blockDim.x
#define epsilon 1e-6
#define IMAX 1000
double itsol_norm(double *x, int n, int nthread)
{
    int i;
    double t = 0.;
    for (i = 0; i < n; i++)
        t += x[i] * x[i];

    return sqrt(t);
}
double itsol_dot(double *x, double *y, int n, int nthread)
{
    int i;
    double t = 0.;
    for (i = 0; i < n; i++)
        t += x[i] * y[i];

    return t;
}
void mv(int n, int *Rowptr, int *ColIndex, double *Value, double *x, double *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0.0;
        for (int j = Rowptr[i]; j < Rowptr[i + 1]; j++)
        {
            int k = ColIndex[j];
            y[i] += Value[j] * x[k];
        }
    }
}

void scalarassign(double *dest, double *src)
{
    cudaMemcpy(dest, src, sizeof(double), cudaMemcpyDeviceToDevice);
}
__global__ void scalardiv(double *dest, double *src)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dest) = (*src) / (*dest);
    }
}
__global__ void scalardiv_new(double *dest, double *src1, double *src2)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dest) = (*src1) / (*src2);
    }
}
__global__ void scalardiv_five(double *prb, double *r1, double *pra, double *r0, double *prc)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*prb) = ((*r1) * (*pra)) / ((*r0) * (*prc));
    }
}
__global__ void yminus_mult(int n, double *sg, double *rg, double *vg, double *pra)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        sg[i] = rg[i] - (*pra) * vg[i];
}

__global__ void yminus_mult_new(int n, double *x, double *pg, double *sg, double *rg, double *tg, double *pra, double *prc)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
    {
        x[i] = x[i] + (*pra) * pg[i] + (*prc) * sg[i];
        rg[i] = sg[i] - (*prc) * tg[i];
    }
}
__global__ void yminus_final(int n, double *pg, double *rg, double *prb, double *prc, double *vg)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        pg[i] = rg[i] + (*prb) * (pg[i] - (*prc) * vg[i]);
}
__global__ void sdot2_2(double *a, double *b, double *c, int n)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double temp;
    temp = 0;
    __shared__ double s_data[256];
    unsigned int tid = threadIdx.x;
    for (int i = index; i < n; i += stride)
    {
        temp += (a[i] * b[i]);
    }
    s_data[tid] = temp;
    __syncthreads();
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
    if (tid == 0&&s_data[0]!=0)
    {
        atomicAdd(c, s_data[0]);
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr(int tilem, int tilen, int rowA, int colA, int nnzA,
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

__global__ void device_convert(double *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = x[tid];
    }
}


__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                         int *signal_dot1,
                                                         int *d_ori_block_signal,
                                                         double *k_rh,
                                                         double *k_pra,
                                                         double *k_r1,
                                                         double *k_sg,
                                                         double *k_rg,
                                                         double *k_tg,
                                                         double *k_tmp1,
                                                         double *k_tmp2,
                                                         double *k_x,
                                                         double *k_residual,
                                                         double *k_r_new,
                                                         double *k_r0,
                                                         int *d_balance_tile_ptr,
                                                         int *d_row_each_block,
                                                         int *d_index_each_block,
                                                         int balance_row,
                                                         int *d_non_each_block_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_pra[WARP_PER_BLOCK];
    __shared__ double s_prc[WARP_PER_BLOCK];
    __shared__ double s_prb[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
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
        for (int iter = 0; (iter < 50); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_r_new[0];
                s_pra[threadIdx.x] = 0;
                s_prc[threadIdx.x] = 0;
                s_prb[threadIdx.x] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
                k_tg[global_id] = 0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; 
            }
            __threadfence();
            if (global_id == 0)
            {
                k_r0[0]=k_r_new[0];
                signal_dot[0] = tilem; 
                signal_dot1[0] = 0;
                k_pra[0] = 0;
                k_tmp1[0] = 0;
                k_tmp2[0] = 0;
                signal_final[0] = 0;
                k_residual[0] = 0;
                k_r_new[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
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
                

                if ((lane_id < BLOCK_SIZE))
                {
                    atomicAdd(k_pra, (d_y_d[blki_blc * BLOCK_SIZE + lane_id] * k_rh[blki_blc * BLOCK_SIZE + lane_id]));
                }
                
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_pra[local_warp_id] = s_snew[local_warp_id] / k_pra[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_sg[blki_blc * BLOCK_SIZE + lane_id] = k_rg[blki_blc * BLOCK_SIZE + lane_id] - s_pra[local_warp_id] * d_y_d[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (k_sg[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&k_tg[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != d_ori_block_signal[blki_blc]);
                if ((lane_id < BLOCK_SIZE))
                {
                    atomicAdd(k_tmp1, (k_tg[blki_blc * BLOCK_SIZE + lane_id] * k_sg[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                    atomicAdd(k_tmp2, (k_tg[blki_blc * BLOCK_SIZE + lane_id] * k_tg[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                }
                if ((lane_id == 0))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);
                if (lane_id == 0)
                {
                    s_prc[local_warp_id] = k_tmp1[0] / k_tmp2[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_x[blki_blc * BLOCK_SIZE + lane_id] = k_x[blki_blc * BLOCK_SIZE + lane_id] + s_pra[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]+s_prc[local_warp_id]*k_sg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                    k_rg[blki_blc * BLOCK_SIZE + lane_id] = k_sg[blki_blc * BLOCK_SIZE + lane_id] - s_prc[local_warp_id]*k_tg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if ((lane_id < BLOCK_SIZE))
                {
                    atomicAdd(k_residual, (k_rg[blki_blc * BLOCK_SIZE + lane_id] * k_rg[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                    atomicAdd(k_r_new, (k_rg[blki_blc * BLOCK_SIZE + lane_id] * k_rh[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot1, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot1[0] != tilem);
                if (lane_id == 0)
                {
                    s_prb[local_warp_id] = (k_r_new[0]*s_pra[local_warp_id]) / (k_r0[0]*s_prc[local_warp_id]);
                }
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                {
                    d_x_d[blki_blc * BLOCK_SIZE + lane_id]=k_rg[blki_blc * BLOCK_SIZE + lane_id]+(s_prb[local_warp_id])*(d_x_d[blki_blc * BLOCK_SIZE + lane_id]-(s_prc[local_warp_id]*d_y_d[blki_blc * BLOCK_SIZE + lane_id]));
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != 0);
        }
    }
}



__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                         int *signal_dot1,
                                                         int *d_ori_block_signal,
                                                         double *k_rh,
                                                         double *k_pra,
                                                         double *k_sg,
                                                         double *k_rg,
                                                         double *k_tg,
                                                         double *k_tmp1,
                                                         double *k_tmp2,
                                                         double *k_x,
                                                         double *k_residual,
                                                         double *k_r_new,
                                                         double *k_r0,
                                                         int *d_balance_tile_ptr,
                                                         int *d_row_each_block,
                                                         int *d_index_each_block,
                                                         int balance_row,
                                                         int *d_non_each_block_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot3[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot3_val = &s_dot3[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot4[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot4_val = &s_dot4[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot5[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot5_val = &s_dot5[local_warp_id * BLOCK_SIZE];
    
    
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_pra[WARP_PER_BLOCK];
    __shared__ double s_prc[WARP_PER_BLOCK];
    __shared__ double s_prb[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
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
        int offset=blki_blc * BLOCK_SIZE;
        for (int iter = 0; (iter < 50); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_r_new[0];
                s_pra[threadIdx.x] = 0;
                s_prc[threadIdx.x] = 0;
                s_prb[threadIdx.x] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
                k_tg[global_id] = 0;
            }
            __threadfence();
            if (lane_id < BLOCK_SIZE)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
                s_dot3_val[lane_id] = 0.0;
                s_dot4_val[lane_id] = 0.0;
                s_dot5_val[lane_id] = 0.0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; 
            }
            __threadfence();
            if (global_id == 0)
            {
                k_r0[0]=k_r_new[0];
                signal_dot[0] = tilem;
                signal_dot1[0] = 0;
                k_pra[0] = 0;
                k_tmp1[0] = 0;
                k_tmp2[0] = 0;
                signal_final[0] = 0;
                k_residual[0] = 0;
                k_r_new[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
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
                
                index_dot=offset + lane_id;
               
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id]+=(d_y_d[index_dot] * k_rh[index_dot]);
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
                    atomicAdd(k_pra, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_pra[local_warp_id] = s_snew[local_warp_id] / k_pra[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_sg[blki_blc * BLOCK_SIZE + lane_id] = k_rg[blki_blc * BLOCK_SIZE + lane_id] - s_pra[local_warp_id] * d_y_d[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (k_sg[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&k_tg[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != d_ori_block_signal[blki_blc]);
                
                index_dot=offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot2_val[lane_id]+=(k_tg[index_dot] * k_sg[index_dot]);
                    s_dot3_val[lane_id]+=(k_tg[index_dot] * k_tg[index_dot]);
                }
                __syncthreads();
                int i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                        s_dot3[threadIdx.x] += s_dot3[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_tmp1, s_dot2[0]);
                    atomicAdd(k_tmp2, s_dot3[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);
                if (lane_id == 0)
                {
                    s_prc[local_warp_id] = k_tmp1[0] / k_tmp2[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_x[blki_blc * BLOCK_SIZE + lane_id] = k_x[blki_blc * BLOCK_SIZE + lane_id] + s_pra[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]+s_prc[local_warp_id]*k_sg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                    k_rg[blki_blc * BLOCK_SIZE + lane_id] = k_sg[blki_blc * BLOCK_SIZE + lane_id] - s_prc[local_warp_id]*k_tg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);

                
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot4_val[lane_id]+=(k_rg[index_dot]*k_rg[index_dot]);
                    s_dot5_val[lane_id]+=(k_rg[index_dot]*k_rh[index_dot]);
                }
                __syncthreads();
                i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot4[threadIdx.x] += s_dot4[threadIdx.x + i];
                        s_dot5[threadIdx.x] += s_dot5[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_residual, s_dot4[0]);
                    atomicAdd(k_r_new, s_dot5[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot1, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot1[0] != tilem);
                if (lane_id == 0)
                {
                    s_prb[local_warp_id] = (k_r_new[0]*s_pra[local_warp_id]) / (k_r0[0]*s_prc[local_warp_id]);
                }
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                {
                    d_x_d[blki_blc * BLOCK_SIZE + lane_id]=k_rg[blki_blc * BLOCK_SIZE + lane_id]+(s_prb[local_warp_id])*(d_x_d[blki_blc * BLOCK_SIZE + lane_id]-(s_prc[local_warp_id]*d_y_d[blki_blc * BLOCK_SIZE + lane_id]));
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != 0);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                         int *signal_dot1,
                                                         int *d_ori_block_signal,
                                                         double *k_rh,
                                                         double *k_pra,
                                                         double *k_r1,
                                                         double *k_sg,
                                                         double *k_rg,
                                                         double *k_tg,
                                                         double *k_tmp1,
                                                         double *k_tmp2,
                                                         double *k_x,
                                                         double *k_residual,
                                                         double *k_r_new,
                                                         double *k_r0,
                                                         int *d_balance_tile_ptr,
                                                         int *d_row_each_block,
                                                         int *d_index_each_block,
                                                         int balance_row,
                                                         int *d_non_each_block_offset,
                                                         int vector_each_warp,
                                                         int vector_total
                                                         )
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_pra[WARP_PER_BLOCK];
    __shared__ double s_prc[WARP_PER_BLOCK];
    __shared__ double s_prb[WARP_PER_BLOCK];

    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    __shared__ double s_dot3[WARP_PER_BLOCK * 32];
    double *s_dot3_val = &s_dot3[local_warp_id * 32];
    __shared__ double s_dot4[WARP_PER_BLOCK * 32];
    double *s_dot4_val = &s_dot4[local_warp_id * 32];
    __shared__ double s_dot5[WARP_PER_BLOCK * 32];
    double *s_dot5_val = &s_dot5[local_warp_id * 32];

    if (blki_blc < balance_row)
    {
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
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
        int index_dot;
        int offset=blki_blc * vector_each_warp;
        int csrcol;
        int u;
        for (int iter = 0; (iter < 50); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_r_new[0];
                s_pra[threadIdx.x] = 0;
                s_prc[threadIdx.x] = 0;
                s_prb[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
                s_dot3_val[lane_id] = 0.0;
                s_dot4_val[lane_id] = 0.0;
                s_dot5_val[lane_id] = 0.0;
            }
            __syncthreads();
            __threadfence();
            
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id];
            }
            __threadfence();
            if (global_id == 0)
            {
                k_r0[0]=k_r_new[0];
                signal_dot[0] = vector_total; 
                signal_dot1[0] = 0;
                k_pra[0] = 0;
                k_tmp1[0] = 0;
                k_tmp2[0] = 0;
                signal_final[0] = 0;
                k_residual[0] = 0;
                k_r_new[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicSub(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < vector_total)
            {
                for(u = 0; u < vector_each_warp; u++)
                {
                    int off=blki_blc * vector_each_warp*2;
                    do
                    {
                        __threadfence_system();
                    }  while (d_block_signal[(off + u)] != 0);
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * k_rh[index_dot]);
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
                    atomicAdd(k_pra, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_pra[local_warp_id] = s_snew[local_warp_id] / k_pra[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    k_sg[index_dot] = k_rg[index_dot] - s_pra[local_warp_id] * d_y_d[index_dot];
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (k_sg[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&k_tg[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < vector_total)
            {
                for(u = 0; u < vector_each_warp; u++)
                {
                    int off=blki_blc * vector_each_warp*2;
                    do
                    {
                        __threadfence_system();
                    }  while (d_block_signal[(off + u)] != d_ori_block_signal[(off + u)]);
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    s_dot2_val[lane_id]+=k_tg[index_dot]*k_sg[index_dot];
                    s_dot3_val[lane_id]+=k_tg[index_dot]*k_tg[index_dot];
                }
                __syncthreads();
                int i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot1[threadIdx.x + i];
                        s_dot3[threadIdx.x] += s_dot3[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_tmp1, s_dot2[0]);
                    atomicAdd(k_tmp2, s_dot3[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);
                if (lane_id == 0)
                {
                    s_prc[local_warp_id] = k_tmp1[0] / k_tmp2[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_pra[local_warp_id] * d_x_d[index_dot]+s_prc[local_warp_id]*k_sg[index_dot];
                    __threadfence();
                    k_rg[index_dot] = k_sg[index_dot] - s_prc[local_warp_id]*k_tg[index_dot];
                    __threadfence();
                }
                if (lane_id == 0)
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    s_dot4_val[lane_id]+=(k_rg[index_dot]*k_rg[index_dot]);
                    s_dot5_val[lane_id]+=(k_rg[index_dot]*k_rh[index_dot]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot4[threadIdx.x] += s_dot4[threadIdx.x + i];
                        s_dot5[threadIdx.x] += s_dot5[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }

                if (threadIdx.x == 0)
                {
                    atomicAdd(k_residual, s_dot4[0]);
                    atomicAdd(k_r_new, s_dot5[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot1, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot1[0] != vector_total);
                 if (lane_id == 0)
                {
                    s_prb[local_warp_id] = (k_r_new[0]*s_pra[local_warp_id]) / (k_r0[0]*s_prc[local_warp_id]);
                }
                __threadfence();
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    d_x_d[index_dot]=k_rg[index_dot]+(s_prb[local_warp_id])*(d_x_d[index_dot]-(s_prc[local_warp_id]*d_y_d[index_dot]));
                    __threadfence();
                    d_y_d[index_dot]=0.0;
                    k_tg[index_dot]=0.0;
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != 0);
        }
    }
}
