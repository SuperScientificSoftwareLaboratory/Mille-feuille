#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include "./biio2.0/src/biio.h"
#include "Mille-feuille-bicg.h"
#include "csr2block.h"
#include "blockspmv_cpu.h"

int bicgstab_sync(int *RowPtr, int *ColIdx, double *Val, double *rhs, double *x,
             int nnzR, char *filename, int n) 
{
    int nnz = nnzR;
    float *Val_Low = (float *)malloc(sizeof(float) * nnz);
    for (int i = 0; i < nnz; i++)
    {
        Val_Low[i] = (float)Val[i];
    }
    int colA = n;
    n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    int rowA = n;
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
                 rowA, colA, nnz,
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
    cudaMalloc((void **)&d_Blockcsr_Val_float, (csrsize) * sizeof(float));
    cudaMemcpy(d_Blockcsr_Val_float, Blockcsr_Val_Low, (csrsize) * sizeof(float), cudaMemcpyHostToDevice);
  

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
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
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
    int *block_signal = (int *)malloc(sizeof(int) * (tilem + 1));
    memset(block_signal, 0, sizeof(int) * (tilem + 1)); 
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


    int *non_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        
    int *non_each_block_offset = (int *)malloc(sizeof(int) * (tilenum + 1)); 
    int *row_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));       
    int *index_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));      
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
    int *row_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));  
    int *index_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1)); 
    int *non_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(row_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(index_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(non_each_block_new, 0, sizeof(int) * (tilenum + 1));
    int each_block_nnz = 16;

    int cnt = 0;
    int balance_row = 0;
    int index = 1;
    
    int block_per_warp=180;
    int i = 0;
    int j = tilenum - 1;
    cnt = 0;
    index = 1;
    int step = 0;
    int cnt_block1=0;
    int nnz_list[12]={16,32,64,96,128,256,512,1024,2048,4096,nnzR/6912};
    while(1)
    {
    for(int k=0;k<12;k++)
    {
    each_block_nnz=nnz_list[k];
    i = 0;
    j = tilenum - 1;
    cnt = 0;
    index = 1;
    step = 0;
    cnt_block1=0;
    while (i < j)
    {
        if (((non_each_block[i] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[i];
            i++;
            cnt_block1++;
        }
        else if (((non_each_block[i] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            i++;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
        if (((non_each_block[j] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[j];
            j--;
            cnt_block1++;
        }
        else if (((non_each_block[j] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            j--;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
    }
    if(index<6912)
    break;
    }
    if(index<6912)
    break;
    block_per_warp=block_per_warp*2;
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
        vector_each_warp_32 = vector_each_warp_16*2;
        vector_total_32 = tilem_32 / vector_each_warp_32;
        vector_total_32 = (vector_total_32/WARP_PER_BLOCK)*WARP_PER_BLOCK;
    }
    if (index > 6912||index==0||tilem==0)
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
    cnt_block1=0;
    while (i < j)
    {
        if (((non_each_block[i] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[i];
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[i] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
         if (((non_each_block[j] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[j];
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[j] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1=0;
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
    int shared_nnz_each_block=256;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            int blkj=index_each_block_new[j];
            if (j == balance_tile_ptr_new[i])
                non_each_block_offset[j] = 0;
            cnt_nnz += non_each_block_new[j];
            cnt_block++;
            if (j != balance_tile_ptr_new[i] && cnt_nnz <=shared_nnz_each_block)
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
    int cnt_nnz_shared = 0;
    int cnt_nnz_total = 0;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        cnt_nnz_shared = 0;
        cnt_nnz_total = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz_total += non_each_block_new[j];
        }
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_shared_end[i + 1]; j++)
        {
            cnt_nnz_shared += non_each_block_new[j];
        }
        for (int j = balance_tile_ptr_shared_end[i + 1]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz += non_each_block_new[j];
        }
    }
    int *d_non_each_block_offset;
    cudaMalloc((void **)&d_non_each_block_offset, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_non_each_block_offset, non_each_block_offset, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);

    int *d_balance_tile_ptr_shared_end;
    cudaMalloc((void **)&d_balance_tile_ptr_shared_end, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_shared_end, balance_tile_ptr_shared_end, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_block_signal;
    cudaMalloc((void **)&d_block_signal, sizeof(int) * (tilem + 1));
    int *signal_dot;
    cudaMalloc((void **)&signal_dot, sizeof(int));
    int *signal_final;
    cudaMalloc((void **)&signal_final, sizeof(int));
    int *signal_final1;
    cudaMalloc((void **)&signal_final1, sizeof(int));
    double *k_threshold;
    cudaMalloc((void **)&k_threshold, sizeof(double));
    int *d_ori_block_signal;
    cudaMalloc((void **)&d_ori_block_signal, sizeof(int) * (tilem + 1));
    cudaMemcpy(d_block_signal, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ori_block_signal, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
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
    double time_cg = 0;
    double time_spmv = 0;
    double time_sptrsv = 0;
    struct timeval t1, t2, t3, t4, t5, t6;
    double *rg, *rh, *pg, *ph, *sg, *sh, *tg, *vg, *tp;
    double *k_rg, *k_rh, *k_pg, *k_ph, *k_sg, *k_sh, *k_tg, *k_vg, *k_tp;
    float  *k_vg_float;
    float  *k_tg_float;
    double *k_pg_last;
    double *k_sg_last;
    float *k_pg_float,*k_sg_float;
    double r0 = 0, r1 = 0, pra = 0, prb = 0, prc = 0;
    double residual, err_rel = 0;
    double *k_r0, *k_r1, *k_pra, *k_prb, *k_prc;
    double *k_residual, *k_err_rel;
    double *k_x;
    double *k_tmp1, *k_tmp2;
    int retval = 0;
    int itr = 0.;
    double tol = 1e-5;
    int maxits = 50;
    double *x_last = (double *)malloc(sizeof(double) * (n + 1));
    int nthreads = 8;
    for (int i = 0; i < n; i++)
    {
        rhs[i] = 0;
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            rhs[i] += Val[j];
        }
    }
    rg = (double *)malloc(n * sizeof(double));
    rh = (double *)malloc(n * sizeof(double));
    pg = (double *)malloc(n * sizeof(double));
    ph = (double *)malloc(n * sizeof(double));
    sg = (double *)malloc(n * sizeof(double));
    sh = (double *)malloc(n * sizeof(double));
    tg = (double *)malloc(n * sizeof(double));
    vg = (double *)malloc(n * sizeof(double));
    tp = (double *)malloc(n * sizeof(double));
    cudaMalloc((void **)&k_rg, sizeof(double) * n);
    cudaMalloc((void **)&k_rh, sizeof(double) * n);
    cudaMalloc((void **)&k_pg, sizeof(double) * n);
    cudaMalloc((void **)&k_pg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_sg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_pg_last, sizeof(double) * n);
    cudaMalloc((void **)&k_ph, sizeof(double) * n);
    cudaMalloc((void **)&k_sg, sizeof(double) * n);
    cudaMalloc((void **)&k_sg_last, sizeof(double) * n);
    cudaMalloc((void **)&k_sh, sizeof(double) * n);
    cudaMalloc((void **)&k_tg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_tg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_tp, sizeof(double) * n);
    cudaMalloc((void **)&k_x, sizeof(double) * n);

    cudaMalloc((void **)&k_r0, sizeof(double));
    cudaMalloc((void **)&k_r1, sizeof(double));
    double *k_r_new;
    cudaMalloc((void **)&k_r_new, sizeof(double));
    cudaMemset(k_r_new,0,sizeof(double));
    cudaMalloc((void **)&k_pra, sizeof(double));
    cudaMalloc((void **)&k_prb, sizeof(double));
    cudaMalloc((void **)&k_prc, sizeof(double));
    cudaMalloc((void **)&k_residual, sizeof(double));
    cudaMalloc((void **)&k_err_rel, sizeof(double));
    cudaMalloc((void **)&k_tmp1, sizeof(double));
    cudaMalloc((void **)&k_tmp2, sizeof(double));
    int *k_findrm, *k_colm;
    double *k_val;
    cudaMalloc((void **)&k_findrm, sizeof(int) * (n + 1));
    cudaMemcpy(k_findrm, RowPtr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_colm, sizeof(int) * (nnzR));
    cudaMemcpy(k_colm, ColIdx, sizeof(int) * (nnzR), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_val, sizeof(double) * (nnzR));
    cudaMemcpy(k_val, Val, sizeof(double) * (nnzR), cudaMemcpyHostToDevice);
    mv(n, RowPtr, ColIdx, Val, x, tp);
    for (i = 0; i < n; i++)
        rg[i] = rhs[i] - tp[i];
    for (i = 0; i < n; i++)
    {
        rh[i] = rg[i];
        sh[i] = ph[i] = 0.;
    }
    int *vis_pg = (int *)malloc(sizeof(int) * n);
    int *vis_sg = (int *)malloc(sizeof(int) * n);
    residual = err_rel = itsol_norm(rg, n, nthreads);
    tol = residual * fabs(tol);
    // int cnt_pg=0;
    for (i = 0; i < n; i++)
        pg[i] = rg[i];
    r1 = itsol_dot(rg, rh, n, nthreads);
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);
    cudaMemcpy(k_pg, pg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rg, rg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rh, rh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sh, sh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_ph, ph, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sg, sg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tg, tg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_vg, vg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tp, tp, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_x, x, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r0, &r0, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r1, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r_new, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_pra, &pra, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prb, &prb, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prc, &prc, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_residual, &residual, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_err_rel, &err_rel, sizeof(double), cudaMemcpyHostToDevice);
    int *d_block_signal_new;
    int *d_ori_block_signal_new;
    double *k_vg_new;
    double *k_pg_new;
    double *k_rh_new;
    double *k_sg_new;
    double *k_rg_new;
    double *k_tg_new;
    double *k_x_new;
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    {
        int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
        if(index>=tilem)
        {
            int tilem_new=(tilem/WARP_PER_BLOCK+2)*WARP_PER_BLOCK;
            int re_size=(tilem_new)*BLOCK_SIZE;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&k_vg_new, sizeof(double) * re_size);
            cudaMalloc((void **)&k_pg_new, sizeof(double) * re_size);
            cudaMemset(k_pg_new, 0,  re_size* sizeof(double));
            cudaMemcpy(k_pg_new, k_pg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_rh_new, sizeof(double) * re_size);
            cudaMemcpy(k_rh_new, k_rh, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_sg_new, sizeof(double) * re_size);
            cudaMemcpy(k_sg_new, k_sg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_rg_new, sizeof(double) * re_size);
            cudaMemcpy(k_rg_new, k_rg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_tg_new, sizeof(double) * re_size);
            cudaMemcpy(k_tg_new, k_tg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_x_new, sizeof(double) * re_size);
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_pg_new, k_vg_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                              k_rh_new,k_pra,k_sg_new,k_rg_new,k_tg_new,k_tmp1,k_tmp2,k_x_new,k_residual,k_r_new,k_r0,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset);
            cudaDeviceSynchronize();
            gettimeofday(&t4, NULL);
            time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
            
        }
        else
        {
            if(vector_each_warp_32*vector_total_32*32>rowA)
            {
                rowA=vector_each_warp_32*vector_total_32*32;
            }
            int tilem_new=rowA/BLOCK_SIZE;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&k_vg_new, sizeof(double) * rowA);
            cudaMalloc((void **)&k_pg_new, sizeof(double) * rowA);
            cudaMemset(k_pg_new, 0,  rowA* sizeof(double));
            cudaMemcpy(k_pg_new, k_pg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_rh_new, sizeof(double) * rowA);
            cudaMemcpy(k_rh_new, k_rh, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_sg_new, sizeof(double) * rowA);
            cudaMemcpy(k_sg_new, k_sg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_rg_new, sizeof(double) * rowA);
            cudaMemcpy(k_rg_new, k_rg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_tg_new, sizeof(double) * rowA);
            cudaMemcpy(k_tg_new, k_tg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_x_new, sizeof(double) * rowA);
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_pg_new, k_vg_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                              k_rh_new,k_pra,k_r1,k_sg_new,k_rg_new,k_tg_new,k_tmp1,k_tmp2,k_x_new,k_residual,k_r_new,k_r0,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,vector_each_warp_32,vector_total_32);
            cudaDeviceSynchronize();
            gettimeofday(&t4, NULL);
            time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        }
        cudaMemcpy(&residual,k_residual,sizeof(double),cudaMemcpyDeviceToHost);
    }
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    double Gflops_spmv= (2 * nnzR) / ((time_spmv/ itr/2)*pow(10, 6));
    double Gflops_bicg= (2 * nnzR) / ((time_cg/ itr) * pow(10, 6));
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (rhs[i] * rhs[i]);
    }
    double l2_norm = sqrt(residual) / sqrt(sum_ori);
    char *s = (char *)malloc(sizeof(char) * 100);
    printf("time_bicg=%lf ms\n", time_spmv);
    sprintf(s, "%d,%.3f,%d,%e,%e\n", 50, time_spmv, nnzR,l2_norm,residual/err_rel);
    FILE *file1 = fopen("data/bicg_performance.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return 0;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    fclose(file1);
    cudaFree(d_ori_block_signal_new);
    cudaFree(d_ori_block_signal_new);
    cudaFree(k_vg_new);
    cudaFree(k_pg_new);
    cudaFree(k_rh_new);
    cudaFree(k_sg_new);
    cudaFree(k_rg_new);
    cudaFree(k_tg_new);
    cudaFree(k_x_new);
    free(rg);
    free(rh);
    free(pg);
    free(ph);
    free(sg);
    free(sh);
    free(tg);
    free(tp);
    free(vg);
    if (itr >= maxits)
        retval = 1;
    return retval;
}
int bicgstab_inc(int *RowPtr, int *ColIdx, double *Val, double *rhs, double *x,
             int nnzR, char *filename, int n) // bicg
{
    int nnz = nnzR;
    float *Val_Low = (float *)malloc(sizeof(float) * nnz);
    for (int i = 0; i < nnz; i++)
    {
        Val_Low[i] = (float)Val[i];
    }
    int colA = n;
    n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    int rowA = n;
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
                 rowA, colA, nnz,
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
    cudaMalloc((void **)&d_Blockcsr_Val_float, (csrsize) * sizeof(float));
    cudaMemcpy(d_Blockcsr_Val_float, Blockcsr_Val_Low, (csrsize) * sizeof(float), cudaMemcpyHostToDevice);
  

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
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
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
    int *block_signal = (int *)malloc(sizeof(int) * (tilem + 1));
    memset(block_signal, 0, sizeof(int) * (tilem + 1)); // 记录块数
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


    int *non_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        
    int *non_each_block_offset = (int *)malloc(sizeof(int) * (tilenum + 1)); 
    int *row_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        
    int *index_each_block = (int *)malloc(sizeof(int) * (tilenum + 1)); 
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

    //printf("nnz_total=%d each_block_nnz=%d\n", nnz_total, each_block_nnz);
    int cnt = 0;
    int balance_row = 0;
    int index = 1;
    int block_per_warp=180;
    int i = 0;
    int j = tilenum - 1;
    cnt = 0;
    index = 1;
    int step = 0;
    int cnt_block1=0;
    int nnz_list[12]={16,32,64,96,128,256,512,1024,2048,4096,nnzR/6912};
    while(1)
    {
    for(int k=0;k<12;k++)
    {
    each_block_nnz=nnz_list[k];
    i = 0;
    j = tilenum - 1;
    cnt = 0;
    index = 1;
    step = 0;
    cnt_block1=0;
    while (i < j)
    {
        if ((non_each_block[i] + cnt) < each_block_nnz)
        {
            cnt += non_each_block[i];
            i++;
        }
        else if ((non_each_block[i] + cnt) >= each_block_nnz)
        {
            i++;
            index++;
            cnt = 0;
        }
        if ((non_each_block[j] + cnt) < each_block_nnz)
        {
            cnt += non_each_block[j];
            j--;
        }
        else if ((non_each_block[j] + cnt) >= each_block_nnz)
        {
            j--;
            index++;
            cnt = 0;
        }
    }
    if(index<6912)
    break;
    }
    if(index<6912)
    break;
    block_per_warp=block_per_warp*2;
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
        vector_each_warp_32 = vector_each_warp_16*2;
        vector_total_32 = tilem_32 / vector_each_warp_32;
        vector_total_32 = (vector_total_32/WARP_PER_BLOCK)*WARP_PER_BLOCK;
    }
    if (index > 6912)
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
    while (i < j)
    {
        if ((non_each_block[i] + cnt) < each_block_nnz)
        {
            cnt += non_each_block[i];
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
        }
        else if ((non_each_block[i] + cnt) >= each_block_nnz)
        {
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
        }
        if ((non_each_block[j] + cnt) < each_block_nnz)
        {
            cnt += non_each_block[j];
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
        }
        else if ((non_each_block[j] + cnt) >= each_block_nnz)
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
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
    // 双指针的划分方式
    int cnt_block = 0;
    int cnt_nnz = 0;
    for (int i = 0; i <= index; i++)
    {
        balance_tile_ptr_shared_end[i] = balance_tile_ptr_new[i];
    }
    int shared_nnz_each_block=256;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            int blkj=index_each_block_new[j];
            if (j == balance_tile_ptr_new[i])
                non_each_block_offset[j] = 0;
            cnt_nnz += non_each_block_new[j];
            cnt_block++;
            if (j != balance_tile_ptr_new[i] && cnt_nnz <=256)
            {
                non_each_block_offset[j] = non_each_block_new[j - 1];
                non_each_block_offset[j] += non_each_block_offset[j - 1];
            }
            if (cnt_nnz > 256)
            {
                balance_tile_ptr_shared_end[i + 1] = j;
                break;
            }
        }
    }
    int cnt_nnz_shared = 0;
    int cnt_nnz_total = 0;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        cnt_nnz_shared = 0;
        cnt_nnz_total = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz_total += non_each_block_new[j];
        }
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_shared_end[i + 1]; j++)
        {
            cnt_nnz_shared += non_each_block_new[j];
        }
        for (int j = balance_tile_ptr_shared_end[i + 1]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz += non_each_block_new[j];
        }
    }
    int *d_non_each_block_offset;
    cudaMalloc((void **)&d_non_each_block_offset, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_non_each_block_offset, non_each_block_offset, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);

    int *d_balance_tile_ptr_shared_end;
    cudaMalloc((void **)&d_balance_tile_ptr_shared_end, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_shared_end, balance_tile_ptr_shared_end, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_block_signal;
    cudaMalloc((void **)&d_block_signal, sizeof(int) * (tilem + 1));
    int *signal_dot;
    cudaMalloc((void **)&signal_dot, sizeof(int));
    int *signal_final;
    cudaMalloc((void **)&signal_final, sizeof(int));
    int *signal_final1;
    cudaMalloc((void **)&signal_final1, sizeof(int));
    double *k_threshold;
    cudaMalloc((void **)&k_threshold, sizeof(double));
    int *d_ori_block_signal;
    cudaMalloc((void **)&d_ori_block_signal, sizeof(int) * (tilem + 1));
    cudaMemcpy(d_block_signal, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ori_block_signal, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
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
    double time_cg = 0;
    double time_spmv = 0;
    double time_sptrsv = 0;
    struct timeval t1, t2, t3, t4, t5, t6;
    double *rg, *rh, *pg, *ph, *sg, *sh, *tg, *vg, *tp;
    double *k_rg, *k_rh, *k_pg, *k_ph, *k_sg, *k_sh, *k_tg, *k_vg, *k_tp;
    float  *k_vg_float;
    float  *k_tg_float;
    double *k_pg_last;
    double *k_sg_last;
    float *k_pg_float,*k_sg_float;
    double r0 = 0, r1 = 0, pra = 0, prb = 0, prc = 0;
    double residual, err_rel = 0;
    double *k_r0, *k_r1, *k_pra, *k_prb, *k_prc;
    double *k_residual, *k_err_rel;
    double *k_x;
    double *k_tmp1, *k_tmp2;
    //int i; 
    int retval = 0;
    int itr = 0.;
    double tol = 1e-5;
    int maxits = 50;
    double *x_last = (double *)malloc(sizeof(double) * (n + 1));
    int nthreads = 8;
    for (int i = 0; i < n; i++)
    {
        rhs[i] = 0;
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            rhs[i] += Val[j];
        }
    }
    rg = (double *)malloc(n * sizeof(double));
    rh = (double *)malloc(n * sizeof(double));
    pg = (double *)malloc(n * sizeof(double));
    ph = (double *)malloc(n * sizeof(double));
    sg = (double *)malloc(n * sizeof(double));
    sh = (double *)malloc(n * sizeof(double));
    tg = (double *)malloc(n * sizeof(double));
    vg = (double *)malloc(n * sizeof(double));
    tp = (double *)malloc(n * sizeof(double));
    cudaMalloc((void **)&k_rg, sizeof(double) * n);
    cudaMalloc((void **)&k_rh, sizeof(double) * n);
    cudaMalloc((void **)&k_pg, sizeof(double) * n);
    cudaMalloc((void **)&k_pg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_sg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_pg_last, sizeof(double) * n);
    cudaMalloc((void **)&k_ph, sizeof(double) * n);
    cudaMalloc((void **)&k_sg, sizeof(double) * n);
    cudaMalloc((void **)&k_sg_last, sizeof(double) * n);
    cudaMalloc((void **)&k_sh, sizeof(double) * n);
    cudaMalloc((void **)&k_tg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_tg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_tp, sizeof(double) * n);
    cudaMalloc((void **)&k_x, sizeof(double) * n);

    cudaMalloc((void **)&k_r0, sizeof(double));
    cudaMalloc((void **)&k_r1, sizeof(double));
    double *k_r_new;
    cudaMalloc((void **)&k_r_new, sizeof(double));
    cudaMemset(k_r_new,0,sizeof(double));
    cudaMalloc((void **)&k_pra, sizeof(double));
    cudaMalloc((void **)&k_prb, sizeof(double));
    cudaMalloc((void **)&k_prc, sizeof(double));
    cudaMalloc((void **)&k_residual, sizeof(double));
    cudaMalloc((void **)&k_err_rel, sizeof(double));
    cudaMalloc((void **)&k_tmp1, sizeof(double));
    cudaMalloc((void **)&k_tmp2, sizeof(double));
    int *k_findrm, *k_colm;
    double *k_val;
    cudaMalloc((void **)&k_findrm, sizeof(int) * (n + 1));
    cudaMemcpy(k_findrm, RowPtr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_colm, sizeof(int) * (nnzR));
    cudaMemcpy(k_colm, ColIdx, sizeof(int) * (nnzR), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_val, sizeof(double) * (nnzR));
    cudaMemcpy(k_val, Val, sizeof(double) * (nnzR), cudaMemcpyHostToDevice);
    mv(n, RowPtr, ColIdx, Val, x, tp);
    for (i = 0; i < n; i++)
        rg[i] = rhs[i] - tp[i];
    for (i = 0; i < n; i++)
    {
        rh[i] = rg[i];
        sh[i] = ph[i] = 0.;
    }
    int *vis_pg = (int *)malloc(sizeof(int) * n);
    int *vis_sg = (int *)malloc(sizeof(int) * n);
    residual = err_rel = itsol_norm(rg, n, nthreads);
    tol = residual * fabs(tol);
    // int cnt_pg=0;
    for (i = 0; i < n; i++)
        pg[i] = rg[i];
    r1 = itsol_dot(rg, rh, n, nthreads);
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);
    cudaMemcpy(k_pg, pg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rg, rg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rh, rh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sh, sh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_ph, ph, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sg, sg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tg, tg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_vg, vg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tp, tp, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_x, x, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r0, &r0, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r1, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r_new, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_pra, &pra, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prb, &prb, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prc, &prc, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_residual, &residual, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_err_rel, &err_rel, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    //for (itr = 0; itr < maxits; itr++)
    {
        cudaDeviceSynchronize();
        gettimeofday(&t3, NULL);
        int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
        if(index>=tilem)
        {
            stir_spmv_cuda_kernel_newcsr_nnz_balance<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_pg, k_vg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal,
                                                                                              k_rh,k_pra,k_r1,k_sg,k_rg,k_tg,k_tmp1,k_tmp2,k_x,k_residual,k_r_new,k_r0,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset);
        }
        else
        {

            stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_pg, k_vg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal,
                                                                                              k_rh,k_pra,k_r1,k_sg,k_rg,k_tg,k_tmp1,k_tmp2,k_x,k_residual,k_r_new,k_r0,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,vector_each_warp_32,vector_total_32);

        }
        cudaDeviceSynchronize();
        gettimeofday(&t4, NULL);
        time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        cudaMemcpy(&residual,k_residual,sizeof(double),cudaMemcpyDeviceToHost);
    }
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    double Gflops_spmv= (2 * nnzR) / ((time_spmv/ itr/2)*pow(10, 6));
    double Gflops_bicg= (2 * nnzR) / ((time_cg/ itr) * pow(10, 6));
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (rhs[i] * rhs[i]);
    }
    double l2_norm = sqrt(residual) / sqrt(sum_ori);
    printf("time_bicg=%lf ms\n", time_cg);
    char *s = (char *)malloc(sizeof(char) * 100);
    sprintf(s, "%d,%.3f,%d,%e,%e\n", 50, time_spmv, nnzR,l2_norm,residual/err_rel);
    FILE *file1 = fopen("data/bicg_performance.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return 0;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    fclose(file1);
    free(rg);
    free(rh);
    free(pg);
    free(ph);
    free(sg);
    free(sh);
    free(tg);
    free(tp);
    free(vg);
    if (itr >= maxits)
        retval = 1;
    return retval;
}
int bicgstab(int *RowPtr, int *ColIdx, double *Val, double *rhs, double *x,
             int nnzR, char *filename, int n) // bicg
{
    int nnz = nnzR;
    float *Val_Low = (float *)malloc(sizeof(float) * nnz);
    for (int i = 0; i < nnz; i++)
    {
        Val_Low[i] = (float)Val[i];
    }
    int colA = n;
    n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    int rowA = n;
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
                 rowA, colA, nnz,
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
    //printf("rowA=%d,rowblkblock=%d\n", tilem, rowblkblock);
    char *tilewidth = matrix->tilewidth;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
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
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
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

    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
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
    double time_bicg = 0;
    double time_spmv = 0;
    double time_dot = 0;
    double time_sptrsv = 0;
    struct timeval t1, t2, t3, t4, t5, t6,t7,t8;
    double *rg, *rh, *pg, *ph, *sg, *sh, *tg, *vg, *tp;
    double *k_rg, *k_rh, *k_pg, *k_ph, *k_sg, *k_sh, *k_tg, *k_vg, *k_tp;
    double r0 = 0, r1 = 0, pra = 0, prb = 0, prc = 0;
    double residual, err_rel = 0;
    double *k_r0, *k_r1, *k_pra, *k_prb, *k_prc;
    double *k_residual, *k_err_rel;
    double *k_x;
    double *k_tmp1, *k_tmp2;
    int i, retval = 0;
    int itr = 0.;
    double tol = 1e-10;
    int maxits = 1000;
    double *x_last = (double *)malloc(sizeof(double) * (n + 1));
    int nthreads = 8;
    for (int i = 0; i < n; i++)
    {
        rhs[i] = 0;
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            rhs[i] += Val[j];
        }
    }
    rg = (double *)malloc(n * sizeof(double));
    rh = (double *)malloc(n * sizeof(double));
    pg = (double *)malloc(n * sizeof(double));
    ph = (double *)malloc(n * sizeof(double));
    sg = (double *)malloc(n * sizeof(double));
    sh = (double *)malloc(n * sizeof(double));
    tg = (double *)malloc(n * sizeof(double));
    vg = (double *)malloc(n * sizeof(double));
    tp = (double *)malloc(n * sizeof(double));
    cudaMalloc((void **)&k_rg, sizeof(double) * n);
    cudaMalloc((void **)&k_rh, sizeof(double) * n);
    cudaMalloc((void **)&k_pg, sizeof(double) * n);
    cudaMalloc((void **)&k_ph, sizeof(double) * n);
    cudaMalloc((void **)&k_sg, sizeof(double) * n);
    cudaMalloc((void **)&k_sh, sizeof(double) * n);
    cudaMalloc((void **)&k_tg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg, sizeof(double) * n);
    cudaMalloc((void **)&k_tp, sizeof(double) * n);
    cudaMalloc((void **)&k_x, sizeof(double) * n);

    cudaMalloc((void **)&k_r0, sizeof(double));
    cudaMalloc((void **)&k_r1, sizeof(double));
    cudaMalloc((void **)&k_pra, sizeof(double));
    cudaMalloc((void **)&k_prb, sizeof(double));
    cudaMalloc((void **)&k_prc, sizeof(double));
    cudaMalloc((void **)&k_residual, sizeof(double));
    cudaMalloc((void **)&k_err_rel, sizeof(double));
    cudaMalloc((void **)&k_tmp1, sizeof(double));
    cudaMalloc((void **)&k_tmp2, sizeof(double));
    mv(n, RowPtr, ColIdx, Val, x, tp);
    for (i = 0; i < n; i++)
        rg[i] = rhs[i] - tp[i];
    for (i = 0; i < n; i++)
    {
        rh[i] = rg[i];
        sh[i] = ph[i] = 0.;
    }
    int *vis_pg = (int *)malloc(sizeof(int) * n);
    int *vis_sg = (int *)malloc(sizeof(int) * n);
    residual = err_rel = itsol_norm(rg, n, nthreads);
    tol = residual * fabs(tol);
    // int cnt_pg=0;
    for (i = 0; i < n; i++)
        pg[i] = rg[i];
    r1 = itsol_dot(rg, rh, n, nthreads);


    dim3 BlockDim(256);
    dim3 GridDim((n/256+1));

    cudaMemcpy(k_pg, pg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rg, rg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rh, rh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sh, sh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_ph, ph, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sg, sg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tg, tg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_vg, vg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tp, tp, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_x, x, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r0, &r0, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r1, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_pra, &pra, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prb, &prb, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prc, &prc, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_residual, &residual, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_err_rel, &err_rel, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    //for (itr = 0; itr < maxits; itr++)
    for (itr = 0; itr < 10000; itr++)
    {
        scalarassign(k_r0, k_r1);
        cudaMemset(k_vg, 0, n * sizeof(double));
        cudaMemset(k_tg, 0, n * sizeof(double));
        stir_spmv_cuda_kernel_newcsr<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnz,
                                                                      d_tile_ptr, d_tile_columnidx,
                                                                      d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                      d_ptroffset1, d_ptroffset2,
                                                                      rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                      k_pg, k_vg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col);
  
        cudaMemset(k_pra, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_rh, k_vg, k_pra, n);
        scalardiv<<<1, 1>>>(k_pra, k_r1);
        yminus_mult<<<GridDim, BlockDim>>>(n, k_sg, k_rg, k_vg, k_pra);
        stir_spmv_cuda_kernel_newcsr<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnz,
                                                                    d_tile_ptr, d_tile_columnidx,
                                                                    d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                    d_ptroffset1, d_ptroffset2,
                                                                    rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                    k_sg, k_tg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col);
        
        cudaMemset(k_tmp1, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_tg, k_sg, k_tmp1, n);
        cudaMemset(k_tmp2, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_tg, k_tg, k_tmp2, n);
        scalardiv_new<<<1, 1>>>(k_prc, k_tmp1, k_tmp2);
        yminus_mult_new<<<GridDim, BlockDim>>>(n, k_x, k_pg, k_sg, k_rg, k_tg, k_pra, k_prc);
        cudaMemset(k_residual, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_rg, k_rg, k_residual, n);
        cudaMemcpy(&residual, k_residual, sizeof(double), cudaMemcpyDeviceToHost);
        residual=sqrt(residual);
        cudaMemset(k_r1, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_rg, k_rh, k_r1, n);
        scalardiv_five<<<1, 1>>>(k_prb, k_r1, k_pra, k_r0, k_prc);
        yminus_final<<<GridDim, BlockDim>>>(n, k_pg, k_rg, k_prb, k_prc, k_vg);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time_bicg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    double Gflops_spmv= (2 * nnzR) / ((time_spmv/ itr/2)*pow(10, 6));
    double Gflops_bicg= (2 * nnzR) / ((time_bicg/ itr) * pow(10, 6));
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (rhs[i] * rhs[i]);
    }
    double norm=residual;
    double l2_norm = residual / sqrt(sum_ori);
    printf("time_bicg=%lf ms\n", time_bicg);
    char *s = (char *)malloc(sizeof(char) * 200);
    sprintf(s, "%d,%.3f,%d,%e,%e\n", 50, time_bicg/200, nnzR,l2_norm,norm);
    FILE *file1 = fopen("data/bicg_performance.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return 0;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    fclose(file1);
    free(rg);
    free(rh);
    free(pg);
    free(ph);
    free(sg);
    free(sh);
    free(tg);
    free(tp);
    free(vg);
    if (itr >= maxits)
        retval = 1;
    return retval;
}
int main(int argc, char **argv)
{
    int n;
    char *filename = argv[1];
    int m, n_csr, nnzR, isSymmetric;
    FILE *p = fopen(filename, "r");
    int *RowPtr;
    int *ColIdx;
    double *Val;
    read_Dmatrix_32(&m, &n_csr, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
    double *x1 = (double *)malloc(m * sizeof(double));
    int nn;
    double *rhs1 = (double *)malloc(m * sizeof(double));

    for (int i = 0; i < m; i++)
    {
        x1[i] = 0.0;
        rhs1[i] = 1;
        int cc;
    }
    if(nnzR<10000)
    bicgstab_inc(RowPtr, ColIdx, Val, rhs1, x1, nnzR, filename, m);
    else if(nnzR<100000&&nnzR>=10000)
    bicgstab_sync(RowPtr, ColIdx, Val, rhs1, x1, nnzR, filename, m);
    else if(nnzR>=100000)
    bicgstab(RowPtr, ColIdx, Val, rhs1, x1, nnzR, filename, m);
    fclose(p);
    free(x1);
    free(rhs1);
    return 0;
}
