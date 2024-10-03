#include "common.h"
#include "encode.h"
#include "format.h"
#include "utils.h"

void convert_step1(Tile_matrix *matrix,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)
{

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    int *tile_ptr = matrix->tile_ptr;

    unsigned thread = omp_get_max_threads();
    char *flag_g = (char *)malloc(thread * tilen * sizeof(char));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *flag = flag_g + thread_id * tilen;
        memset(flag, 0, tilen * sizeof(char));
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
        for (int j = csrRowPtrA[start]; j < csrRowPtrA[end]; j++)
        {
            int jc = csrColIdxA[j] / BLOCK_SIZE;
            if (flag[jc] == 0)
            {
                flag[jc] = 1;
                tile_ptr[blki]++;
            }
        }
    }
    free(flag_g);
}

void convert_step2(Tile_matrix *matrix,
                   unsigned char *tile_csr_ptr,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)
{

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    int *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    unsigned thread = omp_get_max_threads();
    char *col_temp_g = (char *)malloc((thread * tilen) * sizeof(char));
    int *nnz_temp_g = (int *)malloc((thread * tilen) * sizeof(int));
    unsigned char *ptr_per_tile_g = (unsigned char *)malloc((thread * tilen * BLOCK_SIZE) * sizeof(unsigned char));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *col_temp = col_temp_g + thread_id * tilen;
        memset(col_temp, 0, tilen * sizeof(char));
        int *nnz_temp = nnz_temp_g + thread_id * tilen;
        memset(nnz_temp, 0, tilen * sizeof(int));
        unsigned char *ptr_per_tile = ptr_per_tile_g + thread_id * tilen * BLOCK_SIZE;
        memset(ptr_per_tile, 0, tilen * BLOCK_SIZE * sizeof(unsigned char));
        int pre_tile = tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;

        for (int ri = 0; ri < rowlen; ri++)
        {
            for (int j = csrRowPtrA[start + ri]; j < csrRowPtrA[start + ri + 1]; j++)
            {
                int jc = csrColIdxA[j] / BLOCK_SIZE;
                col_temp[jc] = 1;
                nnz_temp[jc]++;
                ptr_per_tile[jc * BLOCK_SIZE + ri]++;
            }
        }

        int count = 0;
        for (int blkj = 0; blkj < tilen; blkj++)
        {
            if (col_temp[blkj] == 1)
            {
                tile_columnidx[pre_tile + count] = blkj;
                tile_nnz[pre_tile + count] = nnz_temp[blkj];
                for (int ri = 0; ri < rowlen; ri++)
                {
                    tile_csr_ptr[(pre_tile + count) * BLOCK_SIZE + ri] = ptr_per_tile[blkj * BLOCK_SIZE + ri];
                }
                count++;
            }
        }
    }
    free(col_temp_g);
    free(nnz_temp_g);
    free(ptr_per_tile_g);
}

void convert_step3(Tile_matrix *matrix,
                   unsigned char *tile_csr_ptr,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA)

{
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    unsigned char *blknnznnz = matrix->blknnznnz;
    char *tilewidth = matrix->tilewidth;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int collen = tile_columnidx[tile_id] == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int nnztmp = tile_nnz[tile_id + 1] - tile_nnz[tile_id]; // the number of nnz of tile_id
            int nnzthreshold = rowlen * collen * 0.75;
            {
                Format[tile_id] = 0;
                blknnz[tile_id] = nnztmp;
                csr_offset[tile_id] = nnztmp;
                csrptr_offset[tile_id] = rowlen;
            }
        }
    }
}

void convert_step4(Tile_matrix *matrix,
                   unsigned char *tile_csr_ptr,
                   unsigned char *Blockcsr_Col,
                   int nnz_temp,
                   int tile_count_temp,
                   int rowA,
                   int colA,
                   MAT_PTR_TYPE nnzA,
                   MAT_PTR_TYPE *csrRowPtrA,
                   int *csrColIdxA,
                   MAT_VAL_TYPE *csrValA,
                   MAT_VAL_LOW_TYPE *csrValA_Low)

{
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    unsigned char *blknnznnz = matrix->blknnznnz;
    char *tilewidth = matrix->tilewidth;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    MAT_VAL_LOW_TYPE *Blockcsr_Val_Low = matrix->Blockcsr_Val_Low;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    unsigned char *Tile_csr_Col = matrix->Tile_csr_Col;
    unsigned thread = omp_get_max_threads();
    unsigned char *csr_colidx_temp_g = (unsigned char *)malloc((thread * nnz_temp) * sizeof(unsigned char));
    MAT_VAL_TYPE *csr_val_temp_g = (MAT_VAL_TYPE *)malloc((thread * nnz_temp) * sizeof(MAT_VAL_TYPE));
    MAT_VAL_LOW_TYPE *csr_val_temp_g_Low = (MAT_VAL_LOW_TYPE *)malloc((thread * nnz_temp) * sizeof(MAT_VAL_LOW_TYPE));
    int *tile_count_g = (int *)malloc(thread * tile_count_temp * sizeof(int));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {

        int thread_id = omp_get_thread_num();
        unsigned char *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_temp;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_temp;
        MAT_VAL_LOW_TYPE *csr_val_temp_low = csr_val_temp_g_Low + thread_id * nnz_temp;
        int *tile_count = tile_count_g + thread_id * tile_count_temp;
        memset(csr_colidx_temp, 0, (nnz_temp) * sizeof(unsigned char));
        memset(csr_val_temp, 0, (nnz_temp) * sizeof(MAT_VAL_TYPE));
        memset(csr_val_temp_low, 0, (nnz_temp) * sizeof(MAT_VAL_LOW_TYPE));
        memset(tile_count, 0, (tile_count_temp) * sizeof(int));
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
        for (int blkj = csrRowPtrA[start]; blkj < csrRowPtrA[end]; blkj++)
        {
            int jc_temp = csrColIdxA[blkj] / BLOCK_SIZE;
            for (int bi = 0; bi < tilenum_per_row; bi++)
            {
                int tile_id = tile_ptr[blki] + bi;
                int jc = tile_columnidx[tile_id];
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                if (jc == jc_temp)
                {
                    csr_val_temp[pre_nnz + tile_count[bi]] = csrValA[blkj];
                    csr_val_temp_low[pre_nnz + tile_count[bi]] = csrValA_Low[blkj];
                    csr_colidx_temp[pre_nnz + tile_count[bi]] = csrColIdxA[blkj] - jc * BLOCK_SIZE;
                    tile_count[bi]++;
                    break;
                }
            }
        }
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
            int nnztmp = tile_nnz[tile_id + 1] - tile_nnz[tile_id]; // blknnz[tile_id+1] - blknnz[tile_id] ;
            int collen = tile_columnidx[tile_id] == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int format = Format[tile_id];
            switch (format)
            {
            case 0:
            {
                int offset = csr_offset[tile_id];
                int ptr_offset = csrptr_offset[tile_id];

                unsigned char *ptr_temp = tile_csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan_char(ptr_temp, rowlen);

                for (int ri = 0; ri < rowlen; ri++)
                {
                    int start = ptr_temp[ri];
                    int stop = ri == rowlen - 1 ? nnztmp : ptr_temp[ri + 1];
                    ;
                    for (int k = start; k < stop; k++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                        Blockcsr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                        Blockcsr_Val_Low[offset + k] = csr_val_temp_low[pre_nnz + k];
                        Blockcsr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                        Tile_csr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                    }
                    Blockcsr_Ptr[ptr_offset + ri] = ptr_temp[ri];
                }
                break;
            }
            default:
                break;
            }
        }
    }
    free(csr_colidx_temp_g);
    free(csr_val_temp_g);
    free(tile_count_g);
    free(csr_val_temp_g_Low);
}

void Tile_create(Tile_matrix *matrix,
                 int rowA,
                 int colA,
                 MAT_PTR_TYPE nnzA,
                 MAT_PTR_TYPE *csrRowPtrA,
                 int *csrColIdxA,
                 MAT_VAL_TYPE *csrValA,
                 MAT_VAL_LOW_TYPE *csrValA_Low)
{

    struct timeval t1, t2;
    double time_conversion = 0;

    matrix->tilem = rowA % BLOCK_SIZE == 0 ? rowA / BLOCK_SIZE : (rowA / BLOCK_SIZE) + 1;
    matrix->tilen = colA % BLOCK_SIZE == 0 ? colA / BLOCK_SIZE : (colA / BLOCK_SIZE) + 1;
    matrix->tile_ptr = (int *)malloc((matrix->tilem + 1) * sizeof(int));
    memset(matrix->tile_ptr, 0, (matrix->tilem + 1) * sizeof(int));

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif
    convert_step1(matrix,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);

#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    exclusive_scan(matrix->tile_ptr, matrix->tilem + 1);
    matrix->tilenum = matrix->tile_ptr[matrix->tilem];
    int tilenum = matrix->tilenum;

    matrix->tile_columnidx = (int *)malloc(tilenum * sizeof(int));
    memset(matrix->tile_columnidx, 0, tilenum * sizeof(int));

    matrix->tile_nnz = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->tile_nnz, 0, (tilenum + 1) * sizeof(int));
    unsigned char *tile_csr_ptr = (unsigned char *)malloc((tilenum * BLOCK_SIZE) * sizeof(unsigned char));
    memset(tile_csr_ptr, 0, (tilenum * BLOCK_SIZE) * sizeof(unsigned char));

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif
    convert_step2(matrix, tile_csr_ptr,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);
#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif
    exclusive_scan(matrix->tile_nnz, tilenum + 1);

    matrix->Format = (char *)malloc(tilenum * sizeof(char));
    memset(matrix->Format, 0, tilenum * sizeof(char));
    matrix->blknnz = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->blknnz, 0, (tilenum + 1) * sizeof(int));
    matrix->blknnznnz = (unsigned char *)malloc((tilenum + 1) * sizeof(unsigned char));
    memset(matrix->blknnznnz, 0, (tilenum + 1) * sizeof(unsigned char));
    matrix->tilewidth = (char *)malloc(tilenum * sizeof(char));
    memset(matrix->tilewidth, 0, tilenum * sizeof(char));
    matrix->csr_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->csr_offset, 0, (tilenum + 1) * sizeof(int));
    matrix->csrptr_offset = (int *)malloc((tilenum + 1) * sizeof(int));
    memset(matrix->csrptr_offset, 0, (tilenum + 1) * sizeof(int));

#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif

    convert_step3(matrix, tile_csr_ptr,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA);
#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    exclusive_scan(matrix->csr_offset, tilenum + 1);
    exclusive_scan(matrix->csrptr_offset, tilenum + 1);

    matrix->csrsize = 0;
    matrix->csrptrlen = 0;

    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int rowlength = blki == matrix->tilem - 1 ? rowA - (matrix->tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int rowbnum = matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki];
        for (int bi = 0; bi < rowbnum; bi++)
        {
            int tile_id = matrix->tile_ptr[blki] + bi;
            char format = matrix->Format[tile_id];
            switch (format)
            {
            case 0: // csr
                matrix->csrsize += matrix->blknnz[tile_id];
                matrix->csrptrlen += rowlength;
                break;

            default:
                break;
            }
        }
    }

    for (int i = 0; i < tilenum + 1; i++)
        matrix->blknnznnz[i] = matrix->blknnz[i];

    exclusive_scan(matrix->blknnz, tilenum + 1);

    // CSR
    matrix->Blockcsr_Val = (MAT_VAL_TYPE *)malloc((matrix->csrsize) * sizeof(MAT_VAL_TYPE));
    memset(matrix->Blockcsr_Val, 0, (matrix->csrsize) * sizeof(MAT_VAL_TYPE));
    matrix->Blockcsr_Val_Low = (MAT_VAL_LOW_TYPE *)malloc((matrix->csrsize) * sizeof(MAT_VAL_LOW_TYPE));
    memset(matrix->Blockcsr_Val_Low, 0, (matrix->csrsize) * sizeof(MAT_VAL_LOW_TYPE));
    unsigned char *Blockcsr_Col_tmp = (unsigned char *)malloc((matrix->csrsize) * sizeof(unsigned char));
    memset(Blockcsr_Col_tmp, 0, (matrix->csrsize) * sizeof(unsigned char));
    matrix->Blockcsr_Ptr = (unsigned char *)malloc((matrix->csrptrlen) * sizeof(unsigned char));
    memset(matrix->Blockcsr_Ptr, 0, (matrix->csrptrlen) * sizeof(unsigned char));
    int compressed_csr_size = matrix->csrsize % 2 == 0 ? matrix->csrsize / 2 : matrix->csrsize / 2 + 1;
    matrix->csr_compressedIdx = (unsigned char *)malloc((compressed_csr_size) * sizeof(unsigned char));
    memset(matrix->csr_compressedIdx, 0, (compressed_csr_size) * sizeof(unsigned char));
    matrix->Tile_csr_Col = (unsigned char *)malloc((matrix->csrsize) * sizeof(unsigned char));
    memset(matrix->Tile_csr_Col, 0, (matrix->csrsize) * sizeof(unsigned char));
    


    int nnz_temp = 0;
    int tile_count_temp = 0;
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int start = blki * BLOCK_SIZE;
        int end = blki == matrix->tilem - 1 ? rowA : (blki + 1) * BLOCK_SIZE;
        nnz_temp = nnz_temp < csrRowPtrA[end] - csrRowPtrA[start] ? csrRowPtrA[end] - csrRowPtrA[start] : nnz_temp;
        tile_count_temp = tile_count_temp < matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] ? matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] : tile_count_temp;
    }
#if FORMAT_CONVERSION
    gettimeofday(&t1, NULL);
#endif

    convert_step4(matrix, tile_csr_ptr,
                  Blockcsr_Col_tmp,
                  nnz_temp, tile_count_temp,
                  rowA, colA, nnzA,
                  csrRowPtrA, csrColIdxA, csrValA,
                  csrValA_Low);

#if FORMAT_CONVERSION
    gettimeofday(&t2, NULL);
    time_conversion += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    free(Blockcsr_Col_tmp);
    free(tile_csr_ptr);
}
