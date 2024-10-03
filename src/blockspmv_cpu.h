#include "common.h"
void blockspmv_cpu(Tile_matrix *matrix,
                  int *ptroffset1,
                  int *ptroffset2,
                  int *rowblkblock,
                  unsigned int **blkcoostylerowidx,
                  int **blkcoostylerowidx_colstart,
                  int **blkcoostylerowidx_colstop,
                  int rowA, int colA, MAT_PTR_TYPE nnzA,
                  MAT_PTR_TYPE *csrRowPtrA,
                  int *csrColIdxA,
                  MAT_VAL_TYPE *csrValA,
                  MAT_VAL_TYPE *x,
                  MAT_VAL_TYPE *y,
                  MAT_VAL_TYPE *y_golden

)
{
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    unsigned char *blknnznnz = matrix->blknnznnz;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    unsigned char *csr_compressedIdx = matrix->csr_compressedIdx;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;

    int csroffset = 0;
    int csrcount = 0;

    int rowblkblock_tmp = 0;
    for (int blki = 0; blki < tilem; blki++)
    {
        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH)
            rowblkblock_tmp++;
        else
        {
            rowblkblock_tmp += ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
        }
    }
    *rowblkblock = rowblkblock_tmp;

    *blkcoostylerowidx = (unsigned int *)malloc(sizeof(unsigned int) * *rowblkblock);
    unsigned int *blkcoostylerowidx_tmp = *blkcoostylerowidx;
    memset(blkcoostylerowidx_tmp, 0, sizeof(unsigned int) * *rowblkblock);

    *blkcoostylerowidx_colstart = (int *)malloc(sizeof(int) * *rowblkblock);
    int *blkcoostylerowidx_colstart_tmp = *blkcoostylerowidx_colstart;
    memset(blkcoostylerowidx_colstart_tmp, 0, sizeof(int) * *rowblkblock);
    *blkcoostylerowidx_colstop = (int *)malloc(sizeof(int) * *rowblkblock);
    int *blkcoostylerowidx_colstop_tmp = *blkcoostylerowidx_colstop;
    memset(blkcoostylerowidx_colstop_tmp, 0, sizeof(int) * *rowblkblock);

    int rowblkblockcnt = 0;
    for (int blki = 0; blki < tilem; blki++)
    {
        int balancenumblk = tile_ptr[blki + 1] - tile_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH)
        {
            blkcoostylerowidx_tmp[rowblkblockcnt] = blki;
            rowblkblockcnt++;
        }
        else
        {
            int numblklocal = ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
            int lenblklocal = ceil((double)balancenumblk / (double)numblklocal);
            for (int iii = 0; iii < numblklocal; iii++)
            {
                blkcoostylerowidx_tmp[rowblkblockcnt] = blki | 0x80000000; // can generate -0
                blkcoostylerowidx_colstart_tmp[rowblkblockcnt] = tile_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = tile_ptr[blki] + balancenumblk;
                else
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = tile_ptr[blki] + (iii + 1) * lenblklocal;

                rowblkblockcnt++;
            }
        }
    }

    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int ri = 0; ri < BLOCK_SIZE; ri++)
        {
            y[blki * BLOCK_SIZE + ri] = 0;
        }
        for (int blkj = tile_ptr[blki]; blkj < tile_ptr[blki + 1]; blkj++)
        {
            int collength = tile_columnidx[blkj] == tilen - 1 ? colA - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int x_offset = tile_columnidx[blkj] * BLOCK_SIZE;
            ptroffset1[blkj] = csroffset;
            ptroffset2[blkj] = csrcount;
            for (int ri = 0; ri < rowlength; ri++)
            {
                MAT_VAL_TYPE sum = 0;
                int stop = ri == rowlength - 1 ? (blknnz[blkj + 1] - blknnz[blkj]) : Blockcsr_Ptr[ri + 1 + csrcount];
                for (int rj = Blockcsr_Ptr[csrcount + ri]; rj < stop; rj++)
                {
                    int csrcol = (csroffset + rj) % 2 == 0 ? (csr_compressedIdx[(csroffset + rj) / 2] & num_f) >> 4 : csr_compressedIdx[(csroffset + rj) / 2] & num_b;
                    sum += x[x_offset + csrcol] * Blockcsr_Val[csroffset + rj];
                }
                y[blki * BLOCK_SIZE + ri] += sum;
            }
            csroffset += blknnz[blkj + 1] - blknnz[blkj];
            csrcount += rowlength;
        }
    }
}