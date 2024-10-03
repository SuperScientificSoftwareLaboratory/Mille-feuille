#include "common.h"

typedef struct csrval
{
    double csrval_high;
    float csrval_low;
}mixval;

typedef struct Data{
    double *v1;
    float *v2;
}va;

typedef struct
{
    int tilem;
    int tilen;
    int tilenum;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_nnz;
    char *Format;
    int *blknnz;
    unsigned char *blknnznnz;
    int *dnsrowptr;
    int *dnscolptr;
    char *tilewidth;
    int *csr_offset;
    int *csrptr_offset;
    int *coo_offset;
    int *ell_offset;
    int *hyb_offset;
    int *hyb_coocount;
    int *dns_offset;
    int *dnsrow_offset;
    int *dnscol_offset;
    int *new_coocount;
    MAT_VAL_TYPE *Blockcsr_Val;
    MAT_VAL_LOW_TYPE *Blockcsr_Val_Low;
    unsigned char *Tile_csr_Col;
    mixval *Blockcsr_Val_mix;
    unsigned char *Blockcsr_Ptr;
    unsigned char *csr_compressedIdx;
    int csrsize;
    int csrptrlen;
    MAT_VAL_TYPE *Blockcoo_Val;
    MAT_VAL_LOW_TYPE *Blockcoo_Val_Low;
    unsigned char *coo_compressed_Idx;
    int coosize;
    MAT_VAL_TYPE *Blockell_Val;
    MAT_VAL_LOW_TYPE *Blockell_Val_Low;
    unsigned char *ell_compressedIdx;
    int ellsize;
    MAT_VAL_TYPE *Blockhyb_Val;
    MAT_VAL_LOW_TYPE *Blockhyb_Val_Low;
    unsigned char *hybIdx;
    int hybsize;
    int hybellsize;
    int hybcoosize;
    MAT_VAL_TYPE *Blockdense_Val;
    MAT_VAL_LOW_TYPE *Blockdense_Val_Low;
    int dnssize;
    MAT_VAL_TYPE *Blockdenserow_Val;
    MAT_VAL_LOW_TYPE *Blockdenserow_Val_Low;
    char *denserowid;
    int dnsrowsize;
    MAT_VAL_TYPE *Blockdensecol_Val;
    MAT_VAL_LOW_TYPE *Blockdensecol_Val_Low;
    char *densecolid;
    int dnscolsize;
    int coototal;
    MAT_VAL_TYPE *deferredcoo_val;
    MAT_VAL_LOW_TYPE *deferredcoo_val_Low;
    int *deferredcoo_colidx;
    MAT_PTR_TYPE *deferredcoo_ptr;
    int rowblkblock;
    unsigned int *flag_bal_tile_rowidx;
    int *tile_bal_rowidx_colstart;
    int *tile_bal_rowidx_colstop;
    int *flag_tilerow_start;
    int *flag_tilerow_stop;
    int *tile_bal_rowidx_colstart_v2;
    int *tile_bal_rowidx_colstop_v2;
    int *map;

} Tile_matrix;

void Tile_destroy(Tile_matrix *matrix)
{

    free(matrix->tile_ptr);
    free(matrix->tile_columnidx);
    free(matrix->tile_nnz);
    free(matrix->blknnz);
    free(matrix->blknnznnz);
    free(matrix->Blockcsr_Val);
    free(matrix->csr_compressedIdx);
    free(matrix->Blockcsr_Ptr);
    free(matrix->Blockcoo_Val);
    free(matrix->Blockell_Val);
    free(matrix->Blockhyb_Val);
    free(matrix->hybIdx);
    free(matrix->Blockdense_Val);
    free(matrix->Blockdenserow_Val);
    free(matrix->denserowid);
    free(matrix->dnsrowptr);
    free(matrix->Blockdensecol_Val);
    free(matrix->densecolid);
    free(matrix->dnscolptr);
    free(matrix->tilewidth);
    free(matrix->csrptr_offset);
    free(matrix->coo_offset);
    free(matrix->ell_offset);
    free(matrix->hyb_offset);
    free(matrix->hyb_coocount);
    free(matrix->dns_offset);
    free(matrix->dnsrow_offset);
    free(matrix->dnscol_offset);
    free(matrix->new_coocount);

    free(matrix->deferredcoo_val);
    free(matrix->deferredcoo_colidx);
    free(matrix->deferredcoo_ptr);
}
