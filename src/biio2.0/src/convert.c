#include "biio.h"

int main(int *argc, char *argv[])
{
    int row;
    int col;
    int64_t nnz;
    int64_t *row_ptr;
    int *col_idx;
    double *val;
    int isSymmeticeR;
    // char *filename = "1138_bus.mtx";
    char *filename = argv[1];
    read_Dmatrix_convert(&row, &col, &nnz, &row_ptr, &col_idx, &val, &isSymmeticeR, filename);
    return 0;
}