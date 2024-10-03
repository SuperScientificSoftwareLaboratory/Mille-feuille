#include "biio.h"

int main(int *argc, char *argv[])
{
    int row;
    int col;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *val;
    int isSymmeticeR;
    // char *filename = "1138_bus.mtx";
    char *filename = argv[1];
    read_Dmatrix_32(&row, &col, &nnz, &row_ptr, &col_idx, &val, &isSymmeticeR, filename);
    printf("%d,%d,%d\n", row, col, nnz);
    return 0;
}