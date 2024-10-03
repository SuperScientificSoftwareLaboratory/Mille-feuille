#include "biio.h"

int ana_row_ptr(int *row_ptr, int row)
{
    int ret = 0;
    for (size_t i = 0; i < row; i++)
    {
        row_ptr[i];
    }
    return ret;
}

int ana_col_idx(int *row_ptr, int row, int *col_idx, int nnz)
{
    int ret = 0;
    for (size_t j = 0; j < row; j++)
    {
        for (size_t i = row_ptr[j]; i < row_ptr[j + 1]; i++)
        {
            col_idx[i];
        }
    }
    return ret;
}

int ana_val(int *row_ptr, int row, int *val, int nnz)
{
    int ret = 0;
    for (size_t i = 0; i < nnz; i++)
    {
        val[i];
    }
    return ret;
}

int main(int argc, char *argv[])
{
    int row;
    int col;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *val;
    int isSymmeticeR;
    char *filename = atoi(argv[1]);

    read_Dmatrix(&row, &col, &nnz, &row_ptr, &col_idx, &val, &isSymmeticeR, filename);

    return 0;
}