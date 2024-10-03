#if !defined(BIIO_H_)
#define BIIO_H_

// #define PRINT_INFO

#ifdef PRINT_INFO
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "", ##args)
#else
#define INFO_LOG(fmt, args...)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <sys/mman.h>
#include <fcntl.h>

#include "mmio_highlevel.h"

int binary_read_csr(int *row, int *col, int64_t *nnz, int64_t **row_ptr, int **col_idx, double **val, char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r+");
    if (fp == NULL)
    {
        INFO_LOG("binary file %s don't exit\n", filename);
        return -1;
    }
    if (1 != fread(row, sizeof(int), 1, fp))
    {
        INFO_LOG("read row error\n");
        return -1;
    }

    if (1 != fread(col, sizeof(int), 1, fp))
    {
        INFO_LOG("read col error\n");
        return -1;
    }
    if (1 != fread(nnz, sizeof(int64_t), 1, fp))
    {
        INFO_LOG("read nnz error\n");
        return -1;
    }

    *row_ptr = (int64_t *)malloc(sizeof(int64_t) * (*row + 1));
    if ((*row + 1) != fread(*row_ptr, sizeof(int64_t), *row + 1, fp))
    {
        INFO_LOG("read row_ptr error\n");
        return -1;
    }

    *col_idx = (int *)malloc(sizeof(int) * (*nnz));
    if ((*nnz) != fread(*col_idx, sizeof(int), (*nnz), fp))
    {
        INFO_LOG("read col_idx error\n");
        return -1;
    }

    *val = (double *)malloc(sizeof(double) * (*nnz));
    if ((*nnz) != fread(*val, sizeof(double), (*nnz), fp))
    {
        INFO_LOG("read val error\n");
        return -1;
    }

    fclose(fp);

    return 0;
}

void binary_write_csr(int row, int col, int64_t nnz, int64_t *row_ptr, int *col_idx, double *val, char *filename)
{

    INFO_LOG("write binary file %s\n", filename);
    FILE *fp = fopen(filename, "w+");
    if (fp != NULL)
    {
        INFO_LOG("write row is %d col is %d nnz is %ld\n", row, col, nnz);
        fwrite(&row, sizeof(int), 1, fp);
        fwrite(&col, sizeof(int), 1, fp);
        fwrite(&nnz, sizeof(int64_t), 1, fp);
        fwrite(row_ptr, sizeof(int64_t), row + 1, fp);
        fwrite(col_idx, sizeof(int), nnz, fp);
        fwrite(val, sizeof(double), nnz, fp);
    }
    else
    {
        INFO_LOG("open error\n");
        return;
    }

    fclose(fp);
    return;
}

void read_Dmatrix_convert(int *row, int *col, int64_t *nnz, int64_t **row_ptr, int **col_idx, double **val, int *isSymmeticeR, char *filename)
{
    char binary_name[512];
    int file_length = strlen(filename);
    strcpy(binary_name, filename);
    binary_name[file_length - 1] = 'd';
    binary_name[file_length - 2] = 'b';
    binary_name[file_length - 3] = 'c';
    int flag = binary_read_csr(row, col, nnz, row_ptr, col_idx, val, binary_name);
    if (flag == -1)
    {
        INFO_LOG("read file name is %s\n", filename);
        mmio_info(row, col, nnz, isSymmeticeR, filename);
        INFO_LOG("row %d, col %d, nnz %ld\n", *row, *col, *nnz);
        *row_ptr = (int64_t *)malloc(sizeof(int64_t) * (*row + 1));
        *col_idx = (int *)malloc(sizeof(int) * (*nnz));
        *val = (double *)malloc(sizeof(double) * (*nnz));
        mmio_data(*row_ptr, *col_idx, *val, filename);
        INFO_LOG("begin write binary file\n");
        binary_write_csr(*row, *col, *nnz, *row_ptr, *col_idx, *val, binary_name);
        INFO_LOG("end write binary file\n");
    }
    else
    {
        INFO_LOG("read binary file %s\n", binary_name);
    }
    return;
}

void read_Dmatrix(int *row, int *col, int64_t *nnz, int64_t **row_ptr, int **col_idx, double **val, int *isSymmeticeR, char *filename)
{

    int name_len = strlen(filename);
    if (filename[name_len - 3] == 'c' && filename[name_len - 2] == 'b' && filename[name_len - 1] == 'd')
    {
        INFO_LOG("read binary file %s\n", filename);
        binary_read_csr(row, col, nnz, row_ptr, col_idx, val, filename);
    }
    else if (filename[name_len - 3] == 'm' && filename[name_len - 2] == 't' && filename[name_len - 1] == 'x')
    {
        INFO_LOG("read regular file %s\n", filename);
        mmio_info(row, col, nnz, isSymmeticeR, filename);
        *row_ptr = (int64_t *)malloc(sizeof(int64_t) * (*row + 1));
        *col_idx = (int *)malloc(sizeof(int) * (*nnz));
        *val = (double *)malloc(sizeof(double) * (*nnz));
        mmio_data(*row_ptr, *col_idx, *val, filename);
    }
    else
    {
        INFO_LOG("File type unsupport\n");
    }
    return;
}

void read_Dmatrix_32(int *row, int *col, int *nnz, int **row_ptr, int **col_idx, double **val, int *isSymmeticeR, char *filename)
{
    int64_t _nnz;
    int64_t *_row_ptr;

    int name_len = strlen(filename);
    if (filename[name_len - 3] == 'c' && filename[name_len - 2] == 'b' && filename[name_len - 1] == 'd')
    {
        printf("read binary file %s\n",filename);
        INFO_LOG("read binary file %s\n", filename);
        binary_read_csr(row, col, &_nnz, &_row_ptr, col_idx, val, filename);
    }
    else if (filename[name_len - 3] == 'm' && filename[name_len - 2] == 't' && filename[name_len - 1] == 'x')
    {
        INFO_LOG("read regular file %s\n", filename);
        mmio_info(row, col, &_nnz, isSymmeticeR, filename);
        _row_ptr = (int64_t *)malloc(sizeof(int64_t) * (*row + 1));
        *col_idx = (int *)malloc(sizeof(int) * _nnz);
        *val = (double *)malloc(sizeof(double) * _nnz);
        mmio_data(_row_ptr, *col_idx, *val, filename);
    }
    else
    {
        INFO_LOG("File type unsupport\n");
    }

    if (_nnz > INT_MAX)
    {
        INFO_LOG("64 bit nnz and row_ptr are needed to read %s\n", filename);
        return;
    }

    *nnz = _nnz;
    *row_ptr = (int *)malloc(sizeof(int) * (*row + 1));
    for (size_t i = 0; i < (*row + 1); i++)
    {
        (*row_ptr)[i] = _row_ptr[i];
    }
    free(_row_ptr);
    
    return;
}

#endif // BIIO_H_
