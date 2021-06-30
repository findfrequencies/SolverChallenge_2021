/*
This is the reference code of SolverChallenge to optimize by the participants
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang, Haocheng Lian
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "mmio_highlevel.h"
#include "my_solver.h"

double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b);
void store_x(int n, double *x, char *filename);
void load_b(int n, double *b, char *filename);

int main(int argc, char **argv)
{
    int ierr;
    int m, n, nnzA, isSymmetricA;
    int *row_ptr; // the csr row pointer array of matrix A
    int *col_idx; // the csr column index array of matrix A
    double *val;  // the csr value array of matrix A

    char *filename_matrix = argv[1];  // the filename of matrix A
    char *filename_b = argv[2];       // the filename of right-hand side vector b
    char *filename_x = argv[3];       // the filename of solution vector x
    char *matrix_tolerance = argv[4]; // the tolerance of input matrix
    int iter = 0;                     // the number of iterations
    double residule = 0.0;
    double tolerance = atof(matrix_tolerance);
 


    //load matrix
    mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
    if (m != n)
    {
        printf("Invalid matrix size.\n");
        return 0;
    }

    double *x = (double *)malloc(sizeof(double) * n);
    double *b = (double *)malloc(sizeof(double) * m);
    // load right-hand side vector b
    load_b(n, b, filename_b);
    double n_b=vec2norm(b,n);
    // solve x and record wall-time
    struct timeval t_start, t_stop;
    printf("n_b = %lf ",n_b);
    gettimeofday(&t_start, NULL);
 
   
     ierr= my_solver( argc,argv, n,row_ptr,col_idx, val,x, b, &iter, tolerance);
    gettimeofday(&t_stop, NULL);

    
    double total_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec)/1000.0 ;
    
     store_x(n, x, filename_x);
    //store x to a file
    residule = check_correctness(n, row_ptr, col_idx, val, x, b);
    //check the correctness
    //print the #iteration, residual and total_time
  
    printf("The number of iteration = %d, residule = %e, total_time = %.5lf sec\n", iter, residule, total_time / 1000.0);

   
}

//validate the x by b-A*x
double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b)
{
    double *b_new = (double *)malloc(sizeof(double) * n);
    double *check_b = (double *)malloc(sizeof(double) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    for (int i = 0; i < n; i++)
        check_b[i] = b_new[i] - b[i];
    return vec2norm(check_b, n) / vec2norm(b, n);
}

//store x to a file
void store_x(int n, double *x, char *filename)

{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%d\n", n);
    for (int i = 0; i < n; i++)
        fprintf(p, "%lf\n", x[i]);
    fclose(p);
}

//load right-hand side vector b
void load_b(int n, double *b, char *filename)
{
    FILE *p = fopen(filename, "r");
    int n_right;
    int r = fscanf(p, "%d", &n_right);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    for (int i = 0; i < n_right; i++)
        r = fscanf(p, "%lf", &b[i]);
    fclose(p);
}
