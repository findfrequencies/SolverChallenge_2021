/*
A simple serial CG iterative method to solve a linear system
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang,Haocheng Lian
*/
// Multiply a csr matrix with a vector x, and get the resulting vector y
#include <petscksp.h>

void spmv(int n, int *row_ptr, int *col_idx, double *val, double *x, double *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            y[i] += val[j] * x[col_idx[j]];
    }
}

// Calculate the 2-norm of a vector
double vec2norm(double *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

// Compute dot product of two vectors, and return the result
double dotproduct(double *x1, double *x2, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x1[i] * x2[i];
    return sum;
}




//



/*T
   Concepts: KSP^solving a system of linear equations
   Processors: 1
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers

  Note:  The corresponding parallel example is ex23.c
*/

static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";
int my_solver(int argc,char **args,int ns, int *row_ptr, int *col_idx, double *val,double *x_o, double *b_o, int *iter_o, double tolerance)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm,norm_b;  /* norm of solution error */
  PetscErrorCode ierr;
  Vec            residual;     /* residual  of  mat    */
  PetscInt       i,its;
  PetscInt       * row_ptr_o,* col_idx_o;
  PetscScalar     * val_o;
  PetscInt        m,n;
  PetscMPIInt     size;
  PetscReal rtol, abstol, dtol;
  PetscInt maxits;
  PetscScalar *array;  /* store vec to array */
   

  //debug
  n = ns;
  m = n;
  val_o = val;
  row_ptr_o = row_ptr;
  col_idx_o = col_idx;
  PetscScalar     *b_os;
  b_os=b_o;
  
  PetscInt n_o;
  
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n_o,NULL);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_WORLD," n is %d \n ",n);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&residual);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  
/*
     edit by ff in 21/6 

     Assemble matrix
     
  */
   /*
   modify by ff 21/6
create mat with csr ;
   
   */

   ierr = PetscPrintf(PETSC_COMM_WORLD,"start to create mat \n");CHKERRQ(ierr);

 
   ierr=MatSeqAIJSetPreallocationCSR(A,row_ptr_o,col_idx_o,val_o);CHKERRQ(ierr);
   /*

   when we finsh insert a value to mat,we must call the two fuc
   modify by ff 21/6 
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   */
  
/*
  Assemble rhs vec

*/
   for (i=0;i<n;i++){

  ierr=VecSetValues(b,1,&i,&b_os[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr=VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr=VecAssemblyEnd(b);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-1,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /*
     View solver info; we could instead use the option -ksp_view to
     print this info to the screen at the conclusion of KSPSolve().
  */
//  ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatMult(A,x,u);CHKERRQ(ierr);
  ierr = VecWAXPY(residual,-1.0,u,b);CHKERRQ(ierr);

    /*
    modify by ff 22/6
    PetscErrorCode  VecWAXPY(Vec w,PetscScalar alpha,Vec x,Vec y)
    Computes w = alpha x + y.
  */
   
   /*
   PetscErrorCode  VecAXPY(Vec y,PetscScalar alpha,Vec x)
    Computes y = alpha x + y.
   
   */

  /*
  modify by ff 22/6
  Gets the relative, absolute, divergence, 
  and maximum iteration tolerances used by the default KSP convergence tests.
Synopsis
#include "petscksp.h" 
PetscErrorCode  KSPGetTolerances(KSP ksp,PetscReal *rtol,PetscReal *abstol,PetscReal *dtol,PetscInt *maxits)
  */


  //ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);

  ierr = VecNorm(residual,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&norm_b);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  *iter_o=its;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Relative Residual Norm %g, Residual Norm %g, Iterations %D\n",(double)norm/norm_b,(double)norm,its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  //ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  for(i=0;i<n;i++)
  {x_o[i]=array[i];
   /* dubug to figure out the function of vga mark by ff 23/6 2021
   if(i<10)
   printf("%.5lf",x_o[i]);
   */
  }
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr); //when call vga then must call vesa mark by ff 23/6 2021
  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2_aijcusparse
      requires: cuda
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda

   test:
      suffix: 3
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 3_aijcusparse
      requires: cuda
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda

   test:
      suffix: aijcusparse
      requires: cuda
      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -mat_type aijcusparse -vec_type cuda
      output_file: output/ex1_1_aijcusparse.out

TEST*/




