#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define  N        576
#define  NPROCS   288

#define  DEBUG  1
#define  EPS    1.0e-18

void MatMat0(double* c, double* a, double* b, int n);
void MatMat1(double* c, double* a, double* b, int n);
int main(int argc, char* argv[]) {

     double  t0, t1, t2, t_w;
     double  dc_inv, d_mflops;
     int     ierr;
     int     i, j;      
     int     iflag, iflag_t;
     ierr = MPI_Init(&argc, &argv);
     ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
     ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
     /* matrix generation --------------------------*/
     if (DEBUG == 1) {
       for(j=0; j<N/NPROCS; j++) {
         for(i=0; i<N; i++) {
           A[j][i] = 1.0;
           C[j][i] = 0.0;
         }
       }
       for(j=0; j<N; j++) {
         for(i=0; i<N/NPROCS; i++) {
           B[j][i] = 1.0;
         }
       }
     } else {
       srand(myid);
       dc_inv = 1.0/(double)RAND_MAX;
      for(j=0; j<N/NPROCS; j++) {
         for(i=0; i<N; i++) {
           A[j][i] = rand()*dc_inv;
           C[j][i] = 0.0;
         }
       }
      for(j=0; j<N; j++) {
         for(i=0; i<N/NPROCS; i++) {
           B[j][i] = rand()*dc_inv;
         }
       }

     } /* end of matrix generation --------------------------*/

     /* Start of mat-vec routine ----------------------------*/
     ierr = MPI_Barrier(MPI_COMM_WORLD);
     t1 = MPI_Wtime();

     MyMatMat(C, A, B, N);

     ierr = MPI_Barrier(MPI_COMM_WORLD);
     t2 = MPI_Wtime();
     t0 =  t2 - t1; 
     ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     /* End of mat-vec routine --------------------------- */

     if (myid == 0) {

       printf("N  = %d \n",N);
       printf("Mat-Mat time  = %lf [sec.] \n",t_w);

       d_mflops = 2.0*(double)N*(double)N*(double)N/t_w;
       d_mflops = d_mflops * 1.0e-6;
       printf(" %lf [MFLOPS] \n", d_mflops);
     }

     if (DEBUG == 1) {
       /* Verification routine ----------------- */
       iflag = 0;
       for(j=0; j<N/NPROCS; j++) { 
         for(i=0; i<N; i++) { 
           if (fabs(C[j][i] - (double)N) > EPS) {
             printf(" Error! in ( %d , %d )-th argument in PE %d \n",j, i, myid);
             printf("%lf\n", C[j][i]);
             iflag = 1;
             ierr = MPI_Finalize();
             exit(1);
           } 
         }
       }
       /* ------------------------------------- */

       MPI_Reduce(&iflag, &iflag_t, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
       if (myid == 0) {
         if (iflag_t == 0) printf(" OK! \n");
       }

     }

     ierr = MPI_Finalize();

     exit(0);
}

void MatMat0(double* c, double* a, double* b, int n) {
    int i, j, k;
    for (i = 0; i < n; ++i)
        for(j = 0; j < n; ++j)
            for (k = 0; k < n; ++k)
                c[i * n + j] = a[i * n + k] + b[k * n + j];
}
void MatMat1(double C[N/NPROCS][N], 
              double A[N/NPROCS][N], double B[N][N/NPROCS], int n) 
{
     int  i, j, k;
     int  block_len;
     int  ierr;
     int  jstart; 
     int  isendPE, irecvPE;
     int process_i;
     MPI_Status istatus;
     /* Information of Send and recv PEs */
     isendPE = (myid + numprocs - 1) % numprocs;
     irecvPE = (myid + 1) % numprocs;
     block_len = n / numprocs;
     /* Local Matrix-Matrix Multiplication -------------------  */
     /* This shoule be modified to finish */
     for (process_i = 0; process_i < numprocs; ++process_i) {
       jstart = block_len * ((process_i + myid) % numprocs);
       for(i = 0; i < block_len; ++i) { // iは毎回同じ
         for(j = 0; j < block_len; ++j) { // jは右にずれる
           C[i][j + jstart] = 0;
           for (k = 0; k < n; ++k) {
             C[i][j + jstart] += A[i][k] * B[k][j]; 
           }
         }
       }
       if (process_i == numprocs) break;
       if ((myid & 1) == 0) { // 先に送信する
         ierr = MPI_Send(B, n * block_len, MPI_DOUBLE, isendPE, myid + process_i, MPI_COMM_WORLD);
         ierr = MPI_Recv(B_T, n * block_len, MPI_DOUBLE, irecvPE, irecvPE + process_i, MPI_COMM_WORLD, &istatus);
       } else {
         ierr = MPI_Recv(B_T, n * block_len, MPI_DOUBLE, irecvPE, irecvPE + process_i, MPI_COMM_WORLD, &istatus);
         ierr = MPI_Send(B, n * block_len, MPI_DOUBLE, isendPE, myid + process_i, MPI_COMM_WORLD);
       }
       for (i = 0; i < n; ++i)
         for (j = 0; j < block_len; ++j)
           B[i][j] = B_T[i][j];
     }
}


