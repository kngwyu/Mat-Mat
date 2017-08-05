#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define  N        576
#define  NPROCS   288

#define  DEBUG  1
#define  EPS    1.0e-18
#define  MIN(a, b) ((a) > (b) ? (b) : (a))
#define  MAX(a, b) ((a) < (b) ? (b) : (a))
#define  FILL(a, n, x) do{int i;for(i=0;i<n;++i){a[i]=x;}}while(0);
#define  COPY(a, b, n) do{int i;for(i=0;i<n;++i){b[i]=a[i];}}while(0);

int myid, numprocs;
void MyMatMat(double* c, double* a, double* b, int n);
int main(int argc, char* argv[]) {
    double  t0, t1, t2, t_w;
    double  dc_inv, d_mflops;
    int     ierr;
    int     i, j;      
    int     iflag, iflag_t;
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int istart = (N / NPROCS) * myid;
    int iend = MIN(istart + N / NPROCS, N);
    int block_len = iend - istart;
    double* a = (double*)malloc(sizeof(double) * N * block_len);
    double* b = (double*)malloc(sizeof(double) * N * block_len);
    double* c = (double*)malloc(sizeof(double) * N * block_len);
    /* matrix generation --------------------------*/
    if (DEBUG == 1) {
        FILL(a, N * block_len, 1.0);
        FILL(b, N * block_len, 1.0);
        FILL(c, N * block_len, 0.0);
    } else {
        srand(myid);
        dc_inv = 1.0/(double)RAND_MAX;
        for (i = 0; i < N * block_len; ++i) {
            a[i] = rand() * dc_inv;
            b[i] = rand() * dc_inv;
            c[i] = 0.0;
        }
    } /* end of matrix generation --------------------------*/

    /* Start of mat-vec routine ----------------------------*/
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MyMatMat(c, a, b, N);
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
        for(i = 0; i < N * block_len; ++i) {
            for (j = 0; j < N; ++j) {
                if (fabs(c[i * block_len + j] - (double)N) > EPS) {
                    printf(" Error! in ( %d , %d )-th argument in PE %d \n",j, i, myid);
                    iflag = 1;
                    ierr = MPI_Finalize();
                    goto END;
                } 
            }
        }
        /* ------------------------------------- */
        MPI_Reduce(&iflag, &iflag_t, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myid == 0) {
            if (iflag_t == 0) printf(" OK! \n");
        }

    }
END:
    ierr = MPI_Finalize();
    free(a);
    free(b);
    free(c);
    exit(0);
}

void MyMatMat(double* c, double* a, double* b, int n) {
    int  i, j, k;
    int  ierr;
    int  isendPE, irecvPE;
    int  process_i;
    MPI_Status istatus;
    /* Information of Send and recv PEs */
    isendPE = (myid + numprocs - 1) % numprocs;
    irecvPE = (myid + 1) % numprocs;
    int jstart, block_len;
    {
        int istart = (n / numprocs) * myid;
        int iend = MIN(istart + n / numprocs, n);
        block_len = iend - istart;
    }
    double* buf = (double*)malloc(sizeof(b));
    for (process_i = 0; process_i < numprocs; ++process_i) {
        jstart = block_len * ((process_i + myid) % numprocs);
        for(i = 0; i < block_len; ++i) { // iは毎回同じ
            for(j = 0; j < block_len; ++j) { // jは右にずれる
	      //                c[i * block_len + j + jstart] = 0;
                for (k = 0; k < n; ++k) {
                    c[i * block_len + j] += a[i * block_len + k] * b[k * n + j]; 
                }
            }
        }
        if (process_i == numprocs) break;
        if ((myid & 1) == 0) { // 先に送信する
            ierr = MPI_Send(b, n * block_len, MPI_DOUBLE, isendPE, myid + process_i, MPI_COMM_WORLD);
            ierr = MPI_Recv(buf, n * block_len, MPI_DOUBLE, irecvPE, irecvPE + process_i, MPI_COMM_WORLD, &istatus);
        } else {
            ierr = MPI_Recv(buf, n * block_len, MPI_DOUBLE, irecvPE, irecvPE + process_i, MPI_COMM_WORLD, &istatus);
            ierr = MPI_Send(b, n * block_len, MPI_DOUBLE, isendPE, myid + process_i, MPI_COMM_WORLD);
        }
        COPY(buf, b, n * block_len);
    }
    free(buf);
}


