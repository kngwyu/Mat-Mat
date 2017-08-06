// 単純な並列化をした行列積
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define  N        512
#define  NPROCS   256
#define  BLOCK_LEN (N / NPROCS)

#define  DEBUG  0
#define  PRINT  1
#define  EPS    1.0e-18
#define  MIN(a, b) ((a) > (b) ? (b) : (a))
#define  MAX(a, b) ((a) < (b) ? (b) : (a))

int myid, numprocs;
void MyMatMat(double c[BLOCK_LEN][N], double a[BLOCK_LEN][N], double b[N][BLOCK_LEN]);
int main(int argc, char* argv[]) {
    double  t0, t1, t2, t_w;
    double  dc_inv, d_mflops;
    int     ierr;
    int     i, j;      
    int     iflag, iflag_t;
    if (N % NPROCS != 0) {
        puts("N % NPROCS != 0");
        exit(0);
    }
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    static double a[BLOCK_LEN][N];
    static double b[N][BLOCK_LEN];
    static double c[BLOCK_LEN][N];
    /* matrix generation --------------------------*/
    if (DEBUG == 1) {
        for (i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < N; ++j) {
                a[i][j] = 1.0;
                c[i][j] = 0.0;
            }
        }
        for (i = 0; i < N; ++i) {
            for (j = 0; j < BLOCK_LEN; ++j) {
                b[i][j] = 1.0;
            }
        }
    } else {
        srand(myid);
        dc_inv = 1.0/(double)RAND_MAX;
        for (i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < N; ++j) {
                a[i][j] = (i + myid * BLOCK_LEN) * N + j;
                /* a[i][j] = rand() * dc_inv; */
                c[i][j] = 0.0;
            }
        }
        for (i = 0; i < N; ++i) {
            for (j = 0; j < BLOCK_LEN; ++j) {
                b[i][j] = i * N + j + myid * BLOCK_LEN;
                /* b[i][j] = rand() * dc_inv; */
            }
        }
    } /* end of matrix generation --------------------------*/

    /* Start of routine ----------------------------*/
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MyMatMat(c, a, b);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t0 =  t2 - t1; 
    ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    /* End of routine --------------------------- */
    if (myid == 0) {
        printf("N  = %d \n",N);
        printf("Mat-Mat time  = %lf [sec.] \n",t_w);
    }
    if (DEBUG == 1) {
        /* Verification routine ----------------- */
        iflag = 0;
        for(i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < N; ++j) {
                if (fabs(c[i][j] - (double)N) > EPS) {
                    printf(" Error! in ( %d , %d )-th argument in PE %d \n", i, j, myid);
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
    if (PRINT == 1) {
        for (i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < N; ++j) {
                printf("%d %d %lf\n", i + myid * BLOCK_LEN, j, c[i][j]);
            }
        }
    }
END:
    ierr = MPI_Finalize();
    exit(0);
}

void MyMatMat(double c[BLOCK_LEN][N], double a[BLOCK_LEN][N], double b[N][BLOCK_LEN]) {
    int  i, j, k;
    int  ierr;
    int  isendPE, irecvPE;
    int  process_i;
    MPI_Status istatus;
    /* Information of Send and recv PEs */
    isendPE = (myid + numprocs - 1) % numprocs;
    irecvPE = (myid + 1) % numprocs;
    int jstart;
    static double buf[N][BLOCK_LEN];
    for (process_i = 0; process_i < numprocs; ++process_i) {
        jstart = BLOCK_LEN * ((process_i + myid) % numprocs);
        for(i = 0; i < BLOCK_LEN; ++i) { // iは毎回同じ
            for(j = 0; j < BLOCK_LEN; ++j) { // jは右にずれる
                c[i][j + jstart] = 0;
                for (k = 0; k < N; ++k) {
                    c[i][j + jstart] += a[i][k] * b[k][j]; 
                }
            }
        }
        if (process_i == numprocs) break;
        if ((myid & 1) == 0) { // 先に送信する
            ierr = MPI_Send(b, N * BLOCK_LEN, MPI_DOUBLE, isendPE, myid + process_i, MPI_COMM_WORLD);
            ierr = MPI_Recv(buf, N * BLOCK_LEN, MPI_DOUBLE, irecvPE, irecvPE + process_i, MPI_COMM_WORLD, &istatus);
        } else {
            ierr = MPI_Recv(buf, N * BLOCK_LEN, MPI_DOUBLE, irecvPE, irecvPE + process_i, MPI_COMM_WORLD, &istatus);
            ierr = MPI_Send(b, N * BLOCK_LEN, MPI_DOUBLE, isendPE, myid + process_i, MPI_COMM_WORLD);
        }
        for (i = 0; i < N; ++i)
            for (j = 0; j < BLOCK_LEN; ++j)
                b[i][j] = buf[i][j];
    }
}

