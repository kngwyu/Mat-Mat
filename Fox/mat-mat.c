// Foxのアルゴリズムによる行列積
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define  N        32768
#define  NPROCS   256
#define  PROC_SQRT 16
#define  BLOCK_LEN 2048

#define  DEBUG  0
#define  PRINT  0
#define  EPS    1.0e-18
#define  MIN(a, b) ((a) > (b) ? (b) : (a))
#define  MAX(a, b) ((a) < (b) ? (b) : (a))

int myid, numprocs;
void MyMatMat(double c[BLOCK_LEN][BLOCK_LEN], double a[BLOCK_LEN][BLOCK_LEN], double b[BLOCK_LEN][BLOCK_LEN]);
int main(int argc, char* argv[]) {
    double  t0, t1, t2, t_w;
    double  dc_inv, d_mflops;
    int     ierr;
    int     i, j;      
    int     iflag, iflag_t;
    // 一応ミスしないように確認
    if (N % NPROCS != 0) {
        puts("N % NPROCS != 0");
        exit(0);
    }
    if (PROC_SQRT * PROC_SQRT != NPROCS) {
        puts("PROC_SQRT * PROC_SQRT != NPROCS");
        exit(0);
    }
    if (BLOCK_LEN * PROC_SQRT != N) {
        puts("BLOCK_LEN * PROC_SQRT != N");
        exit(0);
    }
    
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    static double a[BLOCK_LEN][BLOCK_LEN];
    static double b[BLOCK_LEN][BLOCK_LEN];
    static double c[BLOCK_LEN][BLOCK_LEN];
    int my_i = myid / PROC_SQRT, my_j = myid % PROC_SQRT;
    /* matrix generation --------------------------*/
    if (DEBUG == 1) {
        for (i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < BLOCK_LEN; ++j) {
                a[i][j] = 1.0;
                b[i][j] = 1.0;
                c[i][j] = 0.0;
            }
        }
    } else {
        srand(myid);
        dc_inv = 1.0/(double)RAND_MAX;
        for (i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < BLOCK_LEN; ++j) {
                double num = (my_i * BLOCK_LEN + i) + my_j * BLOCK_LEN + j;
                a[i][j] = b[i][j] = num;
                /* a[i][j] = rand() * dc_inv; */
                /* b[i][j] = rand() * dc_inv; */
                c[i][j] = 0.0;
            }
        }
    }
    /* end of matrix generation --------------------------*/
    /* Start of routine ----------------------------*/
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MyMatMat(c, a, b);
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t0 =  t2 - t1; 
    ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    /* End of routine --------------------------- */
    if (PRINT == 0 && myid == 0) {
        printf("N  = %d \n",N);
        printf("Mat-Mat time  = %lf [sec.] \n",t_w);
    }

    if (DEBUG == 1) {
        /* Verification routine ----------------- */
        iflag = 0;
        for(i = 0; i < BLOCK_LEN; ++i) {
            for (j = 0; j < BLOCK_LEN; ++j) {
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
        for (i = 0; i < BLOCK_LEN; ++i)
            for (j = 0; j < BLOCK_LEN; ++j)
                printf("%d %d %lf\n", i + my_i * BLOCK_LEN, j + my_j * BLOCK_LEN, c[i][j]);
    }
END:
    ierr = MPI_Finalize();
    exit(0);
}



void MyMatMat(double c[BLOCK_LEN][BLOCK_LEN], double a[BLOCK_LEN][BLOCK_LEN], double b[BLOCK_LEN][BLOCK_LEN]) {
    int i, j, k, ope;
    int ierr;
    MPI_Status istatus;
    // 自分が最初に持つ小行列の番号
    int my_i = myid / PROC_SQRT, my_j = myid % PROC_SQRT;
    // A は例えばA00 -> A01 -> A02のような順で同じ行のものは全部同じものを持つ
    MPI_Comm row_group;
    ierr = MPI_Comm_split(MPI_COMM_WORLD, my_i, my_j, &row_group);
    // Bは上シフトするだけ
    // 上シフトするPE(横の番号は同じ)
    int up_pe = ((my_i + PROC_SQRT - 1) % PROC_SQRT) * PROC_SQRT + my_j;
    int down_pe = ((my_i + 1) % PROC_SQRT) * PROC_SQRT + my_j;
    //    printf("my_id: %d left: %d righr: %d up: %d down: %d\n", myid, left_pe, right_pe, up_pe, down_pe);
    static double buf_a[BLOCK_LEN][BLOCK_LEN];
    for (ope = 0; ope < PROC_SQRT; ++ope) {
        // Aを放送する (A00, A11, ...) -> (A01, A12, ...)の順
        int root_j = (my_i + ope) % PROC_SQRT;
        if (my_j == root_j) {
            ierr = MPI_Bcast(a, BLOCK_LEN * BLOCK_LEN, MPI_DOUBLE, root_j, row_group);
            for (i = 0; i < BLOCK_LEN; ++i)
                for (j = 0; j < BLOCK_LEN; ++j) 
                    for (k = 0; k < BLOCK_LEN; ++k)
                        c[i][j] += a[i][k] * b[k][j];
        } else {
            ierr = MPI_Bcast(buf_a, BLOCK_LEN * BLOCK_LEN, MPI_DOUBLE, root_j, row_group);
            for (i = 0; i < BLOCK_LEN; ++i)
                for (j = 0; j < BLOCK_LEN; ++j) 
                    for (k = 0; k < BLOCK_LEN; ++k)
                        c[i][j] += buf_a[i][k] * b[k][j];
        }
        if (ope == PROC_SQRT - 1) break;
        // Bを上シフト
        ierr = MPI_Sendrecv_replace(b, BLOCK_LEN * BLOCK_LEN, MPI_DOUBLE, up_pe, ope, down_pe, ope, MPI_COMM_WORLD, &istatus);
    }
}
