#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <time.h>
#include <complex.h>
#include "taylor.h"

#define JJ _NUMBER_OF_JET_VARS_
#define DD _MAX_DEGREE_OF_JET_VARS_
#define CC _JET_COEFFICIENTS_COUNT_TOTAL_

#define OD_zero_TOL 1e-20
#define IEV_TOL 1e-15

#define IEV_ind 1

#define verbose          0
#define verbose_b      ( 0 || verbose )
#define verbose_wIEVEC ( 0 || verbose )
#define verbose_IEVEC  ( 0 || verbose )

MY_FLOAT s=77.27, w=0.161, q=8.375e-6, fff=1.0;

// 1e+5
// w_2: 6.366646677037290e-01 -3.092762194173271e-01 7.097130580012648e-01 , (3 iterations)
// IEV_2: -1.002338192168862e+00 -3.354368720653866e-04 2.083194336549453e-03 
// Execution time: 4.594112 seconds


// 1e+2
// w_2: 6.366646714555763e-01 -3.092762126739395e-01 7.097130575721921e-01 , (3 iterations)
// IEV_2: -1.002338192183664e+00 -3.354362510189501e-04 2.083186632811708e-03 
// Execution time: 25.689656 seconds


int main(int argc, char *argv[]){

    int jj, cc, i, od;
    int OD;

    char OD_num_name[] = "ORBIT DECOMP/num_OD.txt", OD_EVAL_name[] = "ORBIT DECOMP/EVAL.bin", OD_IEVEC_name[] = "ORBIT DECOMP/IEVEC_2.bin", OD_R_name[] = "ORBIT DECOMP/R.bin", OD_Q_name[] = "ORBIT DECOMP/Q.bin", OD_b_name[]="ORBIT DECOMP/b.bin";
    FILE *OD_num_txt, *OD_EVAL_bin, *OD_IEVEC_bin, *OD_R_bin, *OD_Q_bin, *OD_b_bin;

    OD_IEVEC_bin = fopen(OD_IEVEC_name, "wb");

    OD_num_txt = fopen(OD_num_name, "r");
    if (OD_num_txt==NULL) exit(0);
    fscanf(OD_num_txt, "%d", &OD);

    clock_t CLOCK_begin = clock();

    OD_EVAL_bin = fopen(OD_EVAL_name, "rb");
    if (OD_EVAL_bin==NULL) exit(0);
    
    OD_R_bin = fopen(OD_R_name, "rb");
    if (OD_R_bin==NULL) exit(0);

    OD_Q_bin = fopen(OD_Q_name, "rb");
    if (OD_Q_bin==NULL) exit(0);

    OD_b_bin = fopen(OD_b_name, "rb");
    if (OD_b_bin==NULL) exit(0);

    double **OD_EVAL = malloc(OD*sizeof(double *));
    double **OD_b = malloc(OD*sizeof(double *));
    double **OD_R = malloc(OD*sizeof(double *));
    double **OD_Q = malloc(OD*sizeof(double *));
    for (od=0; od<OD; od++){
        OD_EVAL[od] = malloc(JJ*sizeof(double));
        OD_b[od] = malloc(JJ*sizeof(double));
        OD_R[od] = malloc(JJ*JJ*sizeof(double));
        OD_Q[od] = malloc(JJ*JJ*sizeof(double));

        fread(OD_EVAL[od], sizeof(double), JJ, OD_EVAL_bin);
        fread(OD_b[od], sizeof(double), JJ, OD_b_bin);
        fread(OD_R[od], sizeof(double), JJ*JJ, OD_R_bin);
        fread(OD_Q[od], sizeof(double), JJ*JJ, OD_Q_bin);
    }

    lapack_int LAPACK_info;

    if (verbose_b){
        for (od=0; od<OD; od++){
            printf("rhs_%d: ", od+1);
            for (jj=0; jj<JJ; jj++) printf("%.17le, ", OD_b[od][jj]);
            printf("\n");
        }
    }

    // double IEV_temp[JJ];
    // for (od=0; od<OD-1; od++) {
    //     for (jj=0; jj<JJ; jj++) IEV_temp[jj] = OD_b[od][jj];
    //     cblas_dgemv(CblasRowMajor, CblasTrans, JJ, JJ, -1.0, OD_Q[od+1], JJ, IEV_temp, 1, 0, OD_b[od], 1);
    //     LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_R[od], JJ, OD_b[od], 1);
    //     if (LAPACK_info != 0) { exit(0); }
    // }
    // for (jj=0; jj<JJ; jj++) IEV_temp[jj] = OD_b[OD-1][jj];
    // cblas_dgemv(CblasRowMajor, CblasTrans, JJ, JJ, -1.0, OD_Q[0], JJ, IEV_temp, 1, 0, OD_b[OD-1], 1);
    // LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_R[OD-1], JJ, OD_b[OD-1], 1);
    // if (LAPACK_info != 0) { exit(0); }

    // double **IEV = malloc(OD*sizeof(double *));
    // for (od=0; od<OD; od++) IEV[od] = malloc(JJ*sizeof(double));

    // double IEV_last[JJ], IEV_temp_norm, IEV_error, IEVAL;
    // int it_IEV=0, IEV_stop_next_it=0;

    // for (jj=0; jj<JJ; jj++) {
    //     if (jj<=IEV_ind) IEV[0][jj]=1.0;
    //     else IEV[0][jj]=0.0;
    // }

    // do{

    //     od=0;
    //     do{

    //         IEVAL = pow(OD_EVAL[od], (double)(IEV_ind+1));           

    //         IEV_temp_norm = 0.0;
    //         for (jj=0; jj<JJ; jj++) IEV_temp_norm+=IEV[od][jj]*IEV[od][jj];
    //         IEV_temp_norm = sqrt(IEV_temp_norm);

    //         for (jj=0; jj<JJ; jj++) IEV_temp[jj]=IEV[od][jj]/IEV_temp_norm;

    //         od--;
    //         if(od<0) od=OD-1;
    //         LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_R[od], JJ, IEV_temp, 1);
    //         if (LAPACK_info != 0) { exit(0); } 

    //         for (jj=0; jj<JJ; jj++) IEV[od][jj] = IEVAL*IEV_temp[jj]+OD_b[od][jj];

    //         // printf("w_%d^(%d)=%.17le %.17le %.17le\n", od+1, it_IEV, IEV[od][0], IEV[od][1], IEV[od][2]);

    //     }while(od>0);

    //     if (it_IEV>0){
    //         IEV_error = 0.0;
    //         for (jj=0; jj<JJ; jj++){ 
    //             IEV_error+=fabs(IEV[0][jj]-IEV_last[jj]); //To avoid errors, should consider the case of vectors w diff sign 
    //         }
                        
    //         if (IEV_error<IEV_TOL) IEV_stop_next_it++;
    //         // printf("error E_%d=%le, it_IEV=%d, IEV_stop_next_it=%d\n", IEV_ind+1, IEV_error, it_IEV, IEV_stop_next_it);
    //     }
    //     for (jj=0; jj<JJ; jj++){ IEV_last[jj]=IEV[0][jj]; }

    //     if (IEV_stop_next_it>1) IEV_stop_next_it++;
    //     it_IEV++;

    // }while(IEV_stop_next_it<2);

    double IEV_temp[JJ];
    for (od=0; od<OD-1; od++) {
        for (jj=0; jj<JJ; jj++) IEV_temp[jj] = OD_b[od][jj];
        cblas_dgemv(CblasRowMajor, CblasTrans, JJ, JJ, 1.0, OD_Q[od+1], JJ, IEV_temp, 1, 0, OD_b[od], 1);
        // LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_R[od], JJ, OD_b[od], 1);
        // if (LAPACK_info != 0) { exit(0); }
    }
    for (jj=0; jj<JJ; jj++) IEV_temp[jj] = OD_b[OD-1][jj];
    cblas_dgemv(CblasRowMajor, CblasTrans, JJ, JJ, 1.0, OD_Q[0], JJ, IEV_temp, 1, 0, OD_b[OD-1], 1);
    // LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_R[OD-1], JJ, OD_b[OD-1], 1);
    // if (LAPACK_info != 0) { exit(0); }

    if (verbose_b){
        for (od=0; od<OD; od++){
            printf("b_%d: ", od+1);
            for (jj=0; jj<JJ; jj++) printf("%.17le, ", OD_b[od][jj]);
            printf("\n");
        }
    }

    printf("\nINHOMOGENEOUS EIGENVECTORS:\n\n"); 

    double **IEV = malloc(OD*sizeof(double *));
    double **IEV_last = malloc(OD*sizeof(double *));
    double *IEVAL = malloc(OD*sizeof(double));

    for (od=0; od<OD; od++){
        IEV[od] = malloc(JJ*sizeof(double));
        IEV_last[od] = malloc(JJ*sizeof(double));
        for (jj=0; jj<JJ; jj++){
            // if (jj<=IEV_ind) IEV[od][jj] = 1.0;
            // else IEV[od][jj]=0.0;
            IEV[od][jj]=OD_b[od][jj];
        } 

        IEVAL[od] = OD_EVAL[od][IEV_ind]*OD_EVAL[od][IEV_ind];
    } 

    double IEV_temp_double, IEV_error, IEV_total_error;
    int it_IEV=0, IEV_stop_next_it=0, last_od, IEV_error_count;

    do{
        IEV_error_count=0;
        IEV_total_error=0.0;
        for (od=OD-1; od>=0; od--){

            last_od = ((od-1)+OD)%OD;

            for (jj=0; jj<JJ; jj++){

                if (jj<JJ-1){
                    IEV_temp_double=0.0;
                    for (cc=jj+1; cc<JJ; cc++) IEV_temp_double+=OD_R[od][cc+jj*JJ]*IEV[od][cc];
                    IEV[od][jj]=(-OD_b[od][jj]+IEVAL[od]*IEV[(od+1)%OD][jj]-IEV_temp_double)/OD_R[od][jj+jj*JJ];
                } else {
                    IEV[od][jj]=(OD_b[last_od][jj]+OD_R[last_od][jj+jj*JJ]*IEV[last_od][jj])/IEVAL[last_od];
                }

                if (verbose_wIEVEC){
                    if (jj==0) printf("w_%d^(%d)= ", od+1, it_IEV);
                    printf("%.17le ", IEV[od][jj]);
                    if (jj==JJ-1) printf("\n");
                }
            }

            IEV_error=0.0;
            for (jj=0; jj<JJ; jj++){
                if (it_IEV>0){
                    IEV_temp_double = (IEV[od][jj]-IEV_last[od][jj]);
                    IEV_error+=IEV_temp_double*IEV_temp_double;
                } 
                IEV_last[od][jj] = IEV[od][jj];
            }
            IEV_total_error+=IEV_error;
            IEV_error = sqrt(IEV_error);            
            if (IEV_error>IEV_TOL) IEV_error_count++; //Maybe should use square of TOL instead of computing sqrt of ERROR.

        }

        printf("it:%d,\tERROR count=%d, EV_total_error=%le\n",  it_IEV, IEV_error_count, IEV_total_error);
        it_IEV++;

    }while (it_IEV<2 || IEV_error_count>0);

    printf("w_%d:\t", IEV_ind+1);
    for (jj=0; jj<JJ; jj++) printf("%.17le ", IEV[0][jj]);
    printf("\t (%d iterations)\n", it_IEV);

    //Compute the product v=Qw to recover the inhomo. eigenvector
    cblas_dgemv(CblasRowMajor, CblasNoTrans, JJ, JJ, 1.0, OD_Q[0], JJ, IEV[0], 1, 0, IEV_temp, 1);
    fwrite(IEV_temp, sizeof(double), JJ, OD_IEVEC_bin);

    printf("IEV_%d:\t", IEV_ind+1);
    for (jj=0; jj<JJ; jj++) printf("%.17le, ", IEV_temp[jj]);
    printf("\n");

    free(IEV[0]);
    for (od=1; od<OD; od++){
        cblas_dgemv(CblasRowMajor, CblasNoTrans, JJ, JJ, 1.0, OD_Q[od], JJ, IEV[od], 1, 0, IEV_temp, 1);

        fwrite(IEV_temp, sizeof(double), JJ, OD_IEVEC_bin);

        if (verbose_IEVEC){
            printf("IEV_%d_(%d): ", IEV_ind+1, od+1);
            for (jj=0; jj<JJ; jj++) printf("%.17le, ", IEV_temp[jj]);
            printf("\n");
        }

        free(IEV[od]);
    }
    free(IEV);
    
    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n\n\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    return 0;
}