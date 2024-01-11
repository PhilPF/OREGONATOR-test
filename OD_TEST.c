#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <time.h>
#include <complex.h>
#include "method.h"
#include "rk.h"
#include "taylor.h"

#define JJ _NUMBER_OF_JET_VARS_
#define DD _MAX_DEGREE_OF_JET_VARS_
#define CC _JET_COEFFICIENTS_COUNT_TOTAL_

#define RKP (double)_RKP_
#define _2RKP pow(0.5, RKP)
#define h_fac pow(0.38, 1.0/(RKP+1.0))

#define RK_RTOL 1E-12
#define RK_ATOL RK_RTOL
#define H0 1E-7

#define IND 1
#define IEV_DEG 1
#define SEC_IND 1

#define OD_zero_TOL 1e-16
#define EV_TOL OD_zero_TOL

#define interactive_test   1

#define verbose_test       0
#define verbose_ssc        0
#define verbose_projection 1

MY_FLOAT s=77.27, w=0.161, q=8.375e-6, fff=1.0;


// 1e+2
// MAX ERROR
// x: 1.08681830170098692e-02 (2278431)    3.31655827475537990e+00 (5121141)   6.11988172020122700e+15 (5121141)   
// y: 9.68967948892895947e-04 (3272609)    5.85937500000000000e-03 (5121127)   9.34584883609600000e+12 (5121127)   
// z: 1.97354455103777582e-02 (3272609)    1.31835937500000000e-02 (5121127)   2.30897441832960000e+13 (5121127)   

// MIN ERROR
// x: 0.00000000000000000e+00      0.00000000000000000e+00     0.00000000000000000e+00     
// y: 0.00000000000000000e+00      0.00000000000000000e+00     0.00000000000000000e+00     
// z: 0.00000000000000000e+00      0.00000000000000000e+00     0.00000000000000000e+00     

// MEAN ERROR
// x: 4.20985567399679126e-08      1.81882983890169864e-06     2.58773143757873821e+09     
// y: 3.78893771405982321e-10      1.07372431378933442e-07     7.27411775863984972e+07     
// z: 3.11732822582597975e-08      9.79037665210856290e-08     1.04432371230678424e+08     

// Accepted=5616237, Rejected=326
// h_min=3.920074e-09, h_max=6.341651e-02

// Execution time: 929.578175 seconds



int main(int argc, char *argv[]){

    if (DD<=1){ printf("ERROR. The number of symbols is %d.\n", DD); exit(0);}
    if (IEV_DEG>DD) { printf("ERROR. The number of symbols (%d) does not match the degree (%d) of the inhomogeneous eigenvector.\n", DD, IEV_DEG); exit(0);}

    int jj, ss, dd, cc, i, od, iev_deg;
    int od_t=0, forced_od_bool=0;
    int OD;

    char test_continue; int test_it = 1;

    char OD_num_name[] = "ORBIT DECOMP/num_OD.txt", OD_part_name[] = "ORBIT DECOMP/partition.bin", OD_orbit_name[] = "ORBIT DECOMP/orbit.bin", OD_EVAL_name[] = "ORBIT DECOMP/EVAL.bin",  OD_EVEC_name[] = "ORBIT DECOMP/EVEC.bin", OD_IEVEC_name[] = "ORBIT DECOMP/IEVEC_2.bin";
    FILE *OD_num_txt, *OD_part_bin, *OD_orbit_bin, *OD_EVAL_bin, *OD_EVEC_bin, *OD_IEVEC_bin;

    //Reading number of OD partitions
    OD_num_txt = fopen(OD_num_name, "r");
    if (OD_num_txt==NULL) exit(0);
    fscanf(OD_num_txt, "%d", &OD);

    OD_part_bin = fopen(OD_part_name, "rb");
    if (OD_part_bin==NULL) exit(0);

    double *OD_t = malloc(OD*sizeof(double));
    fread(OD_t, sizeof(double), OD, OD_part_bin);

    OD_orbit_bin = fopen(OD_orbit_name, "rb");
    if (OD_orbit_bin==NULL) exit(0);
    OD_EVAL_bin = fopen(OD_EVAL_name, "rb");
    if (OD_EVAL_bin==NULL) exit(0);
    OD_EVEC_bin = fopen(OD_EVEC_name, "rb");
    if (OD_EVEC_bin==NULL) exit(0);
    OD_IEVEC_bin = fopen(OD_IEVEC_name, "rb");
    if (OD_IEVEC_bin==NULL) exit(0);

    double **OD_orbit = malloc(OD*sizeof(double *));
    double **OD_EVAL  = malloc(OD*sizeof(double *));
    double **OD_EVEC  = malloc(OD*sizeof(double *));
    double **OD_IEVEC = malloc(OD*sizeof(double *));
    for (od=0; od<OD; od++){
        OD_orbit[od] = malloc(JJ*sizeof(double));
        OD_EVAL[od]  = malloc(JJ*sizeof(double));
        OD_EVEC[od]  = malloc(JJ*sizeof(double));
        OD_IEVEC[od] = malloc(JJ*sizeof(double));

        fread(OD_orbit[od], sizeof(double), JJ, OD_orbit_bin);
        fread(OD_EVAL[od],  sizeof(double), JJ, OD_EVAL_bin);
        fread(OD_EVEC[od],  sizeof(double), JJ, OD_EVEC_bin);
        fread(OD_IEVEC[od], sizeof(double), JJ, OD_IEVEC_bin);
    } 

    char TEST_error_name[] = "ORBIT DECOMP/TEST_error.dat";
    FILE *TEST_error_dat;
    TEST_error_dat = fopen(TEST_error_name, "wb");
    if (TEST_error_dat==NULL) exit(0);

    double tmp_write[JJ*CC];

    int IEVEC_max_error_ind[JJ][CC];
    double IEVAL, IEVEC_error[JJ][CC], IEVEC_min_error[JJ][CC], IEVEC_max_error[JJ][CC], IEVEC_mean_error[JJ][CC];
    for (jj=0; jj<JJ; jj++) for (cc=0; cc<CC; cc++) { IEVEC_min_error[jj][cc]=1e+200; IEVEC_max_error[jj][cc]=0.0; IEVEC_max_error_ind[jj][cc]=0; IEVEC_mean_error[jj][cc]=0.0;}

    clock_t CLOCK_begin = clock();

    MY_FLOAT step_size, last_step_size, new_step_size, t, t_F;

    const char **var_names = taylor_get_variable_names();
    const char **monomials = taylor_get_jet_monomials();

    //Print first line with the monomials of the jet
    printf("\t");
    for (cc=0; cc<CC-1; cc++) { printf("\t\t%s", monomials[cc]);}
    printf("\n");

    taylor_initialize_jet_library();
    static MY_JET jet_xx[JJ], tmp[JJ], tmp_1[JJ], tmp_2[JJ];
    for (jj=0; jj<JJ; jj++) {InitJet(jet_xx[jj]); InitJet(tmp[jj]); InitJet(tmp_1[jj]); InitJet(tmp_2[jj]);}

    double rk_error, rk_TOL, h_min=1000, h_max=0;
    int AS=0, RS=0, last_rejected=0, ind_error_TOL; 

    for (jj=0; jj<JJ; jj++){
        MY_JET_DATA(jet_xx[jj], 0) = OD_orbit[0][jj];
        MY_JET_DATA(jet_xx[jj], 1) = OD_EVEC[0][jj];

        for (dd=2; dd<=IEV_DEG; dd++) MY_JET_DATA(jet_xx[jj], dd) = OD_IEVEC[0][jj];
        for (dd=IEV_DEG+1; dd<CC; dd++) MY_JET_DATA(jet_xx[jj], dd) = 0.0;
    }

    //Set initial and final time and step size
    t = 0.0; t_F = 302.858044027585265;
    step_size = H0; 

    //Print initial values
    // printf("t: %f\n", t);
    // for (jj=0; jj<JJ; jj++) {
    //     printf("%s: ", var_names[jj]);
    //     for (cc=0; cc<CC; cc++){ 
    //         printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
    //     }
    //     printf("\n");
    // }

    if (verbose_test || (interactive_test && od_t>test_it)){
        printf("START: t=%.17le, od=%d\n", t, od_t);
        for (jj=0; jj<JJ; jj++) {
            printf("%s: ", var_names[jj]);
            printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
            for (cc=1; cc<CC; cc++){ 
                printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],cc));
            }
            printf("\n");
        }
    }

    while(t<t_F){

        forced_od_bool = 0;
        // printf("t+step_size=%.17le, OD_t[od_t]=%.17le\n", t+step_size, OD_t[od_t]);
        if (t+step_size>=OD_t[od_t]){
            if (fabs(t+step_size-OD_t[od_t])>1e-200){
                step_size=OD_t[od_t]-t;
            } 
            forced_od_bool=1;
        }

        int info_RK;

        do{
            //Compute two steps of stepsize h/2
            info_RK = RK_Implicit(tmp_1, step_size/2, t, jet_xx);
            if (info_RK<0){ step_size/=2; continue;}

            info_RK = RK_Implicit(tmp_2, step_size/2, t+step_size/2, tmp_1);
            if (info_RK<0){ step_size/=2; continue;}


            //Compute one step of stepsize h
            info_RK = RK_Implicit(tmp, step_size, t, jet_xx);    
            if (info_RK<0){ step_size/=2; continue;}

        }while(info_RK!=0);
        
        rk_error = 0;
        for (cc=0;cc<CC; cc++){
            for (jj=0; jj<JJ; jj++){
                rk_TOL=fmax(fabs(MY_JET_DATA(jet_xx[jj], cc)), fabs(MY_JET_DATA(tmp_2[jj], cc)));
                rk_TOL = RK_ATOL+RK_RTOL*rk_TOL;

                rk_error+=pow((MY_JET_DATA(tmp[jj], cc)-MY_JET_DATA(tmp_2[jj], cc))/rk_TOL, 2.0);
            }
        }

        rk_error/=JJ;
        rk_error=sqrt(rk_error)/(1.0-_2RKP);

        new_step_size = fmax(0.5, h_fac*pow(1.0/rk_error, 1.0/(RKP+1.0)));

        if (rk_error>1){
            last_rejected=1;
            if (verbose_ssc) printf("REJECTED h=%le", step_size);
            RS++;
            step_size *= fmin(1, new_step_size);

            if (verbose_ssc) printf("--> NEW h=%le\n", step_size);
            continue;
        }

        for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(tmp_2[jj], cc); }}

        t+=step_size;

        AS++; 
        // fprintf(h_temp, "%lf %lf %d\n", t, step_size, 1);
        // printf("ACCEPTED h=%le", step_size);
        if (last_rejected==0){
            step_size *= fmin(2, new_step_size);
            // printf("--> NEW h=%le\n", step_size);
        } else {
            step_size *= fmin(1, new_step_size);
            // printf("--> NEW h=%le\n", step_size);
            last_rejected=0;
        }
        
        //Check if last step
        if (t+step_size>t_F){
            last_step_size = step_size;
            step_size = t_F-t;
        } else {
            if (step_size>h_max) h_max=step_size;
            if (step_size<h_min) h_min=step_size;
        }

        if (forced_od_bool){ 

            if (od_t+1<OD){

                IEVAL=OD_EVAL[od_t][IND];

                if (verbose_test || (interactive_test && od_t+1>=test_it)){
                    printf("\nEND: t=%.17le, EVAL=%le\n", t, IEVAL);

                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
                        for (cc=1; cc<CC; cc++){ 
                            printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],cc));
                        }
                        printf("\n");
                    }
                }
                for (jj=0; jj<JJ; jj++) {
                    IEVEC_error[jj][0] = MY_JET_DATA(jet_xx[jj],0);
                    for (cc=1; cc<CC; cc++){ 
                        IEVEC_error[jj][cc] = MY_JET_DATA(jet_xx[jj],cc);
                    }
                }

                od_t++;

                for (jj=0; jj<JJ; jj++){
                    MY_JET_DATA(jet_xx[jj], 0) = OD_orbit[od_t][jj];
                    MY_JET_DATA(jet_xx[jj], 1) = OD_EVEC[od_t][jj];
                    for (dd=2; dd<=IEV_DEG; dd++) MY_JET_DATA(jet_xx[jj], dd) = OD_IEVEC[od_t][jj];
                    for (dd=IEV_DEG+1; dd<CC; dd++) MY_JET_DATA(jet_xx[jj], dd) = 0.0;

                    for (cc=0; cc<CC; cc++){
                        IEVEC_error[jj][cc]-=pow(IEVAL, (double)cc)*MY_JET_DATA(jet_xx[jj],cc);
                        IEVEC_error[jj][cc]=fabs(IEVEC_error[jj][cc]);

                        IEVEC_mean_error[jj][cc]+=IEVEC_error[jj][cc]/(double)OD;

                        if (IEVEC_error[jj][cc]>IEVEC_max_error[jj][cc]){ IEVEC_max_error[jj][cc]=IEVEC_error[jj][cc]; IEVEC_max_error_ind[jj][cc] = od_t; }
                        if (IEVEC_error[jj][cc]<IEVEC_min_error[jj][cc]) IEVEC_min_error[jj][cc]=IEVEC_error[jj][cc];
                    } 

                }

                fprintf(TEST_error_dat, "%.17le ", OD_t[od_t]);
                for (jj=0; jj<JJ; jj++) for (cc=0; cc<CC; cc++) fprintf(TEST_error_dat, "%.17le ", IEVEC_error[jj][cc]);
                fprintf(TEST_error_dat, "\n");

                if (verbose_test || interactive_test && od_t>=test_it){
                    printf("\nSTART: t=%.17le, od=%d\n", t, od_t);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
                        for (cc=1; cc<CC; cc++){ 
                            printf("%.17le  \t", pow(IEVAL, (double)cc)*MY_JET_DATA(jet_xx[jj],cc));
                        }
                        printf("\n");
                    }
                }

                if (verbose_test || (interactive_test && od_t>=test_it)){
                    printf("\nERROR:\n");
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        for (cc=0; cc<CC; cc++){ 
                            printf("%.17le  \t", IEVEC_error[jj][cc]);
                        }
                        printf("\n");
                    }
                }

                if (interactive_test && od_t>=test_it){

                    printf("\nMAX ERROR (up to it=%d)\n", od_t);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        for (cc=0; cc<CC; cc++){ 
                            printf("%.17le (%d) \t", IEVEC_max_error[jj][cc], IEVEC_max_error_ind[jj][cc]);
                        }
                        printf("\n");
                    }

                    printf("\nMIN ERROR (up to it=%d)\n", od_t);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        for (cc=0; cc<CC; cc++){ 
                            printf("%.17le  \t", IEVEC_min_error[jj][cc]);
                        }
                        printf("\n");
                    }

                    int temp_test_it;
                    printf("\nContinue test? Specify number of iterations: (0,1, ...)\n");
                    scanf(" %d", &temp_test_it);
                    if (temp_test_it == 0) exit(0);
                    test_it+=temp_test_it;
                    // scanf(" %c", &test_continue);
                    // if (test_continue=='n' || test_continue=='N') break;
                

                }


            } else {

                IEVAL=OD_EVAL[od_t][IND];

                if(verbose_test || verbose_projection){
                    printf("\nEND: t=%.17le, EVAL=%le\n", t, IEVAL);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
                        for (cc=1; cc<CC; cc++){ 
                            printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],cc));
                        }
                        printf("\n");
                    }
                }

                //Projection {
                MY_FLOAT proj_normal[JJ], proj_out[JJ], proj_xx[JJ];
                for (jj=0; jj<JJ; jj++) proj_normal[jj]=0.0;
                proj_normal[SEC_IND]=1.0;
                
                double proj_tau, proj_tau_denom;
                for (jj=0; jj<JJ; jj++){ proj_xx[jj] = MY_JET_DATA(jet_xx[jj], 0);}
                FLOAT_function(proj_out, t, proj_xx);

                static MY_JET jet_TT, **function_jet, temporalJet[JJ];
                InitJet(jet_TT); for (jj=0; jj<JJ; jj++) { InitJet(temporalJet[jj]); } 
                for (cc=0; cc<CC; cc++) MY_JET_DATA(jet_TT, cc)=0.0;
                
                MY_FLOAT temporal_state_xx[JJ];

                double proj_num[JJ], proj_temp[JJ];

                for (dd=1; dd<=DD; dd++){
                    
                    for (jj=0; jj<JJ; jj++) { temporal_state_xx[jj]=0.0; }

                    proj_tau = 0.0; proj_tau_denom = 0.0;
                    for (jj=0; jj<JJ; jj++) {
                        proj_tau+=MY_JET_DATA(jet_xx[jj], dd)*proj_normal[jj];
                        proj_tau_denom+=proj_out[jj]*proj_normal[jj];
                    }
                    proj_tau/=-proj_tau_denom;

                    //Projection with Taylor
                    MY_JET_DATA(jet_TT, dd)=proj_tau;
                    if (verbose_projection){
                        printf("\njet_TT: ");
                        for (cc=0; cc<CC; cc++) printf("%le ", MY_JET_DATA(jet_TT, cc));
                    }

                    taylor_coefficients_taylor_A(t, temporal_state_xx, DD, 0, jet_xx, &function_jet);
                    for (jj=0; jj<JJ; jj++){
                        MultiplyJetJetA(temporalJet[jj], function_jet[jj][DD], jet_TT);
                        AddJetJetA(jet_xx[jj], function_jet[jj][DD-1], temporalJet[jj]);
                        for (ss=DD-2; ss>=0; ss--){
                            MultiplyJetJetA(temporalJet[jj], jet_xx[jj], jet_TT);
                            AddJetJetA(jet_xx[jj], function_jet[jj][ss], temporalJet[jj]);
                        }
                    }
                    MY_JET_DATA(jet_TT, dd)=0.0;

                    if(verbose_test || verbose_projection){
                        printf("\nEND: t=%.17le, EVAL=%le\n", t, IEVAL);
                        for (jj=0; jj<JJ; jj++) {
                            printf("%s: ", var_names[jj]);
                            printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
                            for (cc=1; cc<CC; cc++){ 
                                printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],cc));
                            }
                            printf("\n");
                        }
                    }
                }
                //}

                printf("\nEND: t=%.17le, EVAL=%le\n", t, IEVAL);
                for (jj=0; jj<JJ; jj++) {
                    printf("%s: ", var_names[jj]);
                    printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
                    for (cc=1; cc<CC; cc++){ 
                        printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],cc));
                    }
                    printf("\n");
                }
                for (jj=0; jj<JJ; jj++) {
                    IEVEC_error[jj][0] = MY_JET_DATA(jet_xx[jj],0);
                    for (cc=1; cc<CC; cc++){ 
                        IEVEC_error[jj][cc] = MY_JET_DATA(jet_xx[jj],cc);
                    }
                }

                od_t++;

                for (jj=0; jj<JJ; jj++){
                    MY_JET_DATA(jet_xx[jj], 0) = OD_orbit[0][jj];
                    MY_JET_DATA(jet_xx[jj], 1) = OD_EVEC[0][jj];
                    for (dd=2; dd<=IEV_DEG; dd++) MY_JET_DATA(jet_xx[jj], dd) = OD_IEVEC[0][jj];
                    for (dd=IEV_DEG+1; dd<CC; dd++) MY_JET_DATA(jet_xx[jj], dd) = 0.0;

                    for (cc=0; cc<CC; cc++){
                        IEVEC_error[jj][cc]-=pow(IEVAL, (double)cc)*MY_JET_DATA(jet_xx[jj],cc);
                        IEVEC_error[jj][cc]=fabs(IEVEC_error[jj][cc]);

                        IEVEC_mean_error[jj][cc]+=IEVEC_error[jj][cc]/(double)OD;

                        if (IEVEC_error[jj][cc]>IEVEC_max_error[jj][cc]){ IEVEC_max_error[jj][cc]=IEVEC_error[jj][cc]; IEVEC_max_error_ind[jj][cc] = od_t; }
                        if (IEVEC_error[jj][cc]<IEVEC_min_error[jj][cc]) IEVEC_min_error[jj][cc]=IEVEC_error[jj][cc];
                    } 

                }

                fprintf(TEST_error_dat, "%.17le ", 0.0);
                for (jj=0; jj<JJ; jj++) for (cc=0; cc<CC; cc++) fprintf(TEST_error_dat, "%.17le ", IEVEC_error[jj][cc]);
                fprintf(TEST_error_dat, "\n");

                printf("\nSTART: t=%.17le, od=%d\n", t, od_t);
                for (jj=0; jj<JJ; jj++) {
                    printf("%s: ", var_names[jj]);
                    printf("%.17le  \t", MY_JET_DATA(jet_xx[jj],0));
                    for (cc=1; cc<CC; cc++){ 
                        printf("%.17le  \t", pow(IEVAL, (double)cc)*MY_JET_DATA(jet_xx[jj],cc));
                    }
                    printf("\n");
                }
                printf("\nERROR:\n");
                for (jj=0; jj<JJ; jj++) {
                    printf("%s: ", var_names[jj]);
                    for (cc=0; cc<CC; cc++){ 
                        printf("%.17le  \t", IEVEC_error[jj][cc]);
                    }
                    printf("\n");
                }
                
            }
        }

    }

    printf("MAX ERROR\n");
    for (jj=0; jj<JJ; jj++) {
        printf("%s: ", var_names[jj]);
        for (cc=0; cc<CC; cc++){ 
            printf("%.17le (%d) \t", IEVEC_max_error[jj][cc], IEVEC_max_error_ind[jj][cc]);
        }
        printf("\n");
    }

    printf("\nMIN ERROR\n");
    for (jj=0; jj<JJ; jj++) {
        printf("%s: ", var_names[jj]);
        for (cc=0; cc<CC; cc++){ 
            printf("%.17le  \t", IEVEC_min_error[jj][cc]);
        }
        printf("\n");
    }

    printf("\nMEAN ERROR\n");
    for (jj=0; jj<JJ; jj++) {
        printf("%s: ", var_names[jj]);
        for (cc=0; cc<CC; cc++){ 
            printf("%.17le  \t", IEVEC_mean_error[jj][cc]);
        }
        printf("\n");
    }

    // // Print last values
    // printf("\nt: %.15le\n", t);
    // for (jj=0; jj<JJ; jj++) {
    //     printf("%s: ", var_names[jj]);
    //     for (cc=0; cc<CC; cc++){ 
    //         printf("%.15le   \t", MY_JET_DATA(jet_xx[jj],cc));
    //     }
    //     printf("\n");
    // }

    printf("\nAccepted=%d, Rejected=%d\nh_min=%le, h_max=%le\n", AS, RS, h_min, h_max);

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    return 0;
}