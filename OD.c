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

#define h_fac_min 0.5
#define h_fac_max 1.5
#define OD_fac_min h_fac_min//5.0
#define OD_fac_max h_fac_max //0.1

#define RK_RTOL 1E-12
#define RK_ATOL RK_RTOL
#define H0 1E-5

#define OD_MAX 5000000
#define OD_zero_TOL 1e-20

#define OD_H0_frac 1e5
#define OD_TOL 5
#define OD_kappa_TOL 10
#define OD_sigma_TOL 5.5

#define SEC_IND 2

#define store_ALL 1
#define store_as_dat 1

#define projection 0

#define verbose           1
#define verbose_none      1
#define verbose_ini_fin ( 0 || verbose ) && !verbose_none
#define verbose_JAC     ( 0 || verbose ) && !verbose_none
#define verbose_orbit   ( 0 || verbose ) && !verbose_none
#define verbose_proj    ( 1 || verbose ) && !verbose_none
#define verbose_OD_info ( 0 || verbose ) && !verbose_none
#define verbose_OD_magn ( 1 || verbose || verbose_OD_info ) && !verbose_none
#define verbose_OD_time ( 0 || verbose || verbose_OD_info ) && !verbose_none
#define verbose_OD_pos  ( 1 || verbose || verbose_OD_info ) && !verbose_none
#define verbose_OD_ssc  ( 1 || verbose || verbose_OD_info ) && !verbose_none
#define verbose_OD_part ( 1 || verbose || verbose_OD_info ) && !verbose_none

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double t, double t_F, int od) {
    double percentage = t/t_F;
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\rt=%3.3f, r=%d (%3d%%) [%.*s%*s]", t, od, val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}


MY_FLOAT s=77.27, w=0.161, q=8.375e-6, fff=1.0;

// 1e+5
// Accepted=7022408, Rejected=810533
// h_min=1.311911e-10, h_max=8.285155e-02
// Num of OD:1001080, with 102 decelerations (97 reached RK stepsize)

// Execution time: 1058.062892 seconds


// 1e+2
// Accepted=16819488, Rejected=311365
// h_min=9.947598e-14, h_max=2.675454e-02
// Num of OD:5121141, with 0 decelerations (0 reached RK stepsize)

// Execution time: 2296.460934 seconds

void sigma_condition(double *A_max, double *A_min, MY_JET *jet_xx, double *A){

    int jj, cc;
    double max=1e-200, min=1e+200, temp, _A[JJ*JJ];

    if (A==NULL) for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++) { _A[cc+jj*JJ]=MY_JET_DATA(jet_xx[jj], cc+1); }}
    else for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++) { _A[cc+jj*JJ]=A[cc+jj*JJ]; }}

    //USE EIGENVALUES
    double WR[JJ], WI[JJ];

    lapack_int info;
    info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'N', JJ, _A, JJ, WR, WI, NULL, 1, NULL, 1);

    if (info!=0){ printf("ERROR(%d): eigenvalue decomposition\n", info); exit(0);}

    for (jj=0; jj<JJ; jj++){
        temp = fabs(WR[jj]);
        if (temp>max && temp>1e-200) max=WR[jj];
        if (temp<min && temp>1e-200) min=WR[jj];
    }

    *A_max = max; *A_min = min;

}

void condition(double *A_max, double *A_min, MY_JET *jet_xx, double *A){

    int jj, cc;
    double max=1e-200, min=1e+200, temp, _A[JJ*JJ];

    if (A==NULL) for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++) { _A[cc+jj*JJ]=MY_JET_DATA(jet_xx[jj], cc+1); }}
    else for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++) { _A[cc+jj*JJ]=A[cc+jj*JJ]; }}

    //USE SINGLUAR VALUES
    double SV[JJ];

    lapack_int info;
    double *work = (double*)malloc(sizeof(double));
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'N', 'N', JJ, JJ, _A, JJ, SV, NULL, JJ, NULL, JJ, work);

    if (info!=0){ printf("ERROR(%d): singular value decomposition\n", info); exit(0);}

    for (jj=0; jj<JJ; jj++){
        temp = fabs(SV[jj]);
        if (temp>max && temp>1e-200) max=SV[jj];
        if (temp<min && temp>1e-200) min=SV[jj]; 
    }

    *A_max = max; *A_min = min;

}

int main(int argc, char *argv[]){

    clock_t CLOCK_begin = clock();

    int jj, ss, dd, cc, i, od=0, RK_it=0;

    MY_FLOAT step_size, last_step_size, new_step_size, t, t_F;
    
    char OD_num_name[] = "ORBIT DECOMP/num_OD.txt", OD_part_name[] = "ORBIT DECOMP/partition.bin", OD_orbit_name[] = "ORBIT DECOMP/orbit.bin", OD_A_name[] = "ORBIT DECOMP/A.bin";
    char OD_part_dat_name[] = "ORBIT DECOMP/partition.dat", OD_orbit_dat_name[] = "ORBIT DECOMP/orbit.dat", OD_A_dat_name[] = "ORBIT DECOMP/A.dat";
    FILE *OD_num_txt, *OD_part_bin, *OD_orbit_bin, *OD_A_bin; 
    FILE *OD_part_dat, *OD_orbit_dat, *OD_A_dat;
    if (store_ALL){
        OD_num_txt = fopen(OD_num_name, "w");
        OD_part_bin = fopen(OD_part_name, "wb");
        OD_orbit_bin = fopen(OD_orbit_name, "wb");
        OD_A_bin = fopen(OD_A_name, "wb");
    }
    if (store_as_dat){
        OD_part_dat = fopen(OD_part_dat_name, "w");
        OD_orbit_dat = fopen(OD_orbit_dat_name, "w");
        OD_A_dat = fopen(OD_A_dat_name, "w");
    }

    const char **var_names = taylor_get_variable_names();
    const char **monomials = taylor_get_jet_monomials();

    if (verbose_ini_fin){
        //Print first line with the monomials of the jet
        printf("\t");
        for (cc=0; cc<CC-1; cc++) { printf("\t\t%s", monomials[cc]);}
        printf("\n");
    }

    taylor_initialize_jet_library();
    static MY_JET jet_xx[JJ], tmp[JJ], tmp_1[JJ], tmp_2[JJ];
    for (jj=0; jj<JJ; jj++) {InitJet(jet_xx[jj]); InitJet(tmp[jj]); InitJet(tmp_1[jj]); InitJet(tmp_2[jj]);}

    double rk_error, rk_TOL, h_min=1000, h_max=0;
    int AS=0, RS=0, last_rejected=0, ind_error_TOL; 

    MY_JET_DATA(jet_xx[0], 0) = 1.000566421716767;
    MY_JET_DATA(jet_xx[1], 0) = 1766.454273655898987;
    MY_JET_DATA(jet_xx[2], 0) = 4000.000000000010459;

    double tmp_write[JJ];
    if (store_ALL){
        for (jj=0; jj<JJ; jj++) tmp_write[jj] = MY_JET_DATA(jet_xx[jj], 0);
        fwrite(tmp_write, sizeof(double), JJ, OD_orbit_bin);

        if (store_as_dat){ for (jj=0; jj<JJ; jj++){ fprintf(OD_orbit_dat, "%.17le ", tmp_write[jj]); } fprintf(OD_orbit_dat, "\n");}
    }

    for (jj=0; jj<JJ; jj++){
        for (cc=1; cc<CC; cc++){
            if (jj+1==cc) MY_JET_DATA(jet_xx[jj], cc) = 1.0;
            else MY_JET_DATA(jet_xx[jj], cc) = 0.0;
        }
    }
    if (projection) MY_JET_DATA(jet_xx[SEC_IND], SEC_IND+1) = 0.0;

    //Set initial and final time and step size
    t = 0.0; t_F = 302.858044027579183;
    step_size= H0; 

    double **OD_JAC = malloc(OD_MAX*sizeof(double *));
    for (jj=0; jj<OD_MAX; jj++) OD_JAC[jj] = malloc(JJ*JJ*sizeof(double));

    double last_OD_JAC[JJ*JJ];

    int OD_partition_bool=0, OD_last_rejected=0, is_last_step=0, OD_RS=0, OD_RK_RS=0;
    double OD_stepsize=0.0/*t_F/OD_H0_frac*/;
    // if (OD_stepsize<step_size) OD_stepsize=2*step_size;
    // double OD_position=t, OD_new_stepsize, OD_error;
    // double OD_last_t = t, OD_last_stepsize=step_size, OD_last_OD_position=OD_position, OD_last_OD_stepsize=OD_stepsize, OD_temp_double[JJ*CC];
    double OD_max_magnitude, OD_min_magnitude/*, OD_kappa_cond, OD_kappa_error, OD_sigma_cond, OD_last_sigma_cond=-1.0, OD_max_lambda, OD_min_lambda, OD_sigma_error*/;
    double OD_temp_sigma, OD_temp_kappa=0.0, OD_temp_gamma=0.0, OD_cond_last_norm=0.0;
    static MY_JET OD_last_jet[JJ];
    for (jj=0; jj<JJ; jj++) {
        InitJet(OD_last_jet[jj]);
        for (cc=0; cc<CC; cc++) MY_JET_DATA(OD_last_jet[jj], cc) = MY_JET_DATA(jet_xx[jj], cc); 
    }

    if (verbose_ini_fin){
        //Print initial values
        printf("t: %f\n", t);
        for (jj=0; jj<JJ; jj++) {
            printf("%s: ", var_names[jj]);
            for (cc=0; cc<CC; cc++){ 
                printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
            }
            printf("\n");
        }
    }

    while(t<t_F){

        if (step_size<1e-200){ printf("ERROR. The stepsize is negative or 0 (%.15le)\n", step_size); exit(0);}

        int info_RK;

        do{
            //Compute two steps of stepsize h/2
            info_RK = RK_Implicit(tmp_1, step_size/2, t, jet_xx);
            if (info_RK<0){ is_last_step=0; step_size/=2; continue;}

            info_RK = RK_Implicit(tmp_2, step_size/2, t+step_size/2, tmp_1);
            if (info_RK<0){ is_last_step=0; step_size/=2; continue;}


            //Compute one step of stepsize h
            info_RK = RK_Implicit(tmp, step_size, t, jet_xx);    
            if (info_RK<0){ is_last_step=0; step_size/=2; continue;}

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

        new_step_size = fmax(h_fac_min, h_fac*pow(1.0/rk_error, 1.0/(RKP+1.0)));

        if (rk_error>1){
            last_rejected=1;
            // printf("REJECTED h=%le", step_size);
            RS++;
            // fprintf(h_temp, "%lf %lf %d\n", t, step_size, 0);
            step_size *= fmin(1.0, new_step_size);

            OD_partition_bool=0;

            // printf("--> NEW h=%le\n", step_size);
            continue;
        }


        if (verbose_OD_time) printf("\nt=%.17lf --> t=%.17lf (step_size=%.17le)\n", t, t+step_size, step_size);


        OD_stepsize+=step_size;

        sigma_condition(&OD_max_magnitude, &OD_min_magnitude, tmp, NULL);

        OD_temp_kappa = fmax(OD_temp_kappa, OD_max_magnitude/OD_min_magnitude);
        OD_temp_gamma +=step_size*fmax(OD_max_magnitude/OD_min_magnitude, OD_cond_last_norm);

        OD_temp_sigma = OD_stepsize*OD_temp_kappa/OD_temp_gamma;

        // printf("kappa=%le, T*gamma=%le, T=%le, sigma=%le\n", OD_temp_kappa, OD_temp_gamma, OD_stepsize, OD_temp_sigma);

        OD_cond_last_norm=OD_max_magnitude/OD_min_magnitude;

        if (OD_temp_sigma>OD_sigma_TOL && !is_last_step){

            if (verbose_OD_part) printf("PARTITION! (num %d), OD_stepsize=%le\n", od+1, OD_stepsize);

            if (store_ALL){
                for (jj=0; jj<JJ; jj++) tmp_write[jj] = MY_JET_DATA(tmp[jj], 0);
                fwrite(tmp_write, sizeof(double), JJ, OD_orbit_bin);

                if (store_as_dat){ for (jj=0; jj<JJ; jj++){ fprintf(OD_orbit_dat, "%.17le ", tmp_write[jj]); } fprintf(OD_orbit_dat, "\n");}
            }

            for (jj=0; jj<JJ; jj++){
                for (cc=0; cc<JJ; cc++){
                    OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(tmp[jj], cc+1);
                }
            }

            for (jj=0; jj<JJ; jj++){
                for (cc=1; cc<CC; cc++){
                    if (jj+1==cc) MY_JET_DATA(tmp[jj], cc) = 1.0;
                    else MY_JET_DATA(tmp[jj], cc) = 0.0;
                }
            }

            if (verbose_JAC){
                printf("JAC[%d]: \n", od+1);
                for (jj=0; jj<JJ; jj++) {
                    for (cc=0; cc<JJ; cc++){ 
                        printf("%.17le, \t", OD_JAC[od][cc+jj*JJ]);
                    }
                    printf("\n");
                }
            }

            if (store_ALL) fwrite(&t, sizeof(double), 1, OD_part_bin);
            if (store_as_dat) fprintf(OD_part_dat, "%.17le\n", t);

            od++;

            OD_temp_kappa=0.0;
            OD_temp_gamma=0.0;
            OD_cond_last_norm = 0.0;
            OD_stepsize = 0.0;

        }


        //PARTITION BY MAGNITUDE-CONTROLLED PERIOD-FRACTION WITH TIME TRAVEL


        // if (OD_partition_bool){
            
        //     if (verbose_OD_pos) printf("OD: position=%.17lf (step_size=%.17le)\n", OD_position+OD_stepsize, OD_stepsize);

        //     OD_partition_bool=0;

            


        //     // OD_kappa_cond = OD_max_magnitude/OD_min_magnitude;
        //     // OD_kappa_error = OD_stepsize*OD_kappa_cond/OD_kappa_TOL;

        //     // OD_error = log10(OD_kappa_cond)/OD_TOL;

        //     if (verbose_OD_magn) printf("MAGNITUDES: MAX=%le, MIN=%le, ERROR=%lf, 1/ERROR=%lf\n", OD_max_magnitude, OD_min_magnitude, OD_error, 1/OD_error);

        //     OD_new_stepsize = fmax(OD_fac_min, 0.95*(1.0/OD_kappa_error));

        //     /*sigma_condition(&OD_max_lambda, &OD_min_lambda, tmp, NULL);
        //     OD_sigma_cond = log10(OD_max_lambda/OD_min_lambda);

        //     if (OD_last_sigma_cond==-1.0) OD_last_sigma_cond=OD_sigma_cond;
        //     OD_sigma_error=fabs(OD_sigma_cond-OD_last_sigma_cond)/OD_sigma_TOL;

        //     printf("sigma_cond=%lf, last_sigma_cond=%lf, sigma_error=%lf, 1/sigma_error=%le\n", OD_sigma_cond, OD_last_sigma_cond, OD_sigma_error, 1.0/OD_sigma_error);*/

        //     if (/*OD_error>1 || */OD_kappa_error>1 /* || OD_sigma_error>1*/){

        //         // if (OD_error<=1 && OD_kappa_error>1) OD_new_stepsize = fmax(OD_fac_min, 0.95*(1.0/OD_kappa_error));

        //         // if (OD_error<=1 && OD_sigma_error>1) OD_new_stepsize = fmax(OD_fac_min, 0.95*(1.0/OD_sigma_error));

        //         OD_last_rejected=1;

        //         OD_RS++;
        //         if (verbose_OD_ssc) printf("--> OD deceleration (%d)", OD_RS);
        //         for (jj=0; jj<JJ; jj++) {
        //             for (cc=0; cc<CC; cc++) MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(OD_last_jet[jj], cc); 
        //         }

        //         t = OD_last_t;
        //         step_size = OD_last_stepsize;
        //         OD_position = OD_last_OD_position;
        //         OD_stepsize = OD_last_OD_stepsize;

        //         // if (verbose_OD_ssc){
        //         //     printf(" (OD_last: t=%.17lf, step_size=%.17le, OD_position=%.17lf, OD_stepsize=%.17le)\n\t\t\t OD_last_jet=\n", t, step_size, OD_position, OD_stepsize);
        //         //     for (jj=0; jj<JJ; jj++) {
        //         //         printf("\t\t\t %s: ", var_names[jj]);
        //         //         for (cc=0; cc<CC; cc++){ 
        //         //             printf("%le \t", MY_JET_DATA(OD_last_jet[jj],cc));
        //         //         }
        //         //         printf("\n");
        //         //     }
        //         // }

        //         OD_stepsize *= fmin(1.0, OD_new_stepsize);
        //         OD_last_OD_stepsize = OD_stepsize;

        //         if (OD_stepsize<step_size){ 
        //             if (verbose_OD_ssc) printf("\nOD stepsize reached RK stepsize!!\n"); 
        //             step_size*=OD_new_stepsize;
        //             OD_last_stepsize = step_size;
        //             OD_stepsize=1.2*step_size;
        //             OD_RK_RS++;
        //         }

        //         if (t+step_size>=OD_position+OD_stepsize && t+step_size<t_F){
        //             step_size=(OD_position+OD_stepsize)-t;
        //             OD_partition_bool=1;
        //         }

        //         if (verbose_OD_ssc) printf(" --> stepsize=%.15le\n", OD_stepsize);

        //         continue;
        //     }  


        //     if (verbose_OD_part) printf("PARTITION! (num %d), OD_stepsize=%le\n", od+1, OD_stepsize);

        //     if (store_ALL){
        //         for (jj=0; jj<JJ; jj++) tmp_write[jj] = MY_JET_DATA(tmp[jj], 0);
        //         fwrite(tmp_write, sizeof(double), JJ, OD_orbit_bin);

        //         if (store_as_dat){ for (jj=0; jj<JJ; jj++){ fprintf(OD_orbit_dat, "%.17le ", tmp_write[jj]); } fprintf(OD_orbit_dat, "\n");}
        //     }

        //     for (jj=0; jj<JJ; jj++){
        //         for (cc=0; cc<JJ; cc++){
        //             OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(tmp[jj], cc+1);
        //         }
        //     }

        //     for (jj=0; jj<JJ; jj++){
        //         for (cc=1; cc<CC; cc++){
        //             if (jj+1==cc) MY_JET_DATA(tmp[jj], cc) = 1.0;
        //             else MY_JET_DATA(tmp[jj], cc) = 0.0;
        //         }
        //     }

        //     if (verbose_JAC){
        //         printf("JAC[%d]: \n", od+1);
        //         for (jj=0; jj<JJ; jj++) {
        //             for (cc=0; cc<JJ; cc++){ 
        //                 printf("%.17le, \t", OD_JAC[od][cc+jj*JJ]);
        //             }
        //             printf("\n");
        //         }
        //     }

        //     if (store_ALL) fwrite(&OD_last_t, sizeof(double), 1, OD_part_bin);
        //     if (store_as_dat) fprintf(OD_part_dat, "%.17le\n", OD_last_t);

        //     od++;

        //     OD_last_sigma_cond=OD_sigma_cond;

        //     OD_position+=OD_stepsize;

        //     if (OD_last_rejected==0){
        //         OD_stepsize *= fmin(OD_fac_max, OD_new_stepsize);
        //     } else {
        //         OD_stepsize *= fmin(1.0, OD_new_stepsize);
        //         OD_last_rejected=0;
        //     }      

        //     if (OD_stepsize<step_size){ 
        //         if (verbose_OD_ssc) printf("\nOD stepsize reached RK stepsize!!\n\t"); 
        //         step_size*=0.95;
        //         OD_last_stepsize = step_size;
        //         OD_stepsize=1.2*step_size;
        //         OD_RK_RS++;
        //     }
        
        //     OD_last_t = t+step_size;
        //     OD_last_stepsize = step_size;
        //     OD_last_OD_position = OD_position;
        //     OD_last_OD_stepsize = OD_stepsize;
        //     for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){  MY_JET_DATA(OD_last_jet[jj], cc) = MY_JET_DATA(tmp[jj], cc); }}

        // }
        // //END 

        if (od>=OD_MAX){ printf("ERROR OD_MAX<od. Increase OD_MAX and try again.\n"); exit(0);} 

        for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(tmp[jj], cc); }}

        t+=step_size;

        AS++; RK_it++;
        // fprintf(h_temp, "%lf %lf %d\n", t, step_size, 1);
        // printf("ACCEPTED h=%le", step_size);
        if (last_rejected==0){
            step_size *= fmin(h_fac_max, new_step_size);
            // printf("--> NEW h=%le\n", step_size);
        } else {
            step_size *= fmin(1.0, new_step_size);
            // printf("--> NEW h=%le\n", step_size);
            last_rejected=0;
        }

        if (verbose_orbit){
            printf("\nt: %f\n", t);
            for (jj=0; jj<JJ; jj++) {
                printf("%s: ", var_names[jj]);
                for (cc=0; cc<CC; cc++){ 
                    printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
                }
                printf("\n");
            }
        }
        
        // Check if last step
        if (t+step_size>t_F){
            last_step_size = step_size;
            step_size = t_F-t;
            is_last_step=1;
            // OD_partition_bool = 0;
        } else {
            //PARTITION BY PERIOD AND MAGNITUDE WITH TIME TRAVEL
            // if (t+step_size>=OD_position+OD_stepsize){

            //     step_size=(OD_position+OD_stepsize)-t;
            //     OD_partition_bool=1;

            // }
            //END
            if (step_size>h_max) h_max=step_size;
            if (step_size<h_min) h_min=step_size;
        }

        printProgress(t,t_F,od);

    }

    if (store_ALL) fwrite(&t, sizeof(double), 1, OD_part_bin);
    if (store_as_dat) fprintf(OD_part_dat, "%.17le\n", t);

    for (jj=0; jj<JJ; jj++){
        for (cc=0; cc<JJ; cc++){
            OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(jet_xx[jj], cc+1);
        }
    }

    if (verbose_proj || verbose_JAC){
        printf("JAC[%d]: \n", od+1);
        for (jj=0; jj<JJ; jj++) {
            for (cc=0; cc<JJ; cc++){ 
                printf("%.17le, \t", OD_JAC[od][cc+jj*JJ]);
            }
            printf("\n");
        }
    }

    od++;

    if (store_ALL) fprintf(OD_num_txt,"%d", od);


    //////////////////

    MY_FLOAT proj_normal[JJ], proj_out[JJ], proj_xx[JJ];
    for (jj=0; jj<JJ; jj++) proj_normal[jj]=0.0;
    proj_normal[SEC_IND] = 1.0;

    double proj_tau[JJ], proj_tau_denom;
    for (jj=0; jj<JJ; jj++){ proj_xx[jj] = MY_JET_DATA(jet_xx[jj], 0);}
    FLOAT_function(proj_out, t_F, proj_xx);
    if (verbose_proj){
        printf("\nx*: ");
        for (jj=0; jj<JJ; jj++) printf("%.17le ", proj_xx[jj]);
        printf("\nf(x*): ");
        for (jj=0; jj<JJ; jj++) printf("%.17le ", proj_out[jj]);
        printf("\n");
    }

    // double proj_col[JJ], proj_temp[JJ], proj_temp_1[JJ], proj_col_norm, proj_col_norm_temp;
    
    // proj_col_norm=0.0;
    // for (jj=0; jj<JJ; jj++) proj_col_norm+=proj_out[jj]*proj_out[jj];
    // proj_col_norm=sqrt(proj_col_norm);
    // for (jj=0; jj<JJ; jj++) proj_temp_1[jj]=proj_out[jj]/proj_col_norm;

    // for (i=0; i<od; i++){
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, JJ, JJ, 1.0, OD_JAC[i], JJ, proj_temp_1, 1, 0, proj_temp, 1);

    //     for (jj=0; jj<JJ; jj++) proj_temp_1[jj]=proj_temp[jj];
    // }
    // printf("M·f(x*)/||f(*)||=" );
    // for (jj=0; jj<JJ; jj++) printf("%.17le ", proj_temp[jj]);
    // printf("\n");

    static MY_JET jet_TT, **function_jet, temporalJet[JJ];
    InitJet(jet_TT); for (jj=0; jj<JJ; jj++) { InitJet(temporalJet[jj]); } 
    for (cc=0; cc<CC; cc++) MY_JET_DATA(jet_TT, cc)=0.0;

    for (dd=0; dd<JJ; dd++){

        // if (dd!=SEC_IND){

        //     proj_col_norm=0.0; proj_col_norm_temp=0.0; 

        //     printf("A_1_(%d)= ", dd+1);
        //     for (jj=0; jj<JJ; jj++){
        //         proj_col[jj]=OD_JAC[0][dd+jj*JJ];
        //         proj_col_norm_temp+=proj_col[jj]*proj_col[jj];

        //         printf("%.17le ", proj_col[jj]);
        //     }
        //     printf("\n");

        //     proj_col_norm_temp=sqrt(proj_col_norm_temp);
        //     proj_col_norm+=log10(proj_col_norm_temp);
        //     for (jj=0; jj<JJ; jj++){
        //         proj_col[jj]/=proj_col_norm_temp;
        //         proj_temp[jj]=proj_col[jj];
        //     }

        //     for (i=1; i<od; i++){
        //         printf("A_%d:\n", i+1);
        //         for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++){ printf("%.17le ", OD_JAC[i][cc+jj*JJ]);} printf("\n");}
        //         cblas_dgemv(CblasRowMajor, CblasNoTrans, JJ, JJ, 1.0, OD_JAC[i], JJ, proj_temp, 1, 0, proj_col, 1);

        //         proj_col_norm_temp=0.0; 
        //         for (jj=0; jj<JJ; jj++){
        //             proj_col_norm_temp+=proj_col[jj]*proj_col[jj];
        //         }
        //         proj_col_norm_temp=sqrt(proj_col_norm_temp);
        //         proj_col_norm+=log10(proj_col_norm_temp);

        //         for (jj=0; jj<JJ; jj++){
        //             proj_col[jj]/=proj_col_norm_temp;
        //             proj_temp[jj]=proj_col[jj];
        //         }

        //     }


        //     printf("A_(%d)=1e%lf · ( ", dd+1, proj_col_norm);
        //     for (jj=0; jj<JJ; jj++) printf("%.17le ", proj_col[jj]);
        //     printf(")\n");


            proj_tau[dd] = 0.0; proj_tau_denom = 0.0;
            for (jj=0; jj<JJ; jj++) {
                // proj_tau[dd]+=proj_col[jj]*proj_normal[jj]*pow(10.0,proj_col_norm);
                proj_tau[dd]+=MY_JET_DATA(jet_xx[jj], dd+1)*proj_normal[jj];
                proj_tau_denom+=proj_out[jj]*proj_normal[jj];
            }
            proj_tau[dd]/=-proj_tau_denom;

            // printf("(A_r)_(%d)= ", dd+1);
            // for (jj=0; jj<JJ; jj++) printf("%.17le ", MY_JET_DATA(jet_xx[jj], dd+1));
            // printf("\n");

            if (verbose_proj) printf("tau=%.17le\n", proj_tau[dd]);

            MY_JET_DATA(jet_TT, dd+1)=proj_tau[dd];

        // }
    }

    if (verbose_proj){
        printf("jet_TT: ");
        for (cc=0; cc<CC; cc++) printf("%le ", MY_JET_DATA(jet_TT, cc));
        printf("\n");
    }

    //////////////////

    if (projection){

        // //Projection with Taylor
        MY_FLOAT temporal_state_xx[JJ];
        for (jj=0; jj<JJ; jj++) { temporal_state_xx[jj]=MY_JET_DATA(jet_xx[jj], 0); }
        taylor_coefficients_taylor_A(t_F, temporal_state_xx, 1, 0, jet_xx, &function_jet);
        for (jj=0; jj<JJ; jj++){
            MultiplyJetJetA(temporalJet[jj], function_jet[jj][1], jet_TT);
            AddJetJetA(jet_xx[jj], function_jet[jj][0], temporalJet[jj]);
        }
        
        //Projection with (3.6)
        // for (dd=0; dd<JJ; dd++) for (jj=0; jj<JJ; jj++) MY_JET_DATA(jet_xx[jj], dd+1)+=proj_out[jj]*proj_tau[dd];    

        // for (dd=0; dd<JJ; dd++){
        //     if (fabs(MY_JET_DATA(jet_xx[SEC_IND], dd+1))>1e-200){
        //         printf("\nWARNING: coefficient %d (w. value %le) of projection has been forced to be 0.\n", dd+1, MY_JET_DATA(jet_xx[SEC_IND], dd+1));
        //         MY_JET_DATA(jet_xx[SEC_IND], dd+1) = 0.0;
        //     }
        // }

        //}

        for (jj=0; jj<JJ; jj++){
            for (cc=0; cc<JJ; cc++){
                OD_JAC[od-1][cc+jj*JJ] = MY_JET_DATA(jet_xx[jj], cc+1);
            }
        }

        if (verbose_proj){
            printf("\nJAC[%d]: (projected)\n", od);
            for (jj=0; jj<JJ; jj++) {
                for (cc=0; cc<JJ; cc++){ 
                    printf("%.17le, \t", OD_JAC[od-1][cc+jj*JJ]);
                }
                printf("\n");
            }
        }

    }

    if (verbose_ini_fin){
        // Print last values
        printf("\nt: %.15le\n", t);
        for (jj=0; jj<JJ; jj++) {
            printf("%s: ", var_names[jj]);
            for (cc=0; cc<CC; cc++){ 
                printf("%.15le   \t", MY_JET_DATA(jet_xx[jj],cc));
            }
            printf("\n");
        }
    }

    printf("\nAccepted=%d, Rejected=%d\nh_min=%le, h_max=%le\n", AS, RS, h_min, h_max);

    if (store_ALL){
        for (i=0; i<od; i++) {
            fwrite(OD_JAC[i], sizeof(double), JJ*JJ, OD_A_bin);
            if (store_as_dat){ for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++){ fprintf(OD_A_dat, "%.17le ", OD_JAC[i][cc+jj*JJ]); } fprintf(OD_A_dat, "\n");} fprintf(OD_A_dat, "\n");}
        }
        free (OD_JAC[i]);
    }
    free(OD_JAC);

    printf("Num of OD:%d, with %d decelerations (%d reached RK stepsize)\n", od, OD_RS, OD_RK_RS);

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    return 0;
}