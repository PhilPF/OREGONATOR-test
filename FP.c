#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <time.h>
#include "method.h"
#include "rk.h"
#include "taylor.h"

#define JJ _NUMBER_OF_JET_VARS_
#define DD _MAX_DEGREE_OF_JET_VARS_
#define CC _JET_COEFFICIENTS_COUNT_TOTAL_

#define RKP (double)_RKP_
#define _2RKP pow(0.5, RKP)
#define h_fac pow(0.38, 1.0/(RKP+1.0))

#define RK_RTOL 1E-10
#define RK_ATOL RK_RTOL
#define H0 1E-7

#define SEC_INDEX 2
#define SEC_CONST 4000.0
#define SEC_SAFE_MIN 100.0

#define TIME_NEWT_TOL 1E-10
#define FP_NEWT_TOL TIME_NEWT_TOL


MY_FLOAT s=77.27, w=0.161, q=8.375e-6, fff=1.0;

int get_sign(MY_FLOAT xx){

    if (xx>0) return 1;
    if (xx<0) return -1;

    return 0;

}

int TIME_NEWT_MAX_IT = 50, FP_NEWT_MAX_IT = 50; char it_continue;

void TIME_step_Newt(MY_FLOAT *T, MY_FLOAT T0, MY_JET *jet_xx){

    int jj;

    MY_FLOAT out[JJ], xx[JJ];
    for (jj=0; jj<JJ; jj++){
        xx[jj]=MY_JET_DATA(jet_xx[jj], 0);
    }

    FLOAT_function(out, T0, xx);

    *T = T0-(xx[SEC_INDEX]-SEC_CONST)/out[SEC_INDEX];

}

void FP_step_Newt(MY_FLOAT *p, MY_JET *jet_xx){

    int jj, cc, dd;
    double P_0[JJ-1], dP_0[(JJ-1)*(JJ-1)];

    dd=0;
    for (jj=0; jj<JJ; jj++){
        if(jj!=SEC_INDEX){
            P_0[dd]=MY_JET_DATA(jet_xx[jj], 0)-p[jj];
            for (cc=0; cc<CC-1; cc++){
                dP_0[dd*(JJ-1)+cc]=MY_JET_DATA(jet_xx[jj], cc+1);
                if (dd==cc){ dP_0[dd*(JJ-1)+cc]-=1;}
            }

            dd++;
        } 
    } 

    lapack_int LAPACK_info, *LAPACK_ipiv;
    LAPACK_ipiv = (lapack_int *)malloc((JJ-1)*sizeof(lapack_int)) ;
    LAPACK_info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, (JJ-1), 1, dP_0, (JJ-1), LAPACK_ipiv, P_0, 1);
    if (LAPACK_info != 0) { exit(0); } 

    dd=0;
    for (jj=0; jj<JJ; jj++){
        if (jj!=SEC_INDEX){
            p[jj] -= P_0[dd];
            dd++;
        } 
    }

}


int main(int argc, char *argv[]){

    clock_t CLOCK_begin = clock();

    int jj, ss, dd, cc, i;

    MY_FLOAT step_size, last_step_size, new_step_size, t, t_F;

    // char orbit_file_tex[20], orbit_file_dat[20];
    // char orbit_folder_name[20] = "ORBIT/";
    // snprintf(orbit_file_tex, 100, "set output '%sorbit.tex'", orbit_folder_name);
    // snprintf(orbit_file_dat, 100, "%sorbit.dat", orbit_folder_name);

    // const int ORBIT_numCommandsForGnuplot=8;
    // char * ORBIT_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", "set format y '$$%.0e$$'", orbit_file_tex, "set key reverse Left left", "set logscale y", "set xlabel '$t$'"}; //For vdpol_1_1
    // FILE * ORBIT_temp = fopen(orbit_file_dat, "w");
    // FILE * ORBIT_gnuplotPipe = popen ("gnuplot -persistent", "w");
    // for (i=0; i < ORBIT_numCommandsForGnuplot; i++){
    //     fprintf(ORBIT_gnuplotPipe, "%s \n", ORBIT_commandsForGnuplot[i]); 
    // }

    const char **var_names = taylor_get_variable_names();
    const char **monomials = taylor_get_jet_monomials();

    //Print first line with the monomials of the jet
    printf("\t");
    for (cc=0; cc<CC-1; cc++) { printf("\t\t%s", monomials[cc]);}
    printf("\n");

    taylor_initialize_jet_library();
    static MY_JET jet_xx[JJ], tmp[JJ], tmp_1[JJ], tmp_2[JJ];
    for (jj=0; jj<JJ; jj++) {InitJet(jet_xx[jj]); InitJet(tmp[jj]); InitJet(tmp_1[jj]); InitJet(tmp_2[jj]);}

    MY_JET_DATA(jet_xx[0], 0) = 1.0;
    MY_JET_DATA(jet_xx[1], 0) = 1700.0;
    MY_JET_DATA(jet_xx[2], 0) = 4000.0;

    int FP_it=0;
    double FP_norm;
    MY_FLOAT FP[JJ];
    for (jj=0; jj<JJ; jj++) FP[jj]=MY_JET_DATA(jet_xx[jj], 0);

    int RK_it = 0, last_RK_it = 0;

    MY_FLOAT normal[JJ];
    for (jj=0; jj<JJ; jj++) normal[jj]=0.0;
    normal[SEC_INDEX] =1.0;

    MY_FLOAT out[JJ], xx[JJ];
    double tau[CC-1], tau_denom;

    do{

        if (FP_it>0){

            for (jj=0; jj<JJ; jj++) {
                printf("%s: ", var_names[jj]);
                for (cc=0; cc<CC; cc++){ 
                    printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
                }
                printf("\n");
            }
            printf("--- Projected to ---->\n");

            for (jj=0; jj<JJ; jj++){ xx[jj]=MY_JET_DATA(jet_xx[jj], 0);}
            FLOAT_function(out, t, xx);

            for (cc=1; cc<CC; cc++){ tau[cc-1]=0;}
            tau_denom = 0;
            for (jj=0; jj<JJ; jj++) {
                for (cc=1; cc<CC; cc++){
                    tau[cc-1]+=MY_JET_DATA(jet_xx[jj], cc)*normal[jj];
                }
                tau_denom+=out[jj]*normal[jj];
            }
            for (cc=0; cc<CC-1; cc++){ tau[cc]/=-tau_denom;}

            for (jj=0; jj<JJ; jj++){
                for (cc=1; cc<CC; cc++){
                    MY_JET_DATA(jet_xx[jj], cc)+=out[jj]*tau[cc-1];
                }
            }

            for (jj=0; jj<JJ; jj++) {
                printf("%s: ", var_names[jj]);
                for (cc=0; cc<CC; cc++){ 
                    printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
                }
                printf("\n");
            }
            printf("\n");

            FP_step_Newt(FP, jet_xx);

        }

        dd=1;
        for (jj=0; jj<JJ; jj++){
            MY_JET_DATA(jet_xx[jj], 0) = FP[jj];
            if (jj==SEC_INDEX){
                for (cc=1; cc<CC; cc++){
                    MY_JET_DATA(jet_xx[jj], cc)=0.0;
                }
            } else {
                for (cc=1; cc<CC; cc++){
                    if (cc==dd){ MY_JET_DATA(jet_xx[jj], cc)=1.0;}
                    else{ MY_JET_DATA(jet_xx[jj], cc)=0.0;}
                }
                dd++;
            }        

        }

        double rk_error, rk_TOL, h_min=1000, h_max=0;
        int AS=0, RS=0, last_rejected=0, ind_error_TOL; 

        int last_sign=5, new_sign, sign_change_count=0;

        //Set initial and final time and step size
        t = 0.0; t_F = 360.0;
        step_size= H0;

        //Print initial values
        // printf("t: %f\n", t);
        // fprintf(ORBIT_temp,"%lf ", t);
        for (jj=0; jj<JJ; jj++) {
            printf("%s: ", var_names[jj]);
            // fprintf(ORBIT_temp, "%le ", MY_JET_DATA(jet_xx[jj],0));
            for (cc=0; cc<CC; cc++){ 
                printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
            }
            printf("\n");
        }

        while(t<t_F){

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
                // printf("REJECTED h=%le", step_size);
                RS++;
                // fprintf(h_temp, "%lf %lf %d\n", t, step_size, 0);
                step_size *= fmin(1, new_step_size);

                // printf("--> NEW h=%le\n", step_size);
                continue;
            }

            for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(tmp_2[jj], cc); }}

            t+=step_size;

            new_sign = get_sign(MY_JET_DATA(jet_xx[SEC_INDEX], 0)-SEC_CONST);

            if (new_sign!=last_sign && last_sign!=5) {
                sign_change_count++;
                printf(" --> crossed Sigma {%s=%lf} at t=%lf\n", var_names[SEC_INDEX], SEC_CONST, t);
                if (sign_change_count>1 && t>SEC_SAFE_MIN) { RK_it++; break; }
            }

            last_sign = new_sign;

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

            // // //Print new values
            // printf("\nt: %f\n", t);
            // fprintf(ORBIT_temp,"\n%lf ", t);
            // for (jj=0; jj<JJ; jj++) {
                // fprintf(ORBIT_temp, "%le ", MY_JET_DATA(jet_xx[jj],0));
            //     printf("%s: ", var_names[jj]);
            //     for (cc=0; cc<CC; cc++){ 
            //         printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
            //     }
            //     printf("\n");
            // }
            
            //Check if last step
            if (t+step_size>t_F){
                step_size = t_F-t;
            } else {
                if (step_size>h_max) h_max=step_size;
                if (step_size<h_min) h_min=step_size;
            }

            RK_it++;

        }

        printf("First aproximation of the period: T=%f\n --> with %s=%lf\n", t, var_names[SEC_INDEX], MY_JET_DATA(jet_xx[SEC_INDEX], 0));

        int TIME_newt_it=0;
        MY_FLOAT T, sec_val = MY_JET_DATA(jet_xx[SEC_INDEX], 0);

        while (fabs(sec_val-SEC_CONST)>TIME_NEWT_TOL) {

            // printf("It=%d, sec_val=%le, T=%le\n", TIME_newt_it, sec_val, t);

            TIME_step_Newt(&T, t, jet_xx);

            step_size = T-t;
            RK_Implicit(tmp, step_size, t, jet_xx);
            for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(tmp[jj], cc); }}

            sec_val = MY_JET_DATA(jet_xx[SEC_INDEX], 0);
            t=T;

            TIME_newt_it++;

            if (TIME_newt_it >= TIME_NEWT_MAX_IT){
                printf("Reached %d iterations of temporal Newton. %s=%.15lf\nContinue? (y/n)\n", TIME_newt_it, var_names[SEC_INDEX], MY_JET_DATA(jet_xx[SEC_INDEX], 0));
                scanf(" %c", &it_continue);
                if (it_continue=='y') TIME_NEWT_MAX_IT*=2;
                else break;
            }

        }

        FP_norm = 0;

        printf("Aproximation of the period after Newton on time (%d it.): T=%.15lf\n --> with %s=%.15lf\n", TIME_newt_it, t, var_names[SEC_INDEX], MY_JET_DATA(jet_xx[SEC_INDEX], 0));
        printf("With ");
        for (jj=0; jj<JJ; jj++){
            if (jj!=SEC_INDEX){
                printf("%s=%.15lf ", var_names[jj], FP[jj]);
                FP_norm+=pow(MY_JET_DATA(jet_xx[jj], 0)-FP[jj], 2.0);
            }
        }

        FP_norm = sqrt(FP_norm);
        
        printf("(s.t. ||P(*)-*||=%le)\n\n",FP_norm);
        
        last_RK_it = RK_it;

        FP_it++;

        if (FP_it >= FP_NEWT_MAX_IT){
            printf("Reached %d iterations of fixed point Newton. P(*)-*=%le\nContinue? (y/n)\n", FP_it, FP_norm);
            scanf(" %c", &it_continue);
            if (it_continue=='y') FP_NEWT_MAX_IT*=2;
            else break;
        }

    } while(FP_norm>FP_NEWT_TOL);

    // // Print last values
    // printf("\nt: %.15le\n", t);
    // for (jj=0; jj<JJ; jj++) {
    //     printf("%s: ", var_names[jj]);
    //     for (cc=0; cc<CC; cc++){ 
    //         printf("%.15le   \t", MY_JET_DATA(jet_xx[jj],cc));
    //     }
    //     printf("\n");
    // }


    // printf("\nAccepted=%d, Rejected=%d\nh_min=%le, h_max=%le\n", AS, RS, h_min, h_max);

    // for (i=0; i < numCommandsForGnuplot; i++){
    //     fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); 
    // }

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    // fprintf(ORBIT_gnuplotPipe, "plot '%s' using 1:2 w points ps 0.5 title '$y_%d$'", orbit_file_dat, 1);
    // for (jj=2; jj<JJ+1; jj++) fprintf(ORBIT_gnuplotPipe, ", '%s' using 1:%d w points ps 0.5 title '$y_%d$'", orbit_file_dat, jj+1, jj); 
    // fprintf(ORBIT_gnuplotPipe, "\nunset output\n"); 

    return 0;
}