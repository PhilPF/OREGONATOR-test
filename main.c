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

#define RK_RTOL 1E-10
#define RK_ATOL RK_RTOL
#define H0 1E-7

MY_FLOAT s=63.2, w=0.161, q=1.25e-5, fff=1.0;

void spectrum(double WR[JJ], double WI[JJ], double VR[JJ*JJ], MY_JET *jet_xx){

    int jj, cc;

    double MAT[JJ*JJ];

    // printf("Jac: \n");
    for (jj=0; jj<JJ; jj++){
        for (cc=0; cc<JJ; cc++){
            MAT[cc+jj*JJ]=MY_JET_DATA(jet_xx[jj], 1+cc);
            // printf("%le ", MAT[cc+jj*JJ]);
        }
        // printf("\n");
    }
    // printf("\n");

    lapack_int info;

    if (VR==NULL){
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'N', JJ, MAT, JJ, WR, WI, NULL, 1, NULL, 1);
    }else {
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'V', JJ, MAT, JJ, WR, WI, NULL, 1, VR, JJ);
    }
    if (info != 0) exit(0);

}

void plotPseudospectrum(double complex *pspec_grid, double *pspec, double step_size, MY_JET *jet_xx, double *WR, double *WI, int pspec_N){

    int i, j, jj, cc;

    float grid_expansion = 1e+3;

    double min_grid_real=WR[0], max_grid_real=WR[0], min_grid_imag=WI[0], max_grid_imag=WI[0];
    double grid_size, midpoint_real=WR[0], midpoint_imag=WI[0];

    for (jj=1; jj<JJ; jj++){
        if (WR[jj]<min_grid_real) min_grid_real = WR[jj];
        if (WR[jj]>max_grid_real) max_grid_real = WR[jj];

        if (WI[jj]<min_grid_imag) min_grid_imag = WI[jj];
        if (WI[jj]>max_grid_imag) max_grid_imag = WI[jj];
    }

    midpoint_real=step_size*(min_grid_real+max_grid_real)/2.0;
    midpoint_imag=step_size*(min_grid_imag+max_grid_imag)/2.0;

    grid_size = grid_expansion*1.2*step_size*fmax(max_grid_real-min_grid_real,max_grid_imag-min_grid_imag);

    double linspace = grid_size/(double)pspec_N;

    printf("grid_size=%le, midpoint_real=%le, midpoint_imag=%le\n", grid_size, midpoint_real, midpoint_imag);

    float SV[JJ];
    lapack_complex_float MAT[JJ*JJ];

    for (i=0; i<pspec_N; i++){

        for (j=0; j<pspec_N; j++){

            pspec_grid[i*pspec_N+j] = (midpoint_real-grid_size/2.0+i*linspace) + I*(midpoint_imag-grid_size/2.0+j*linspace);

            for (jj=0; jj<JJ; jj++){
                for (cc=0; cc<JJ; cc++){
                    MAT[cc+jj*JJ]=-step_size*MY_JET_DATA(jet_xx[jj], 1+cc);
                    if (jj==cc) MAT[cc+jj*JJ]+=pspec_grid[i*pspec_N+j];
                }
            }

            lapack_int info;
            float *work = (float*)malloc( sizeof(float) ); 
            info = LAPACKE_cgesvd(LAPACK_ROW_MAJOR,'N', 'N', JJ, JJ, MAT, JJ, SV, NULL, JJ, NULL, JJ, work);
            if (info != 0) exit(0);

            pspec[i*pspec_N+j] = (double)SV[JJ-1];

        }        

    }

}


int main(int argc, char *argv[]){

    clock_t CLOCK_begin = clock();

    int jj, ss, dd, cc, i;

    MY_FLOAT step_size, last_step_size, new_step_size, t, t_F;

    char orbit_file_tex[20], orbit_file_dat[20];
    char orbit_folder_name[20] = "ORBIT/";
    snprintf(orbit_file_tex, 100, "set output '%sorbit.tex'", orbit_folder_name);
    snprintf(orbit_file_dat, 100, "%sorbit.dat", orbit_folder_name);

    const int ORBIT_numCommandsForGnuplot=8;
    char * ORBIT_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", "set format y '$$%.0e$$'", orbit_file_tex, "set key reverse Left left", "set logscale y", "set xlabel '$t$'"}; //For vdpol_1_1
    FILE * ORBIT_temp = fopen(orbit_file_dat, "w");
    FILE * ORBIT_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < ORBIT_numCommandsForGnuplot; i++){
        fprintf(ORBIT_gnuplotPipe, "%s \n", ORBIT_commandsForGnuplot[i]); 
    }

    char pseudospec_file_tex[60], pseudospec_file_dat[60];
    char pseudospec_folder_name[60] = "ORBIT PSEUDOSPEC/";
    snprintf(pseudospec_file_tex, 200, "set output '%spseudospec.tex'", pseudospec_folder_name);
    snprintf(pseudospec_file_dat, 200, "%spseudospec.dat", pseudospec_folder_name);

    const int PSEUDOSPEC_numCommandsForGnuplot=8;
    char * PSEUDOSPEC_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", pseudospec_file_tex, "unset clabel", "set contour base", "set cntrparam bspline level discrete -1,-1.5,-2,-2.5,-3,-3.5,-4,-4.5,-5,-5.5,-6,-6.5", "set view map"}; //For vdpol_1_1
    FILE * PSEUDOSPEC_temp = fopen(pseudospec_file_dat, "w");
    FILE * PSEUDOSPEC_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < PSEUDOSPEC_numCommandsForGnuplot; i++){
        fprintf(PSEUDOSPEC_gnuplotPipe, "%s \n", PSEUDOSPEC_commandsForGnuplot[i]); 
    }

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

    MY_JET_DATA(jet_xx[0], 0) = 1.998885040464527;
    MY_JET_DATA(jet_xx[1], 0) = 2.0;
    MY_JET_DATA(jet_xx[2], 0) = 1.731558539767741;

    for (jj=0; jj<JJ; jj++){
        for (cc=1; cc<CC; cc++){
            if (jj+1==cc) MY_JET_DATA(jet_xx[jj], cc) = 1.0;
            else MY_JET_DATA(jet_xx[jj], cc) = 0.0;
        }
    }

    //Set initial and final time and step size
    t = 0.0; t_F = 243.273501876619463;
    step_size= H0; 

    //Print initial values
    // printf("t: %f\n", t);
    fprintf(ORBIT_temp,"%lf ", t);
    for (jj=0; jj<JJ; jj++) {
        printf("%s: ", var_names[jj]);
        fprintf(ORBIT_temp, "%le ", MY_JET_DATA(jet_xx[jj],0));
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
        fprintf(ORBIT_temp,"\n%lf ", t);
        for (jj=0; jj<JJ; jj++) {
            fprintf(ORBIT_temp, "%le ", MY_JET_DATA(jet_xx[jj],0));
        //     printf("%s: ", var_names[jj]);
        //     for (cc=0; cc<CC; cc++){ 
        //         printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
        //     }
        //     printf("\n");
        }
        
        //Check if last step
        if (t+step_size>t_F){
            last_step_size = step_size;
            step_size = t_F-t;
        } else {
            if (step_size>h_max) h_max=step_size;
            if (step_size<h_min) h_min=step_size;
        }

    }

    // Print last values
    printf("\nt: %.15le\n", t);
    for (jj=0; jj<JJ; jj++) {
        printf("%s: ", var_names[jj]);
        for (cc=0; cc<CC; cc++){ 
            printf("%.15le   \t", MY_JET_DATA(jet_xx[jj],cc));
        }
        printf("\n");
    }

    printf("\nAccepted=%d, Rejected=%d\nh_min=%le, h_max=%le\n", AS, RS, h_min, h_max);

    // for (i=0; i < numCommandsForGnuplot; i++){
    //     fprintf(gnuplotPipe, "%s \n", commandsForGnuplot[i]); 
    // }

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    fprintf(ORBIT_gnuplotPipe, "plot '%s' using 1:2 w points ps 0.5 title '$y_%d$'", orbit_file_dat, 1);
    for (jj=2; jj<JJ+1; jj++) fprintf(ORBIT_gnuplotPipe, ", '%s' using 1:%d w points ps 0.5 title '$y_%d$'", orbit_file_dat, jj+1, jj); 
    fprintf(ORBIT_gnuplotPipe, "\nunset output\n"); 

    double WR[JJ], WI[JJ];

    spectrum(WR, WI, NULL, jet_xx);

    for (jj=0; jj<JJ; jj++) printf("lambda_%d = %le + i %le\n", jj+1, WR[jj], WI[jj]);

    static const int pspec_N=500;
    double complex pspec_grid[pspec_N*pspec_N];
    double pspec[pspec_N*pspec_N];

    plotPseudospectrum(pspec_grid, pspec, last_step_size, jet_xx, WR, WI, pspec_N);

    for (i=0; i<pspec_N; i++){
        for (jj=0; jj<pspec_N; jj++){
            fprintf(PSEUDOSPEC_temp, "%le %le %le\n", crealf(pspec_grid[i*pspec_N+jj]), cimagf(pspec_grid[i*pspec_N+jj]), log10(pspec[i*pspec_N+jj]));
        }
        fprintf(PSEUDOSPEC_temp, "\n");
    }

    fprintf(PSEUDOSPEC_gnuplotPipe, "splot '%s' using 1:2:3 w lines dt 2 lc rgb 'black' nosurface notitle", pseudospec_file_dat);
    fprintf(PSEUDOSPEC_gnuplotPipe, "\nunset output\n"); 


    return 0;
}