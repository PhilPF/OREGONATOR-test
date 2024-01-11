#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <time.h>
#include <string.h>
#include <complex.h>
#include "taylor.h"


#define RKS 4 // num. of stages of RK method
#define RKP (double) 4
#define _2RKP pow(0.5, RKP)
#define h_fac pow(0.38, 1.0/(RKP+1.0))

#define RK_RTOL 1E-10
#define RK_ATOL RK_RTOL
#define H0 1E-7

#define JJ _NUMBER_OF_JET_VARS_
#define CC _JET_COEFFICIENTS_COUNT_TOTAL_
#define DD _MAX_DEGREE_OF_JET_VARS_

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923

int init_flag_FLOAT_function=1, init_flag_RK_Explicit=1;

MY_FLOAT s=63.2, w=0.161, q=8.375e-6, fff=1;
static const int param_config=0;


void function(MY_JET *jet_out, MY_FLOAT t, MY_JET *jet_xx){

    int jj, cc;

    MY_JET **function_jet;
    MY_FLOAT temporal_state_xx[JJ], **coeff;

    coeff = taylor_coefficients_taylor_A(t, temporal_state_xx, 0, 1, jet_xx, &function_jet);

    for (jj=0; jj<JJ; jj++) {
        for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_out[jj], cc) = MY_JET_DATA(function_jet[jj][1],cc);}
    }

}

void FLOAT_function(MY_FLOAT *out, MY_FLOAT t, MY_FLOAT *xx){

    int jj, cc;

    static MY_JET **function_jet, jet_xx[JJ];
    MY_FLOAT temporal_state_xx[JJ], **coeff;

    if (init_flag_FLOAT_function){
        for(jj=0; jj<JJ; jj++){
            InitJet(jet_xx[jj]);
        }
        init_flag_FLOAT_function=0;
    }

    for(jj=0; jj<JJ; jj++){
        MY_JET_DATA(jet_xx[jj], 0) = xx[jj];
        for (cc=1; cc<CC; cc++) MY_JET_DATA(jet_xx[jj], cc)=0.0;
    }

    coeff = taylor_coefficients_taylor_A(t, temporal_state_xx, 0, 1, jet_xx, &function_jet);

    for (jj=0; jj<JJ; jj++) {
        out[jj] = MY_JET_DATA(function_jet[jj][1], 0); 
    }
 
}

void RK_Explicit(MY_JET *jet_out, MY_FLOAT A[RKS][RKS], MY_FLOAT *b, MY_FLOAT *c, MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx){

    int rks, jj, cc;

    int tjj,j;    

    static MY_JET temporalJet[3][JJ];
    static MY_JET stages[RKS][JJ], internal_sum[JJ];
    if (init_flag_RK_Explicit){
        for(jj=0; jj<JJ; jj++){
            for (tjj=0; tjj<3; tjj++) {InitJet(temporalJet[tjj][jj]);}
            for (rks=0; rks<RKS; rks++) {InitJet(stages[rks][jj]);}
            InitJet(internal_sum[jj]);
        }
        init_flag_RK_Explicit = 0;
    }

    function(stages[0], t, jet_xx);

    if(RKS%2==0) {
        for (jj=0; jj<JJ; jj++){ MultiplyFloatJetA(jet_out[jj], step_size*b[0], stages[0][jj]);}
    } else {
        for (jj=0; jj<JJ; jj++){ MultiplyFloatJetA(temporalJet[0][jj], step_size*b[0], stages[0][jj]);}
    }
   
    for (rks=1; rks<RKS; rks++){

        for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++) {
            MY_JET_DATA(temporalJet[1][jj], cc)=0; 
            MY_JET_DATA(internal_sum[jj], cc)=0;
        }}
        
        for (j=rks-1; j>=0; j--) {
            for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[2][jj], step_size*A[rks][j], stages[j][jj]);
            if (j%2==0) {
                for(jj=0; jj<JJ; jj++) AddJetJetA(internal_sum[jj], temporalJet[1][jj], temporalJet[2][jj]);
            } else {
                for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[1][jj], internal_sum[jj], temporalJet[2][jj]);
            }
        }

        for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[2][jj], jet_xx[jj], internal_sum[jj]);
        function(stages[rks], t+step_size*c[rks], temporalJet[2]); 
        
        for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[2][jj], step_size*b[rks], stages[rks][jj]);

        if((rks+RKS)%2==0) {
            for(jj=0; jj<JJ; jj++) AddJetJetA(jet_out[jj], temporalJet[0][jj], temporalJet[2][jj]);
        } else {
            for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[0][jj], jet_out[jj], temporalJet[2][jj]);
        }
    }

    for(jj=0; jj<JJ; jj++) AddJetJetA(jet_out[jj], jet_xx[jj], temporalJet[0][jj]);

}

// void RK_Explicit(MY_JET *jet_out, MY_FLOAT A[RKS][RKS], MY_FLOAT *b, MY_FLOAT *c, MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx){

//     int jj, cc;

//     int TJJ=3;
//     int tjj,i,j;    

//     MY_JET temporalJet[TJJ][JJ];
//     MY_JET stages[RKS][JJ], sumStages[JJ], sumOutStages[JJ];
//     for(jj=0; jj<JJ; jj++){
//         for (tjj=0; tjj<TJJ; tjj++) {InitJet(temporalJet[tjj][jj]);}
//         for (i=0; i<RKS; i++) {InitJet(stages[i][jj]);}
//         InitJet(sumStages[jj]); InitJet(sumOutStages[jj]);
//     }

//     for(jj=0; jj<JJ; jj++) for (cc=0; cc<CC; cc++) MY_JET_DATA(sumStages[jj], cc)=0.0;

//     MY_FLOAT sumC;  
//     sumC=0.0;

//     function(stages[0], t, jet_xx);

//     if(RKS%2==0) {
//         for (jj=0; jj<JJ; jj++){ MultiplyFloatJetA(temporalJet[2][jj], step_size*b[0], stages[0][jj]);}
//     } else {
//         for (jj=0; jj<JJ; jj++){ MultiplyFloatJetA(sumOutStages[jj], step_size*b[0], stages[0][jj]);}
//     }
   
//     sumC = 0.0;

//     for (i=1; i<RKS; i++){

//         sumC += c[i];

//         for(jj=0; jj<JJ; jj++) for (cc=0; cc<CC; cc++) MY_JET_DATA(sumStages[jj], cc)=0.0;     
        
//         if(i%2==1) {
//             for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(sumStages[jj], step_size*A[i][0], stages[0][jj]);
//         } else {
//             for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[1][jj], step_size*A[i][0], stages[0][jj]);
//         }
        
//         for (j=1; j<i; j++) {
//             for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[0][jj], step_size*A[i][j], stages[j][jj]);
//             if ((i+j)%2==0) {
//                 for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[1][jj], sumStages[jj], temporalJet[0][jj]);
//             } else {
//                 for(jj=0; jj<JJ; jj++) AddJetJetA(sumStages[jj], temporalJet[1][jj], temporalJet[0][jj]);
//             }
//         }

//         for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[0][jj], jet_xx[jj], sumStages[jj]);
//         function(stages[i], t+step_size*sumC, temporalJet[0]); 
        
//         for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[0][jj], step_size*b[i], stages[i][jj]);

//         if((i+RKS)%2==0) {
//             for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[0][jj], sumOutStages[jj], temporalJet[0][jj]);
//         } else {
//             for(jj=0; jj<JJ; jj++) AddJetJetA(sumOutStages[jj], temporalJet[2][jj], temporalJet[0][jj]);
//         }
//     }
    
//     for(jj=0; jj<JJ; jj++) AddJetJetA(jet_out[jj], jet_xx[jj], sumOutStages[jj]);

// }

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
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'N', JJ, MAT, JJ, WR, WI, NULL, 1, NULL, JJ);
    }else {
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'V', JJ, MAT, JJ, WR, WI, NULL, 1, VR, JJ);
    }
    if (info != 0) exit(0);

}

void plotStabilityRegion(int SR_N, double complex *SR){

    int i,j;

    static const int coeff_N = 5;
    double complex coeffs[] = {1/24.0+ 0*I,1/6.0+ 0*I,1/2.0+ 0*I,1+ 0*I,1+ 0*I};

    double complex w;
    double linspace = 1/(double)SR_N;

    double complex P, dP;

    SR[0]=0+0*I;

    for(i=0; i<SR_N+1; i++){
        w = cexp(I*2*(coeff_N-1)*PI*(i*linspace));
        coeffs[coeff_N-1] = 1-w;

        if (i>0) SR[i] = SR[i-1];

        do{

            P = coeffs[0];
            dP = ((double)coeff_N-1)*coeffs[0];
            for (j=1; j<coeff_N-1; j++){
                P = coeffs[j] + P*SR[i];
                dP = ((double)coeff_N-j-1)*coeffs[j] + dP*SR[i];
            }                
            P = coeffs[coeff_N-1] + P*SR[i];

            SR[i] -= P/dP;

        }while(cabs(P)>1e-10);

    }

}

void plotPseudospectrum(double complex *pspec_grid, double *pspec, double step_size, MY_JET *jet_xx, double *WR, double *WI, int pspec_N){

    int i, j, jj, cc;

    float grid_expansion = 1.0;

    double min_grid_real=WR[0], max_grid_real=WR[0], min_grid_imag=WI[0], max_grid_imag=WI[0], margin_grid_real, margin_grid_imag;
    double min_grid, max_grid, grid_size, midpoint_real=WR[0], midpoint_imag=WI[0];

    for (jj=1; jj<JJ; jj++){
        if (WR[jj]<min_grid_real) min_grid_real = WR[jj];
        else if (WR[jj]>max_grid_real) max_grid_real = WR[jj];

        if (WI[jj]<min_grid_imag) min_grid_imag = WI[jj];
        else if (WI[jj]>max_grid_imag) max_grid_imag = WI[jj];

    }

    midpoint_real=step_size*(min_grid_real+max_grid_real)/2.0;
    midpoint_imag=step_size*(min_grid_imag+max_grid_imag)/2.0;

    // if (min_grid>-3) min_grid=-3;
    // if (max_grid<3)  max_grid=3;

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
   
    int jj, ss, dd, cc;

    int i,j;

    taylor_initialize_jet_library();

    MY_FLOAT step_size, new_step_size, t, t_F;

    char ssc_file_tex[20], ssc_file_dat[20];
    char ssc_folder_name[20] = "SSC/";
    snprintf(ssc_file_tex, 100, "set output '%sssc_%d.tex'", ssc_folder_name, param_config);
    snprintf(ssc_file_dat, 100, "%sssc_%d.dat", ssc_folder_name, param_config);

    char sr_file_tex[20], sr_file_dat[20];
    char sr_folder_name[20] = "SR/";
    snprintf(sr_file_tex, 100, "set output '%ssr_%d.tex'", sr_folder_name, param_config);
    snprintf(sr_file_dat, 100, "%ssr_%d.dat", sr_folder_name, param_config);

    char sccxsr_file_tex[40], sccxsr_file_dat[40];
    char sccxsr_folder_name[40] = "SSCxSR/";
    snprintf(sccxsr_file_tex, 100, "set output '%ssccxsr_%d.tex'", sccxsr_folder_name, param_config);
    snprintf(sccxsr_file_dat, 100, "%ssccxsr_%d.dat", sccxsr_folder_name, param_config);

    char spec_file_tex[40], spec_file_dat[40];
    char spec_folder_name[40] = "SPEC/";
    snprintf(spec_file_tex, 100, "set output '%sspec_%d.tex'", spec_folder_name, param_config);
    snprintf(spec_file_dat, 100, "%sspec_%d.dat", spec_folder_name, param_config);

    char pseudospec_file_tex[60], pseudospec_file_dat[60];
    char pseudospec_folder_name[60] = "PSEUDOSPEC/";
    snprintf(pseudospec_file_tex, 200, "set output '%spseudospec_%d.tex'", pseudospec_folder_name, param_config);
    snprintf(pseudospec_file_dat, 200, "%spseudospec_%d.dat", pseudospec_folder_name, param_config);

    const int SSC_numCommandsForGnuplot=9;
    char * SSC_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", "set format y '$$%.0e$$'", ssc_file_tex, "set key reverse Left left", "set logscale y", "set xlabel '$t$'", "set ylabel '$h$' rotate by 0"}; //For vdpol_1_1
    FILE * SSC_temp = fopen(ssc_file_dat, "w");
    FILE * SSC_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < SSC_numCommandsForGnuplot; i++){
        fprintf(SSC_gnuplotPipe, "%s \n", SSC_commandsForGnuplot[i]); 
    }

    const int SR_numCommandsForGnuplot=9;
    char * SR_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", "set format y '$$%.0e$$'", sr_file_tex, "set key reverse Left left",  "set logscale y", "set xlabel '$t$'", "set ylabel '$\\rho(J)$' rotate by 0"}; //For vdpol_1_1
    FILE * SR_temp = fopen(sr_file_dat, "w");
    FILE * SR_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < SR_numCommandsForGnuplot; i++){
        fprintf(SR_gnuplotPipe, "%s \n", SR_commandsForGnuplot[i]); 
    }

    const int SSCxSR_numCommandsForGnuplot=9;
    char * SSCxSR_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", "set format y '$$%.0e$$'", sccxsr_file_tex, "set key reverse Left left",  "set logscale y", "set xlabel '$t$'", "set ylabel '$\\rho(J)\\cdot h$' rotate by 0"}; //For vdpol_1_1
    FILE * SSCxSR_temp = fopen(sccxsr_file_dat, "w");
    FILE * SSCxSR_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < SSCxSR_numCommandsForGnuplot; i++){
        fprintf(SSCxSR_gnuplotPipe, "%s \n", SSCxSR_commandsForGnuplot[i]); 
    }

    const int SPEC_numCommandsForGnuplot=5;
    char * SPEC_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", spec_file_tex, "set key reverse Left left"}; //For vdpol_1_1
    FILE * SPEC_temp = fopen(spec_file_dat, "w");
    FILE * SPEC_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < SPEC_numCommandsForGnuplot; i++){
        fprintf(SPEC_gnuplotPipe, "%s \n", SPEC_commandsForGnuplot[i]); 
    }

    const int PSEUDOSPEC_numCommandsForGnuplot=8;
    char * PSEUDOSPEC_commandsForGnuplot[] = {"set terminal epslatex standalone color font ',8'", "set size square", "set format '$$%g$$'", pseudospec_file_tex, "unset clabel", "set contour base", "set cntrparam bspline level discrete -1,-1.5,-2,-2.5,-3,-3.5,-4,-4.5,-5,-5.5,-6,-6.5", "set view map"}; //For vdpol_1_1
    FILE * PSEUDOSPEC_temp = fopen(pseudospec_file_dat, "w");
    FILE * PSEUDOSPEC_gnuplotPipe = popen ("gnuplot -persistent", "w");
    for (i=0; i < PSEUDOSPEC_numCommandsForGnuplot; i++){
        fprintf(PSEUDOSPEC_gnuplotPipe, "%s \n", PSEUDOSPEC_commandsForGnuplot[i]); 
    }

    static const int stabilityRegion_N=100;
    double complex stabilityRegion[stabilityRegion_N+1];
    plotStabilityRegion(stabilityRegion_N, stabilityRegion);

    for (i=0; i<stabilityRegion_N+1; i++){
        fprintf(SPEC_temp, "%le %le %d\n", crealf(stabilityRegion[i]), cimagf(stabilityRegion[i]), 0);
    }

    double minimum_spec=1e50, minimizing_t, minimizing_ssc, minimizing_WR[JJ], minimizing_WI[JJ];
    int minimizing_jj;
    MY_JET minimizing_jet_xx[JJ]; 
    for(jj=0; jj<JJ; jj++){ InitJet(minimizing_jet_xx[jj]);}

    double spectral_radius, temp_spectral_radius;
    double WR[JJ], WI[JJ];

    const char **var_names = taylor_get_variable_names();
    const char **monomials = taylor_get_jet_monomials();

    MY_FLOAT _1_6 = 0.16666666666666666666, _1_3 = 0.33333333333333333333;
    MY_FLOAT A[RKS][RKS]={{0,0,0,0}, {0.5,0,0,0}, {0,0.5,0,0}, {0,0,1.0,0}}, b[RKS]={_1_6,_1_3,_1_3,_1_6}, c[RKS]={0,0.5,0.5,1.0}; //RK4 RKS=4

    //Print first line with the monomials of the jet
    printf("\t");
    for (cc=0; cc<CC-1; cc++) { printf("\t\t%s", monomials[cc]);}
    printf("\n");

    static MY_JET jet_xx[JJ], tmp[JJ], tmp_1[JJ], tmp_2[JJ];
    for (jj=0; jj<JJ; jj++) {InitJet(jet_xx[jj]); InitJet(tmp[jj]); InitJet(tmp_1[jj]); InitJet(tmp_2[jj]);}

    double rk_error, rk_TOL, h_min=1000, h_max=0;
    int AS=0, RS=0, last_rejected=0, ind_error_TOL; 

    MY_JET_DATA(jet_xx[0], 0) = 1.998901480433274;
    MY_JET_DATA(jet_xx[1], 0) = 2;
    MY_JET_DATA(jet_xx[2], 0) = 1.731567488681221;

    for (jj=0; jj<JJ; jj++){
        for (cc=1; cc<CC; cc++){
            if (jj+1==cc) MY_JET_DATA(jet_xx[jj], cc) = 1.0;
            else MY_JET_DATA(jet_xx[jj], cc) = 0.0;
        }
    }

    //Set initial and final time and step size
    t = 0.0; t_F = 255.903421559331122;
    step_size= H0;

    double stiff_kappa=0, stiff_gamma=0, stiff_norm, stiff_last_norm, stiff_T=t_F-t, stiff_nu=0, stiff_VR[JJ*JJ];
    int stiff_VR_jj;

    function(tmp, t, jet_xx);
    spectrum(WR, WI, stiff_VR, tmp);
    spectral_radius=0;
    for (jj=0; jj<JJ; jj++){
        temp_spectral_radius=WR[jj]*WR[jj]+WI[jj]*WI[jj];
        if (temp_spectral_radius>spectral_radius) {
            spectral_radius = temp_spectral_radius;
            stiff_VR_jj=jj;
        }
    }

    for (jj=0; jj<JJ; jj++) stiff_nu+=pow(MY_JET_DATA(jet_xx[jj], 0), 2);
    stiff_nu = sqrt(stiff_nu);
    stiff_last_norm = stiff_nu;

    //Print initial values
    printf("t: %f\n", t);
    // fprintf(temp,"%lf ", t);
    for (jj=0; jj<JJ; jj++) {
        printf("%s: ", var_names[jj]);
        for (cc=0; cc<CC; cc++){ 
            printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
            // fprintf(temp, "%le ", MY_JET_DATA(jet_xx[jj],cc));
        }
        printf("\n");
    }

    while(t<t_F){

        //Compute two steps of stepsize h/2
        RK_Explicit(tmp_1, A, b, c, step_size/2, t, jet_xx);
        RK_Explicit(tmp_2, A, b, c, step_size/2, t+step_size/2, tmp_1);

        //Compute one step of stepsize h
        RK_Explicit(tmp, A, b, c, step_size, t, jet_xx);      
        
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
            // printf("REJECTED h=%le\n", step_size);
            RS++;
            fprintf(SSC_temp, "%lf %le %d\n", t, step_size, 0);
            step_size *= fmin(1, new_step_size);

            // printf("--> NEW h=%le\n", step_size);
            continue;
        }

        for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(tmp_2[jj], cc); }}

        t+=step_size;

        spectrum(WR, WI, NULL, jet_xx);
        
        spectral_radius=0;
        for (jj=0; jj<JJ; jj++){
            temp_spectral_radius=WR[jj]*WR[jj]+WI[jj]*WI[jj];
            if (temp_spectral_radius>spectral_radius) spectral_radius = temp_spectral_radius;
        }
        spectral_radius = sqrt(spectral_radius);

        fprintf(SR_temp, "%lf %le\n", t, spectral_radius);

        fprintf(SSCxSR_temp, "%lf %le\n", t, step_size*spectral_radius);

        for (jj=0; jj<JJ; jj++) { 
            fprintf(SPEC_temp, "%le %le %d\n", WR[jj]*step_size, WI[jj]*step_size, jj+1); 

            if (WR[jj]*step_size<minimum_spec){
                minimum_spec = WR[jj]*step_size;
                minimizing_t   = t;
                minimizing_ssc = step_size;
                minimizing_jj = jj;
                for (dd=0; dd<JJ; dd++) {
                    minimizing_WR[dd]  = WR[dd];
                    minimizing_WI[dd]  = WI[dd];
                    for (cc=0; cc<CC; cc++){
                        MY_JET_DATA(minimizing_jet_xx[dd], cc) = MY_JET_DATA(jet_xx[dd], cc);
                    }
                }
                
            } 

            stiff_norm+=pow(fabs(MY_JET_DATA(jet_xx[jj], 0)), 2);

        }

        stiff_norm=sqrt(stiff_norm);
        if (stiff_norm>stiff_kappa) stiff_kappa=stiff_norm;

        stiff_gamma+=step_size*fmax(stiff_norm, stiff_last_norm);

        stiff_last_norm=stiff_norm;

        AS++; 
        fprintf(SSC_temp, "%lf %le %d\n", t, step_size, 1);
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
        // // fprintf(temp,"\n%lf ", t);
        // for (jj=0; jj<JJ; jj++) {
        //     printf("%s: ", var_names[jj]);
        //     for (cc=0; cc<CC; cc++){ 
        //         printf("%le \t", MY_JET_DATA(jet_xx[jj],cc));
        //         // fprintf(temp, "%le ", MY_JET_DATA(jet_xx[jj],cc));
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

    static const int pspec_N=500;
    double complex pspec_grid[pspec_N*pspec_N];
    double pspec[pspec_N*pspec_N];

    plotPseudospectrum(pspec_grid, pspec, minimizing_ssc, minimizing_jet_xx, minimizing_WR, minimizing_WI, pspec_N);

    for (i=0; i<pspec_N; i++){
        for (j=0; j<pspec_N; j++){
            fprintf(PSEUDOSPEC_temp, "%le %le %le\n", crealf(pspec_grid[i*pspec_N+j]), cimagf(pspec_grid[i*pspec_N+j]), log10(pspec[i*pspec_N+j]));
        }
        fprintf(PSEUDOSPEC_temp, "\n");
    }
    
    fprintf(SSC_gnuplotPipe, "plot '%s' using 1:($3 == 1 ? $2 : 1/0) w linespoints ps 0.5 lc rgb 'black' title 'Accepted (%d)'", ssc_file_dat, AS);
    fprintf(SSC_gnuplotPipe, ", '%s' using 1:($3 == 0 ? $2 : 1/0) w points ps 0.7 lc rgb 'red' title 'Rejected (%d)'\n", ssc_file_dat, RS); 
    fprintf(SSC_gnuplotPipe, "unset output\n"); 

    fprintf(SR_gnuplotPipe, "plot '%s' using 1:2 w points ps 0.7 lc rgb 'blue' notitle\n", sr_file_dat);
    fprintf(SR_gnuplotPipe, "unset output\n"); 

    fprintf(SSCxSR_gnuplotPipe, "plot '%s' using 1:2 w points ps 0.7 lc rgb 'blue' notitle\n", sccxsr_file_dat);
    fprintf(SSCxSR_gnuplotPipe, "unset output\n"); 

    fprintf(SPEC_gnuplotPipe, "plot '%s' using ($3 == 0 ? $1 : 1/0):($3 == 0 ? $2 : 1/0) w lines lc rgb 'black' notitle", spec_file_dat);
    for (jj=1; jj<JJ+1; jj++){
        fprintf(SPEC_gnuplotPipe, ", '%s' using ($3 == %d ? $1 : 1/0):($3 == %d ? $2 : 1/0) w points ps 0.7 title '$\\lambda_%d h$'", spec_file_dat, jj, jj, jj);
    }
    fprintf(SPEC_gnuplotPipe, "\nunset output\n"); 

    fprintf(PSEUDOSPEC_gnuplotPipe, "splot '%s' using 1:2:3 w lines dt 2 lc rgb 'black' nosurface notitle", pseudospec_file_dat);
    fprintf(PSEUDOSPEC_gnuplotPipe, ", '%s' using ($3 == 0 ? $1 : 1/0):($3 == 0 ? $2 : 1/0):(0) w lines lc rgb 'black' nocontour notitle", spec_file_dat);
    fprintf(PSEUDOSPEC_gnuplotPipe, "\nunset output\n"); 


    printf("\nAccepted=%d, Rejected=%d\nh_min=%le, h_max=%le\n", AS, RS, h_min, h_max);

    printf("Most negative lambda·h (%le + i %le)*%le=(%le + i %le)  at t=%lf\n", minimizing_WR[minimizing_jj], minimizing_WI[minimizing_jj], minimizing_ssc, minimizing_ssc*minimizing_WR[minimizing_jj], minimizing_ssc*minimizing_WI[minimizing_jj], minimizing_t);

    for (jj=0; jj<JJ; jj++) printf("lambda_%d=%le + i %le \n", jj+1, minimizing_WR[jj], minimizing_WI[jj]);

    stiff_gamma/=stiff_T;
    printf("Stiffness coefficients: kappa=%le, gamma=%le, sigma=%le, nu=%le\n", stiff_kappa/stiff_nu, stiff_gamma/stiff_nu, stiff_kappa/stiff_gamma, stiff_nu);

    printf("Eigenv. of the dominant eigenv. at origin: v_%d=(%le", stiff_VR_jj+1, stiff_VR[stiff_VR_jj]);
    for (jj=1; jj<JJ; jj++){
        printf(",%le", stiff_VR[stiff_VR_jj+jj*JJ]);
    }
    printf(")\n");

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    return 0;
}
