#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include "rk.h"
#include "method.h"
#include "rk_sigma.h"
#include "rk_rho.h"

#define RKS _RKS_
#define uround 1E-16

#define NEWT_TOL 1E-15
#define NEWT_MAX_IT 10

#define JJ _NUMBER_OF_JET_VARS_
#define SS _NUMBER_OF_MAX_SYMBOLS_
#define DD _MAX_DEGREE_OF_JET_VARS_
#define CC _JET_COEFFICIENTS_COUNT_TOTAL_

#define RKSxJJ RKS*JJ

int init_flag_FLOAT_function=1, init_flag_Z_F=1, init_flag_Z_sigma=1, init_flag_Z_rho=1, init_flag_RK_Implicit=1;
double last_eta=uround;

void function(MY_JET *jet_out, MY_FLOAT t, MY_JET *jet_xx){

    int jj, cc;

    MY_JET **function_jet;
    MY_FLOAT temporal_state_xx[JJ], **coeff;

    coeff = taylor_coefficients_taylor_A(t, temporal_state_xx, 0, 1, jet_xx, &function_jet);

    for (jj=0; jj<JJ; jj++) {
        for (cc=0; cc<CC; cc++){ 
            MY_JET_DATA(jet_out[jj], cc) = MY_JET_DATA(function_jet[jj][1],cc); 
        }
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

void Z_F(MY_JET stages_out[RKS][JJ], void (*function)(MY_JET*, MY_FLOAT, MY_JET*), MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx, MY_JET stages_in[RKS][JJ], int *init_flag){

    int rks, Xrks, jj, cc;

    int tjj;

    static MY_JET temporalJet[2][JJ], f_stages[RKS][JJ];
    if (*init_flag){
        for(jj=0; jj<JJ; jj++){
            for (tjj=0; tjj<2; tjj++) {InitJet(temporalJet[tjj][jj]);}
            for (rks=0; rks<RKS; rks++) {InitJet(f_stages[rks][jj]);}
        }
        *init_flag=0;
    }

    /*Compute f(x+c_j*h, y+z_j) for all j and store it in stage_out*/
    for (rks=0; rks<RKS; rks++){ for (jj=0; jj<JJ; jj++){ AddJetJetA(stages_out[rks][jj], jet_xx[jj], stages_in[rks][jj]); }}
    for (rks=0; rks<RKS; rks++){ function(f_stages[rks], t+c[rks]*step_size, stages_out[rks]); }

    for (rks=0; rks<RKS; rks++){

        if (RKS%2==1){
            for(jj=0; jj<JJ; jj++){ MultiplyFloatJetA(temporalJet[0][jj], step_size*A[rks][0], f_stages[0][jj]); }
        } else {
            for(jj=0; jj<JJ; jj++){ MultiplyFloatJetA(stages_out[rks][jj], step_size*A[rks][0], f_stages[0][jj]); }
        }

        for (Xrks=RKS-1; Xrks>0; Xrks--){
            if (Xrks%2==1){
                for (jj=0; jj<JJ; jj++){ MultiplyFloatJetA(temporalJet[1][jj], step_size*A[rks][Xrks], f_stages[Xrks][jj]); }
                for (jj=0; jj<JJ; jj++){ AddJetJetA(temporalJet[0][jj], stages_out[rks][jj], temporalJet[1][jj]); }
            }
            else {
                for (jj=0; jj<JJ; jj++){ MultiplyFloatJetA(temporalJet[1][jj], step_size*A[rks][Xrks], f_stages[Xrks][jj]); }
                for (jj=0; jj<JJ; jj++){ AddJetJetA(stages_out[rks][jj], temporalJet[0][jj], temporalJet[1][jj]); }
            }
        }

        for(jj=0; jj<JJ; jj++){ SubtractJetJetA(stages_out[rks][jj], stages_in[rks][jj], temporalJet[0][jj]); }

    }

}

void FLOAT_Z_F(MY_FLOAT sigma_F[RKSxJJ], void (*FLOAT_function)(MY_FLOAT*, MY_FLOAT, MY_FLOAT*), MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *xx, MY_FLOAT stages_in[RKS][JJ]){

    int rks, Xrks, jj, cc;

    int tjj;

    MY_FLOAT temporalFloat[1][JJ], f_stages[RKS][JJ];

    /*Compute f(x+c_j*h, y+z_j) for all j and store it in stage_out*/
    for (rks=0; rks<RKS; rks++){ 
        for (jj=0; jj<JJ; jj++){ temporalFloat[0][jj] = xx[jj] + stages_in[rks][jj]; }
        FLOAT_function(f_stages[rks], t+c[rks]*step_size, temporalFloat[0]); 
    }

    for (rks=0; rks<RKS; rks++){

        for (jj=0; jj<JJ; jj++){ temporalFloat[0][jj] = 0.0;}

        for (Xrks=0; Xrks<RKS; Xrks++){
            for (jj=0; jj<JJ; jj++){ temporalFloat[0][jj] += step_size*A[rks][Xrks]*f_stages[Xrks][jj]; }
        }

        for(jj=0; jj<JJ; jj++){ sigma_F[rks*JJ+jj] = stages_in[rks][jj] - temporalFloat[0][jj]; }

    }

}

int stage_Newt(MY_JET stages_in[RKS][JJ], MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx){

    int rks, jj, cc;
    int rksxjj1, rksxjj2;

    MY_FLOAT sigma_F[RKSxJJ], sigma_DF[RKSxJJ][RKSxJJ];
    double norm_sigma_F, last_norm_sigma_F, Omega, eta=pow(fmax(last_eta,uround), 0.8);

    double LAPACK_sigma_DF[RKSxJJ*RKSxJJ];
    lapack_int LAPACK_info, *LAPACK_ipiv;
    LAPACK_ipiv = (lapack_int *)malloc((RKSxJJ)*sizeof(lapack_int)) ;
    
    MY_FLOAT state_jet_xx[JJ], state_stages_in[RKS][JJ];
    for (jj=0; jj<JJ; jj++) { 
        state_jet_xx[jj] = MY_JET_DATA(jet_xx[jj], 0);
        for (rks=0; rks<RKS; rks++) { state_stages_in[rks][jj] = 0; } //Set initial approx.
    }

    //Compute F(x_0) and DF(x_0)
    Z_sigma(sigma_F, sigma_DF, step_size, t, state_jet_xx, state_stages_in, &init_flag_Z_sigma);

    //Store (RKSxJJ)x(RKSxJJ) matrix as (RKSxJJ*RKSxJJ) array for LAPACK
    for (rksxjj1=0; rksxjj1<RKSxJJ; rksxjj1++){
        for (rksxjj2=0; rksxjj2<RKSxJJ; rksxjj2++){
            LAPACK_sigma_DF[rksxjj1*(RKSxJJ)+rksxjj2] = sigma_DF[rksxjj1][rksxjj2];
        }
    }

    LAPACK_info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, RKSxJJ, RKSxJJ, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv);
    if (LAPACK_info != 0) { printf("LAPACK ERROR %d.\n", LAPACK_info); return 1; } 

    int newt_it = 0;

    do{

        //Solve the system DF(x_0)·u=F(x_0)
        LAPACK_info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', RKSxJJ, 1, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv, sigma_F, 1);
        if (LAPACK_info != 0) { printf("LAPACK ERROR %d.\n", LAPACK_info); return 1; } 

        //Store the solution x_1:=x_0-u
        for (rks=0; rks<RKS; rks++){
            for (jj=0; jj<JJ; jj++){
                state_stages_in[rks][jj] -= sigma_F[rks*JJ+jj];
            }
        }

        //Compute F(x_1) for next iteration
        FLOAT_Z_F(sigma_F, FLOAT_function, step_size, t, state_jet_xx, state_stages_in);

        //Compute the norm of F(x_1)
        norm_sigma_F = 0;
        for (rks=0; rks<RKS; rks++){
            for (jj=0; jj<JJ; jj++){
                norm_sigma_F += pow(sigma_F[rks*JJ+jj], 2.0);
            }
        }

        norm_sigma_F/=RKSxJJ;
        norm_sigma_F=sqrt(norm_sigma_F);

        if(newt_it>0){
            Omega=norm_sigma_F/last_norm_sigma_F;
            // if (Omega>1) return -1;

            eta=Omega/(1-Omega);
        }

        last_norm_sigma_F = norm_sigma_F;
        last_eta=eta;

        newt_it ++;

        if (newt_it>NEWT_MAX_IT) return -1;

        // printf("Newt it:%d, \t norm_sigma_F:%le\n", newt_it, norm_sigma_F);

    //Check if norm is close to 0 to terminate Newton
    }while(eta*norm_sigma_F>0.05*NEWT_TOL);

    if (!isfinite(norm_sigma_F)){
        printf("Newton method error, value of norm is %f\n", norm_sigma_F);
        return 1;
    }

    for (rks=0; rks<RKS; rks++){
        for (jj=0; jj<JJ; jj++){
            MY_JET_DATA(stages_in[rks][jj], 0) = state_stages_in[rks][jj];
        }
    }

    free(LAPACK_ipiv);

    return 0;

}

int max_monomial_counts(const int *monomial_counts){

    int dd;
    int max=monomial_counts[0]; 

    for(dd=1; dd<=DD; dd++){
        int mc = monomial_counts[dd];
        if (mc>max) max = mc;
    }

    return max;

}

int RK_Implicit(MY_JET *jet_out, MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx){

    int rks, jj, ss, dd, cc;
    int rksxjj1, rksxjj2;
    int MC, MO;
    int tjj;

    static int *monomial_counts;
    static int *monomial_offsets;
    static int MMC;

    static MY_JET stages_in[RKS][JJ], stages_out[RKS][JJ];
    static MY_JET truncated_jet_xx[JJ];
    static MY_JET temporalJet[2][JJ], f_stages[RKS][JJ];
    if (init_flag_RK_Implicit){
        for (jj=0; jj<JJ; jj++) {
            for (rks=0; rks<RKS; rks++) { 
                InitJet(stages_in[rks][jj]); 
                InitJet(stages_out[rks][jj]); 
                InitJet(f_stages[rks][jj]);
            }
            InitJet(truncated_jet_xx[jj]); 
            for (tjj=0; tjj<2; tjj++) { InitJet(temporalJet[tjj][jj]);}
        }
        monomial_counts = jet_tree_monomial_counts_taylor();
        monomial_offsets = jet_tree_monomial_offsets_taylor();
        MMC = max_monomial_counts(monomial_counts);

        init_flag_RK_Implicit=0;
    }

    //Use Newton to compute the value of the RK stages (0 order coeff.)
    int info_Newt;
    info_Newt = stage_Newt(stages_in, step_size, t, jet_xx); 
    if (info_Newt>0){ exit(0);}
    if (info_Newt<0){ return info_Newt;}

    MY_FLOAT state_jet_xx[JJ], state_stages_in[RKS][JJ];
    for (jj=0; jj<JJ; jj++) { 
        state_jet_xx[jj] = MY_JET_DATA(jet_xx[jj], 0);
        for (rks=0; rks<RKS; rks++) { 
            state_stages_in[rks][jj] = MY_JET_DATA(stages_in[rks][jj], 0); 
            for (cc=1; cc<CC; cc++){ MY_JET_DATA(stages_in[rks][jj], cc)=0.0; }
        }
    }

    MY_FLOAT sigma_DF[RKSxJJ][RKSxJJ], rho_DF[RKSxJJ][JJ];

    //Compute the matrices D_A F(A^[0], B^[0]), D_B F(A^[0], B^[0])
    Z_sigma(NULL, sigma_DF, step_size, t, state_jet_xx, state_stages_in, &init_flag_Z_sigma);
    Z_rho(rho_DF, step_size, t, state_jet_xx, state_stages_in, &init_flag_Z_rho);

    //Convert the matrices to arrays for LAPACK and BLAS
    double LAPACK_sigma_DF[RKSxJJ*RKSxJJ], LAPACK_rho_DF[RKSxJJ*JJ];
    for (rksxjj1=0; rksxjj1<RKSxJJ; rksxjj1++){
        for (rksxjj2=0; rksxjj2<RKSxJJ; rksxjj2++){
            LAPACK_sigma_DF[rksxjj1*(RKSxJJ)+rksxjj2]=sigma_DF[rksxjj1][rksxjj2];
        }
        for (jj=0; jj<JJ; jj++){
            LAPACK_rho_DF[rksxjj1*(JJ)+jj]=rho_DF[rksxjj1][jj];
        }
    }

    lapack_int LAPACK_info, *LAPACK_ipiv;
    LAPACK_ipiv = (lapack_int *)malloc((RKSxJJ)*sizeof(lapack_int));

    //Compute the LU factorization of D_B F(A^[0], B^[0])
    LAPACK_info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, RKSxJJ, RKSxJJ, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv);
    if (LAPACK_info != 0) { printf("LAPACK ERROR %d.\n", LAPACK_info); exit(0); }    

    //This matrices will be used to store all the coefficients per degree of the known (A) and unknown (B) jets
    double sigma_coeffs[MMC][RKSxJJ], rho_coeffs[MMC][JJ];
    //This is the array version of the first one
    const int MMCxRKSxJJ = MMC*RKSxJJ; 
    double LAPACK_sigma_coeffs[MMCxRKSxJJ];

    //We now compute B^[k] for |k|=1
    MC = monomial_counts[1]; MO = monomial_offsets[1];
    if (MC==1){

        //Get the values -A^[k] and multiply them by D_A F(A^[0], B^[0]) 
        // printf("\nrho_coeffs\n");
        for (jj=0; jj<JJ; jj++){ rho_coeffs[0][jj] = -MY_JET_DATA(jet_xx[jj], MO); }
        cblas_dgemv(CblasRowMajor, CblasNoTrans, RKSxJJ, JJ, 1.0, LAPACK_rho_DF, JJ, rho_coeffs[0], 1, 0, sigma_coeffs[0], 1);

        //Find B^[k] solving the linear system
        LAPACK_info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', RKSxJJ, 1, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv, sigma_coeffs[0], 1);
        if (LAPACK_info != 0) { printf("LAPACK ERROR %d.\n", LAPACK_info); exit(0); }    

        //Store the values B^[k] in B to continue the computations for higher order k
        for (rks=0; rks<RKS; rks++){
            for (jj=0; jj<JJ; jj++){
                MY_JET_DATA(stages_in[rks][jj], MO) = sigma_coeffs[0][rks*JJ+jj];
            }
        }

    } else {

        for (ss=0; ss<MC; ss++){
            //Get the values -A^[k] and multiply them by D_A F(A^[0], B^[0]) 
            for (jj=0; jj<JJ; jj++){ rho_coeffs[ss][jj] = -MY_JET_DATA(jet_xx[jj], MO+ss);}
            cblas_dgemv(CblasRowMajor, CblasNoTrans, RKSxJJ, JJ, 1.0, LAPACK_rho_DF, JJ, rho_coeffs[ss], 1, 0, sigma_coeffs[ss], 1);

            //Convert the matrices to arrays so that LAPACK can solve several systems at once
            for (rksxjj1=0; rksxjj1<RKSxJJ; rksxjj1++){
                LAPACK_sigma_coeffs[rksxjj1*MC+ss] = sigma_coeffs[ss][rksxjj1];
            }
        }        

        //Find B^[k] solving the linear system
        LAPACK_info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', RKSxJJ, MC, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv, LAPACK_sigma_coeffs, MC);
        if (LAPACK_info != 0) {printf("LAPACK ERROR.\n"); exit(0);}    

        //Store the values B^[k] in B to continue the computations for higher order k
        for (rks=0; rks<RKS; rks++){
            for (jj=0; jj<JJ; jj++){
                for (ss=0; ss<MC; ss++){
                    MY_JET_DATA(stages_in[rks][jj], MO+ss) = LAPACK_sigma_coeffs[(rks*JJ+jj)*MC+ss];
                }
            }
        }
        
    } 


    for (jj=0; jj<JJ; jj++) {
        MY_JET_DATA(truncated_jet_xx[jj], 0) = MY_JET_DATA(jet_xx[jj], 0);
        for (cc=1; cc<CC; cc++) MY_JET_DATA(truncated_jet_xx[jj], cc) = 0.0;
    }

    for (dd=2; dd<=DD; dd++){

        for (jj=0; jj<JJ; jj++) {
            for (ss=0; ss<MC; ss++){ MY_JET_DATA(truncated_jet_xx[jj], MO+ss) = MY_JET_DATA(jet_xx[jj], MO+ss);}
        }

        //We now compute B^[k] for |k|=m
        //We first compute the jet F([A]^{m-1}_K, [B]^{m-1}_K)
        Z_F(stages_out, function, step_size, t, truncated_jet_xx, stages_in, &init_flag_Z_F);

        MC = monomial_counts[dd]; MO = monomial_offsets[dd];

        if (MC==1){

            //Get the values -A^[k] and multiply them by D_A F(A^[0], B^[0]) 
            for (jj=0; jj<JJ; jj++){ rho_coeffs[0][jj] = -MY_JET_DATA(jet_xx[jj], MO);}
            cblas_dgemv(CblasRowMajor, CblasNoTrans, RKSxJJ, JJ, 1.0, LAPACK_rho_DF, JJ, rho_coeffs[0], 1, 0, sigma_coeffs[0], 1);

            //Now subtract the value F([A]^{m-1}_K, [B]^{m-1}_K)^[k]
            for (rks=0; rks<RKS; rks++){
                for (jj=0; jj<JJ; jj++){
                    sigma_coeffs[0][rks*JJ+jj] -= MY_JET_DATA(stages_out[rks][jj], MO);
                }
            }

            //Find B^[k] solving the linear system
            LAPACK_info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', RKSxJJ, 1, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv, sigma_coeffs[0], 1);
            if (LAPACK_info != 0) {printf("LAPACK ERROR.\n"); exit(0);}    

            //Store the values B^[k] in B to continue the computations for higher order k
            for (rks=0; rks<RKS; rks++){
                for (jj=0; jj<JJ; jj++){
                    MY_JET_DATA(stages_in[rks][jj], MO) = sigma_coeffs[0][rks*JJ+jj];
                }
            }

        } else {

            for (ss=0; ss<MC; ss++){
                //Get the values A^[k] and multiply them by D_A F(A^[0], B^[0]) 
                for (jj=0; jj<JJ; jj++){ rho_coeffs[ss][jj] = MY_JET_DATA(jet_xx[jj], MO+ss);}
                cblas_dgemv(CblasRowMajor, CblasNoTrans, RKSxJJ, JJ, 1.0, LAPACK_rho_DF, JJ, rho_coeffs[ss], 1, 0, sigma_coeffs[ss], 1);

                //Now subtract the value F([A]^{m-1}_K, [B]^{m-1}_K)^[k]
                //and convert the matrices to arrays so that LAPACK can solve several systems at once
                for (rks=0; rks<RKS; rks++){
                    for (jj=0; jj<JJ; jj++){
                        LAPACK_sigma_coeffs[(rks*JJ+jj)*MC+ss] = - MY_JET_DATA(stages_out[rks][jj], MO+ss) - sigma_coeffs[ss][rks*JJ+jj];
                    }
                }
            }

            //Find B^[k] solving the linear system
            LAPACK_info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', RKSxJJ, MC, LAPACK_sigma_DF, RKSxJJ, LAPACK_ipiv, LAPACK_sigma_coeffs, MC);
            if (LAPACK_info != 0) {printf("LAPACK ERROR.\n"); exit(0);}    

            //Store the values B^[k] in B to continue the computations for higher order k
            for (rks=0; rks<RKS; rks++){
                for (jj=0; jj<JJ; jj++){
                    for (ss=0; ss<MC; ss++){
                        MY_JET_DATA(stages_in[rks][jj], MO+ss) = LAPACK_sigma_coeffs[(rks*JJ+jj)*MC+ss];
                    }
                }
            }
        }
    }

    free(LAPACK_ipiv);

    for(rks=0; rks<RKS; rks++){
        for (jj=0; jj<JJ; jj++) { AddJetJetA(stages_out[rks][jj], jet_xx[jj], stages_in[rks][jj]); }
        function(f_stages[rks], t+step_size*c[rks], stages_out[rks]);
    }

    if (RKS%2==0){
        for(jj=0; jj<JJ; jj++) { for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_out[jj], cc) = MY_JET_DATA(jet_xx[jj], cc); }}
    }else{
        for(jj=0; jj<JJ; jj++) { for (cc=0; cc<CC; cc++){ MY_JET_DATA(temporalJet[0][jj], cc) = MY_JET_DATA(jet_xx[jj], cc); }}
    }
    
    for(rks=RKS-1; rks>=0; rks--){
        if (rks%2==1){
            for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[1][jj], step_size*b[rks], f_stages[rks][jj]);
            for(jj=0; jj<JJ; jj++) AddJetJetA(temporalJet[0][jj], jet_out[jj], temporalJet[1][jj]);
        }else{
            for(jj=0; jj<JJ; jj++) MultiplyFloatJetA(temporalJet[1][jj], step_size*b[rks], f_stages[rks][jj]);
            for(jj=0; jj<JJ; jj++) AddJetJetA(jet_out[jj], temporalJet[0][jj], temporalJet[1][jj]);
        }
    }

    return 0;

}