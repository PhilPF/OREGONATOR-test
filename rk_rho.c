#include <stdio.h>
#include <math.h>
#include "rk_rho.h"

#define RKS _RKS_
#define CC _JET_COEFFICIENTS_COUNT_TOTAL_
#define JJ _NUMBER_OF_JET_VARS_

int init_flag_Z_rho_F=1;

void rho_function(MY_JET *jet_out, MY_FLOAT t, MY_JET *jet_xx){

    int cc, jj;

    MY_JET **function_jet;
    MY_FLOAT temporal_state_xx[JJ], **coeff;

    coeff = taylor_coefficients_taylor_rho_A(t, temporal_state_xx, 0, 1, jet_xx, &function_jet);

    for (jj=0; jj<JJ; jj++) {
        for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_out[jj], cc) = MY_JET_DATA(function_jet[jj][1],cc);}
    }
 
}

void Z_rho_F(MY_JET stages_out[RKS][JJ], void (*function)(MY_JET*, MY_FLOAT, MY_JET*), MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx, MY_FLOAT state_stages_in[RKS][JJ], int *init_flag){

    int rks, Xrks, jj, cc;

    int tjj;

    static MY_JET temporalJet[2][JJ], f_stages[RKS][JJ];
    if(*init_flag){
        for(jj=0; jj<JJ; jj++){
            for (tjj=0; tjj<2; tjj++) {InitJet(temporalJet[tjj][jj]);}
            for (rks=0; rks<RKS; rks++) {InitJet(f_stages[rks][jj]);}
        }
        *init_flag = 0;
    }

    /*Compute f(x+c_j*h, y+z_j) for all j and store it in stage_out*/
    for (rks=0; rks<RKS; rks++){ for (jj=0; jj<JJ; jj++){ AddJetFloatA(stages_out[rks][jj], jet_xx[jj], state_stages_in[rks][jj]); }}
    for (rks=0; rks<RKS; rks++){ rho_function(f_stages[rks], t+c[rks]*step_size, stages_out[rks]); }

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

        for(jj=0; jj<JJ; jj++){ SubtractFloatJetA(stages_out[rks][jj], state_stages_in[rks][jj], temporalJet[0][jj]); }

    }

}

// void rho_F_l(MY_JET stages_out[RKS][JJ], int l, void (*rho_function)(MY_JET*, MY_FLOAT, MY_JET*), MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx, MY_FLOAT state_stages_in[RKS][JJ]){

//     int jj, rks;

//     int TJJ=2;
//     int tjj;

//     MY_FLOAT sumStages[JJ];
//     MY_JET temporalJet[TJJ][JJ];
//     for (jj=0; jj<JJ; jj++) { for (tjj=0; tjj<TJJ; tjj++) { InitJet(temporalJet[tjj][jj]); }}

//     for (jj=0; jj<JJ; jj++) sumStages[jj]=0;

//     /*Compute h(sum_{j=1}^s a_{ij}*k_j) and store it in sumStages*/
//     for(rks=0; rks<RKS; rks++){
//         for(jj=0; jj<JJ; jj++) sumStages[jj]+= step_size*A[l][rks]*state_stages_in[rks][jj];
//     }

//     /*Compute y+h(sum_{j=1}^s a_{ij}*k_j) */
//     for(jj=0; jj<JJ; jj++) AddJetFloatA(temporalJet[0][jj], jet_xx[jj], sumStages[jj]);

//     /*Compute f(x+c_i*h, y+h(sum_{j=1}^s a_{ij}*k_j))*/
//     rho_function(temporalJet[1], t+c[l]*step_size, temporalJet[0]);

//     /*Compute k_i-f(x+c_i*h, y+h(sum_{j=1}^s a_{ij}*k_j)) and store it in stages_out[l]*/
//     for(jj=0; jj<JJ; jj++) SubtractFloatJetA(stages_out[l][jj], state_stages_in[l][jj], temporalJet[1][jj]);

// }

void Z_rho(MY_FLOAT rho_DF[RKS*JJ][JJ], MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *state_jet_xx, MY_FLOAT state_stages_in[RKS][JJ], int *init_flag){

    int cc, jj, rks;

    static MY_JET jet_temp[JJ], stages_out[RKS][JJ];
    if (*init_flag){
        for (jj=0; jj<JJ; jj++) { InitJet(jet_temp[jj]);}
        for (rks=0; rks<RKS; rks++) { for (jj=0; jj<JJ; jj++) { InitJet(stages_out[rks][jj]); }}
        *init_flag = 0;    
    }

    for (jj=0; jj<JJ; jj++){
        MY_JET_DATA(jet_temp[jj],0)=state_jet_xx[jj];
        MY_JET_DATA(jet_temp[jj],jj+1)=1.0;
    }

    Z_rho_F(stages_out, rho_function, step_size, t, jet_temp, state_stages_in, &init_flag_Z_rho_F);

    for (rks=0; rks<RKS; rks++){
        for (jj=0; jj<JJ; jj++){
            for (cc=0; cc<CC-1; cc++){
                rho_DF[rks*JJ+jj][cc] = MY_JET_DATA(stages_out[rks][jj], cc+1);
            }
        }
    }
}

// void rho(MY_FLOAT rho_DF[RKS*JJ][JJ], MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *state_jet_xx, MY_FLOAT state_stages_in[RKS][JJ]){

//     int cc, jj, rks;

//     MY_JET jet_temp[JJ], stages_out[RKS][JJ];
//     for (jj=0; jj<JJ; jj++) { InitJet(jet_temp[jj]);}
//     for (rks=0; rks<RKS; rks++) { for (jj=0; jj<JJ; jj++) { InitJet(stages_out[rks][jj]); }}

//     for (jj=0; jj<JJ; jj++){
//         MY_JET_DATA(jet_temp[jj],0)=state_jet_xx[jj];
//         MY_JET_DATA(jet_temp[jj],jj+1)=1.0;
//     }

//     for (rks=0; rks<RKS; rks++){ rho_F_l(stages_out, rks, rho_function, step_size, t, jet_temp, state_stages_in);}

//     for (rks=0; rks<RKS; rks++){
//         for (jj=0; jj<JJ; jj++){
//             for (cc=0; cc<CC-1; cc++){
//                 rho_DF[rks*JJ+jj][cc] = MY_JET_DATA(stages_out[rks][jj], cc+1);
//             }
//         }
//     }
// }