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

#define OD_zero_TOL 1e-20
#define EV_TOL OD_zero_TOL

#define IEV_DEG 2
#define SEC_IND 1

#define verbose              0
#define verbose_OD         ( 0 || verbose )
#define verbose_projection ( 1 || verbose )


// 1e+5
// Accepted=917798, Rejected=31033
// h_min=2.913000e-09, h_max=1.750711e-01

// Execution time: 162.227937 seconds


MY_FLOAT s=77.27, w=0.161, q=8.375e-6, fff=1.0;


int main(int argc, char *argv[]){

    if (DD<=1){ printf("ERROR. The number of symbols is %d.\n", DD); exit(0);}
    if (IEV_DEG!=DD) { printf("ERROR. The number of symbols (%d) does not match the degree (%d) of the inhomogeneous eigenvector.\n", DD, IEV_DEG); exit(0);}

    int jj, ss, dd, cc, i, od, iev_deg;
    int od_t=0, forced_od_bool=0;
    int OD;

    char OD_num_name[] = "ORBIT DECOMP/num_OD.txt", OD_part_name[] = "ORBIT DECOMP/partition.bin", OD_orbit_name[] = "ORBIT DECOMP/orbit.bin", OD_EVEC_name[] = "ORBIT DECOMP/EVEC.bin", OD_b_name[] = "ORBIT DECOMP/b.bin";
    FILE *OD_num_txt, *OD_part_bin, *OD_orbit_bin, *OD_EVEC_bin, *OD_b_bin;
    OD_b_bin = fopen(OD_b_name, "wb");

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
    OD_EVEC_bin = fopen(OD_EVEC_name, "rb");
    if (OD_EVEC_bin==NULL) exit(0);

    double **OD_orbit = malloc(OD*sizeof(double *));
    double **OD_EVEC = malloc(OD*sizeof(double *));
    for (od=0; od<OD; od++){
        OD_orbit[od] = malloc(JJ*sizeof(double));
        OD_EVEC[od] = malloc(JJ*sizeof(double));
        fread(OD_orbit[od], sizeof(double), JJ, OD_orbit_bin);
        fread(OD_EVEC[od], sizeof(double), JJ, OD_EVEC_bin);
    } 

    double tmp_write[JJ];

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
        MY_JET_DATA(jet_xx[jj], 2) = 0.0;
    }

    //Set initial and final time and step size
    t = 0.0; t_F = 302.858044027585265;
    step_size = H0; 

    if (verbose_OD || verbose_projection){
        //Print initial values
        printf("t: %lf\n", t);
        for (jj=0; jj<JJ; jj++) {
            printf("%s: ", var_names[jj]);
            for (cc=0; cc<CC; cc++){ 
                printf("%.17le \t", MY_JET_DATA(jet_xx[jj],cc));
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
                for (jj=0; jj<JJ; jj++) tmp_write[jj] = MY_JET_DATA(jet_xx[jj],CC-1);
                fwrite(tmp_write, sizeof(double), JJ, OD_b_bin);

                if (verbose_OD){
                    printf("\nt: %lf\n", t);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        for (cc=0; cc<CC; cc++){ 
                            printf("%.17le   \t", MY_JET_DATA(jet_xx[jj],cc));
                        }
                        printf("\n");
                    }
                }

                od_t++;

                for (jj=0; jj<JJ; jj++){
                    MY_JET_DATA(jet_xx[jj], 0) = OD_orbit[od_t][jj];
                    MY_JET_DATA(jet_xx[jj], 1) = OD_EVEC[od_t][jj];
                    MY_JET_DATA(jet_xx[jj], 2) = 0.0;
                }
            } else {

                if (verbose_projection){
                    printf("\nt: %lf\n", t);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        for (cc=0; cc<CC; cc++){ 
                            printf("%.17le   \t", MY_JET_DATA(jet_xx[jj],cc));
                        }
                        printf("\n");
                    }
                }

                MY_FLOAT proj_normal[JJ], proj_out[JJ], proj_xx[JJ];
                for (jj=0; jj<JJ; jj++) proj_normal[jj]=0.0;
                proj_normal[SEC_IND] = 1.0;

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

                    if (verbose_projection){
                        printf("\n\nt: %lf   \t", t);
                        for (cc=1; cc<CC; cc++) printf("%le    \t", MY_JET_DATA(jet_TT, cc));
                        printf("\n");
                        for (jj=0; jj<JJ; jj++) {
                            printf("%s: ", var_names[jj]);
                            for (cc=0; cc<CC; cc++){ 
                                printf("%.17le   \t", MY_JET_DATA(jet_xx[jj],cc));
                            }
                            printf("\n");
                        }
                    }

                    MY_JET_DATA(jet_TT, dd)=0.0;

                }

                for (jj=0; jj<JJ; jj++) tmp_write[jj] = MY_JET_DATA(jet_xx[jj],CC-1);
                fwrite(tmp_write, sizeof(double), JJ, OD_b_bin);

                if (verbose_OD && !verbose_projection){
                    // Print last values
                    printf("\n\nt: %lf\n", t);
                    for (jj=0; jj<JJ; jj++) {
                        printf("%s: ", var_names[jj]);
                        for (cc=0; cc<CC; cc++){ 
                            printf("%.17le   \t", MY_JET_DATA(jet_xx[jj],cc));
                        }
                        printf("\n");
                    }
                }

            }

        }

    }

    printf("\nAccepted=%d, Rejected=%d\nh_min=%le, h_max=%le\n", AS, RS, h_min, h_max);

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    return 0;
}