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
#define H0 1E-5

#define OD 5000
#define OD_MAX 5000002
#define OD_zero_TOL 1e-20
#define EV_TOL OD_zero_TOL
#define store_Q 0

MY_FLOAT s=77.27, w=0.161, q=8.375e-6, fff=1.0;

void spectrum(double WR[JJ], double WI[JJ], double VR[JJ*JJ], MY_JET *jet_xx, double JAC[JJ*JJ]){

    int jj, cc;

    double MAT[JJ*JJ];

    if (JAC==NULL){
        for (jj=0; jj<JJ; jj++){
            for (cc=0; cc<JJ; cc++){
                MAT[cc+jj*JJ]=MY_JET_DATA(jet_xx[jj], 1+cc);
            }
        }
    } else {
        for (jj=0; jj<JJ; jj++){
            for (cc=0; cc<JJ; cc++){
                MAT[cc+jj*JJ]=JAC[cc+jj*JJ];
            }
        }
    }
    
    lapack_int info;

    if (VR==NULL){
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'N', JJ, MAT, JJ, WR, WI, NULL, 1, NULL, 1);
    }else {
        info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,'N', 'V', JJ, MAT, JJ, WR, WI, NULL, 1, VR, JJ);
    }
    if (info != 0) exit(0);

}

void condition(double *A_max, double *A_min, MY_JET *jet_xx, double *A){

    int jj, cc;
    double max=0, min=1e+200, temp;

    double WR[JJ], WI[JJ], _A[JJ*JJ];

    if (A==NULL){
        for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++) { _A[cc+jj*JJ]=MY_JET_DATA(jet_xx[jj], cc+1); }}
    } 

    /*if (A==NULL) {
        spectrum(WR, WI, NULL, NULL, _A);
    } else {
        spectrum(WR, WI, NULL, NULL, A);
    }

    // printf("SPECTRUM: ");
    for (jj=0; jj<JJ; jj++){
        temp = fabs(WR[jj]);
        if (temp>max) max=temp;
        if (temp<min)  min=temp;
        // printf("%le ", WR[jj]);
    }

    return max/min;*/

    double SV[JJ];

    lapack_int info;
    double *work = (double*)malloc(sizeof(double));

    if (A==NULL) info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'N', 'N', JJ, JJ, _A, JJ, SV, NULL, JJ, NULL, JJ, work);
    else info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'N', 'N', JJ, JJ, A, JJ, SV, NULL, JJ, NULL, JJ, work);

    if (info!=0) exit(0);

    // printf("SVD: ");
    for (jj=0; jj<JJ; jj++){
        temp = fabs(SV[jj]);
        if (temp>max) max=SV[jj];
        if (temp<min) min=SV[jj]; 
        // printf("%le ", SV[jj]);
    }
    // printf("\n");

    *A_max = max; *A_min = min;

}

void PHTD(double **A, double **Q, int od){

    double alpha, norm, v[JJ], M[JJ*JJ], R[JJ*JJ];

    int i,j;

    int m=1, s=0, t=0;

    while (s<2){

        // printf("\nm=%d, s=%d, t=%d\n", m,s,t);

        alpha=0; norm=0;

        for (i=s; i<JJ; i++) alpha+=A[m][i*JJ+t]*A[m][i*JJ+t];
        alpha = -A[m][s*JJ+t]/fabs(A[m][s*JJ+t])*sqrt(alpha);

        for (i=0; i<JJ-s; i++) {
            v[i] =  A[m][(i+s)*JJ+t];
            if (i==0) v[i]-=alpha;
            norm+=v[i]*v[i];
        }
        norm=sqrt(norm);

        for (i=0; i<JJ-s; i++){ 
            v[i]/=norm; 
        }

        if(store_Q){
            //BUILDING Q_{m+1}^T from v (only for storing Q)
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, JJ, JJ, 1, -2.0, v, 1, v, JJ, 0.0, M, JJ);
            for (i=0; i<JJ; i++){
                for (j=0; j<JJ; j++){
                    M[j+i*JJ] = M[j+i*JJ];
                    if (i==j) M[j+i*JJ]+=1.0; 
                }
            }
            if (m+1==od) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, JJ, JJ, JJ, 1.0, Q[0], JJ, M, JJ, 0.0, R, JJ);
            else cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, JJ, JJ, JJ, 1.0, Q[m+1], JJ, M, JJ, 0.0, R, JJ);
        }

        //Applying implicitly Q_{m+1}^T A_m and A_{m+1} Q_{m+1} (without using the matrix M. This is more stable)
        for (i=0; i<JJ-s; i++){
            for (j=0; j<JJ; j++){
                M[j+i*JJ]=A[m][j+(i+s)*JJ];
            }
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, JJ, JJ-s, 1.0, v, JJ-s, M, JJ, 0.0, R, JJ);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, JJ-s, JJ, 1, 2.0, v, 1, R, JJ, 0.0, M, JJ);


        // printf("R_%d\n", m+1);
        for (i=0; i<JJ; i++){
            for (j=0; j<JJ; j++){
                if (i==s && j==t) A[m][j+i*JJ] = alpha;
                else if (i>s && j==t) A[m][j+i*JJ] = 0.0;
                else if (i>=s) A[m][j+i*JJ] -= M[j+(i-s)*JJ];
                // printf("%le ", A[m][j+i*JJ]);
            }
            // printf("\n");
        }
        // printf("\n");

        m++;
        if (m==od) m=0;

        for (i=0; i<JJ; i++){
            for (j=0; j<JJ-s; j++){
                M[j+i*(JJ-s)]=A[m][(j+s)+i*JJ];
            }
        }

        /*cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, JJ, JJ, JJ, 1.0, A[m], JJ, M, JJ, 0.0, R, JJ);*/

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, JJ, 1, JJ-s, 1.0, M, JJ-s, v, 1, 0.0, R, 1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, JJ, JJ-s, 1, 2.0, R, 1, v, JJ-s, 0.0, M, JJ-s);

        // printf("R_%d\n", m+1);
        for (i=0; i<JJ; i++){
            for (j=0; j<JJ; j++){
                if (j>=s) A[m][j+i*JJ] -= M[(j-s)+i*(JJ-s)];
                // printf("%le ", A[m][j+i*JJ]);
            }
            // printf("\n");
        }
        // printf("\n");

        if (m==0){
            if (s==t) s++;
        } else if (s>t) t++;

    }

}

void Givens_coef(double *c, double *s, double *r, double a, double b){

    double abs_a=fabs(a), abs_b=fabs(b);
    double temp_1, temp_2;

    if (abs_b<OD_zero_TOL){
        if (abs_a<OD_zero_TOL) *c=1.0;
        else *c=a/abs_a;
        *s = 0;
        *r = abs_a;
    } else if (abs_b<OD_zero_TOL){
        *c = 0;
        *s = -b/abs_b;
        *r = abs_b;
    } else if (abs_a > abs_b){
        temp_1 = b/a;
        temp_2 = a/abs_a * sqrt(1+temp_1*temp_1);
        *c = 1/temp_2;
        *s = -temp_1/temp_2;
        *r = a * temp_2;
    } else {
        temp_1 = a/b;
        temp_2 = b/abs_b * sqrt(1+temp_1*temp_1);
        *s = -1/temp_2;
        *c = temp_1/temp_2;
        *r = b*temp_2;
    }

}

int PSD(double **A, double **Q, int od, int S_0, int S_N){

    int i, j, m=0;

    double shift, G_c, G_s, G_r, G_a, G_b;

    double stop_condition[S_N-S_0-1];
    int stop=1;

    while (stop){

        // Use for shift = "last diagonal element"
        /*shift = A[0][(S_N-1)+(S_N-1)*JJ];
        G_a = shift;

        for (i=1; i<od; i++){
            shift*=A[i][(S_N-1)+(S_N-1)*JJ];
            G_a*=(A[i][(S_N-1)+(S_N-1)*JJ]/A[i][S_0+S_0*JJ]);
        } */

        //Use for shift = 0
        shift = 0.0;
        G_a = 0.0;


        /*//Use for shift = 1
        if (S_0+S_N == JJ){
            shift = 1.0;
            G_a = shift;

            for (i=1; i<od; i++){
                G_a/=A[i][S_0+S_0*JJ];
            } 
        } else {
            //Only use shift = 1 for the full matrix
            //Then use this for shift = 0
            shift = 0.0;
            G_a = 0.0;

            // //Then use this for shift = "first diagonal element"
            // shift = A[0][S_0+S_0*JJ];
            // G_a = A[0][S_0+S_0*JJ];

            // for (i=1; i<od; i++){
            //     shift*=A[i][S_0+S_0*JJ];
            //     // G_a*=(A[i][(S_N-2)+(S_N-2)*JJ]/A[i][S_0+S_0*JJ]);
            // } 
        }*/
        
        //The rest is common for all shifts

        G_a = A[0][S_0+S_0*JJ] - G_a;
        G_b = A[0][S_0+(S_0+1)*JJ];

        // printf("Shift: %le, G_a: %le, G_b:% le\n", shift, G_a, G_b);

        Givens_coef (&G_c, &G_s, &G_r,G_a, G_b);

        // printf("--> G_c: %le, G_s: %le, G_r:% le\n", G_c, G_s, G_r);

        double G_temp_A;

        do{

            // printf("\nm=%d\n", m);

            A[m][S_0+S_0*JJ]  = G_r;
            A[m][S_0+(S_0+1)*JJ]  = 0.0;
            for (j=S_0+1; j<S_N; j++){
                G_temp_A = A[m][j+S_0*JJ];
                A[m][j+S_0*JJ]  = G_c*A[m][j+S_0*JJ]-G_s*A[m][j+(S_0+1)*JJ];
                A[m][j+(S_0+1)*JJ]  = G_s*G_temp_A+G_c*A[m][j+(S_0+1)*JJ];
            }
            /*printf("R_%d:\n", m+1);
            for (i=S_0; i<S_N; i++){
                for (j=S_0; j<S_N; j++){
                    printf("%le ", A[m][j+i*JJ]);
                }
                printf("\n");
            }*/

            m++;
            if (m==od) m=0;

            for (i=S_0; i<S_N; i++){  
                G_temp_A = A[m][S_0+i*JJ];
                A[m][S_0+i*JJ] = G_c*A[m][S_0+i*JJ]-G_s*A[m][(S_0+1)+i*JJ];
                A[m][(S_0+1)+i*JJ]  = G_s*G_temp_A+G_c*A[m][(S_0+1)+i*JJ];
                
                if (store_Q){
                    Q[m][S_0+i*JJ] = G_c*Q[m][S_0+i*JJ]-G_s*Q[m][(S_0+1)+i*JJ];
                    Q[m][(S_0+1)+i*JJ]  = G_s*G_temp_A+G_c*Q[m][(S_0+1)+i*JJ];
                }
            }
            /*printf("R_%d:\n", m+1);
            for (i=S_0; i<S_N; i++){
                for (j=S_0; j<S_N; j++){
                    printf("%le ", A[m][j+i*JJ]);
                }
                printf("\n");
            }*/

            // printf("G_a: %le, G_b:% le\n", A[m][S_0+S_0*JJ], A[m][S_0+(S_0+1)*JJ]);

            Givens_coef (&G_c, &G_s, &G_r, A[m][S_0+S_0*JJ], A[m][S_0+(S_0+1)*JJ]);

            // printf("--> G_c: %le, G_s: %le, G_r:% le\n", G_c, G_s, G_r);

        }while(m!=0);

        for (i=S_0; i<S_N-1; i++){
            stop_condition[i-S_0] = fabs(A[0][i+(i+1)*JJ]);
        }

        stop = (stop_condition[0] > OD_zero_TOL);
        for (i=1; i<S_N-S_0-1; i++){
            stop = stop && (stop_condition[i] > OD_zero_TOL); 
        }

        for (i=0; i<S_N-S_0-1; i++){
            stop = (stop_condition[i] > OD_zero_TOL); 
            if (stop == 0) return i; 
        }

    }


}


int main(int argc, char *argv[]){

    clock_t CLOCK_begin = clock();

    int jj, ss, dd, cc, i, od=0, RK_it=0;

    MY_FLOAT step_size, last_step_size, new_step_size, t, t_F;

    char OD_part_name[] = "ORBIT DECOMP/partition.dat", OD_A_name[] = "ORBIT DECOMP/A.dat", OD_R_name[] = "ORBIT DECOMP/R.dat", OD_Q_name[] = "ORBIT DECOMP/Q.dat", OD_EV_name[] = "ORBIT DECOMP/EV.dat";
    FILE * OD_part_dat = fopen(OD_part_name, "w");
    FILE * OD_A_dat = fopen(OD_A_name, "w");
    FILE * OD_R_dat = fopen(OD_R_name, "w");
    FILE * OD_Q_dat = fopen(OD_Q_name, "w");
    FILE * OD_EV_dat = fopen(OD_EV_name, "w");

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

    MY_JET_DATA(jet_xx[0], 0) = 1.999259240725097;
    MY_JET_DATA(jet_xx[1], 0) = 2.0;
    MY_JET_DATA(jet_xx[2], 0) = 1.767511227688383;

    for (jj=0; jj<JJ; jj++){
        for (cc=1; cc<CC; cc++){
            if (jj+1==cc) MY_JET_DATA(jet_xx[jj], cc) = 1.0;
            else MY_JET_DATA(jet_xx[jj], cc) = 0.0;
        }
    }

    //Set initial and final time and step size
    t = 0.0; t_F = 302.858044027585265;
    step_size= H0; 

    double **OD_JAC = malloc(OD_MAX*sizeof(double *));
    for (jj=0; jj<OD_MAX; jj++) OD_JAC[jj] = malloc(JJ*JJ*sizeof(double));

    double last_OD_JAC[JJ*JJ];

    // int OD_part_num=5e6, OD_part_true=0, cond_true=0, OD_frac_od=0;
    // double OD_part=t_F/(double)OD_part_num;
    // double cond_num[JJ*JJ], last_t_part, cond_temp;
    // double cond_OD = 1e7, inv_cond_OD=1.0/cond_OD;

    int OD_stepsize_fraction=1e7, OD_partition_bool=0, OD_first_magnitude_bool=1, OD_magnitude_condition_bool=0, OD_last_rejected=0, OD_RS=0, OD_RK_RS=0;
    double OD_stepsize=t_F/(double)OD_stepsize_fraction, OD_position=t, OD_new_stepsize, OD_error;
    double OD_TOL = log10(1e2)/*, OD_MINTOL = 5e0, OD_MAXTOL = 1e5*/;
    // double OD_TOL_inv=1.0/OD_TOL, OD_MINTOL_inv=1.0/OD_MINTOL, OD_MAXTOL_inv=1.0/OD_MAXTOL;
    double OD_last_t = t, OD_last_stepsize=step_size, OD_last_OD_position=OD_position, OD_last_OD_stepsize=OD_stepsize, OD_temp_double[JJ*JJ], OD_magnitude[JJ*JJ], OD_max_magnitude, OD_min_magnitude;
    static MY_JET OD_last_jet[JJ];
    for (jj=0; jj<JJ; jj++) {
        InitJet(OD_last_jet[jj]);
        for (cc=0; cc<CC; cc++) MY_JET_DATA(OD_last_jet[jj], cc) = MY_JET_DATA(jet_xx[jj], cc); 
    }

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

    // double cond_num_max, cond_num_min, last_cond_num_max=1.0, last_cond_num_min=1.0, last_t_part, cond_temp;

    while(t<t_F){

        if (step_size<1e-200){ printf("ERROR. The stepsize is negative or 0 (%.15le)\n", step_size); exit(0);}

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

        if (OD_first_magnitude_bool){
            OD_first_magnitude_bool = 0;
            for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++){ OD_magnitude[cc+jj*JJ] = fabs(MY_JET_DATA(tmp_2[jj], cc+1)); }}
        }

        //PARTITION BY MAGNITUDE-CONTROLLED PERIOD-FRACTION WITH TIME TRAVEL

        //printf("t=%.15lf (step_size=%.15le)\n", t+step_size, step_size);
        if (OD_partition_bool){
            // printf("OD: position=%.15lf (step_size=%.15le)\n", OD_position+OD_stepsize, OD_stepsize);

            OD_partition_bool=0;

            // OD_max_magnitude=0; OD_min_magnitude=1e+200;
            // for (jj=0; jj<JJ; jj++){
            //     for (cc=0; cc<JJ; cc++){
            //         OD_temp_double[0] = fabs(MY_JET_DATA(tmp_2[jj], cc+1));
            //         if (OD_temp_double[0]>OD_max_magnitude) OD_max_magnitude = OD_temp_double[0];
            //         if (OD_temp_double[0]<OD_min_magnitude) OD_min_magnitude = OD_temp_double[0];
            //     }
            // }

            condition(&OD_max_magnitude, &OD_min_magnitude, tmp_2, NULL);
            OD_error = log10(OD_max_magnitude/OD_min_magnitude)/OD_TOL;

            /*
            OD_error=0;
            OD_max_magnitude=0; OD_min_magnitude=1e+200;
            for (jj=0; jj<JJ; jj++){
                for (cc=0; cc<JJ; cc++){
                    OD_temp_double[cc+jj*JJ] = log10(fabs(MY_JET_DATA(tmp_2[jj], cc+1)))-log10(OD_magnitude[cc+jj*JJ]);
                    OD_error+=OD_temp_double[cc+jj*JJ]*OD_temp_double[cc+jj*JJ];
                    
                    OD_temp_double[cc+jj*JJ] = fabs(MY_JET_DATA(tmp_2[jj], cc+1))/OD_magnitude[cc+jj*JJ];
                    if (OD_temp_double[cc+jj*JJ]>OD_max_magnitude) OD_max_magnitude = OD_temp_double[cc+jj*JJ];
                    if (OD_temp_double[cc+jj*JJ]<OD_min_magnitude) OD_min_magnitude = OD_temp_double[cc+jj*JJ];
                }
            }
            // OD_error/=(JJ);
            OD_error=sqrt(OD_error)/OD_TOL;*/

            // printf("\tMAGNITUDES: MAX=%le, MIN=%le, ERROR=%lf, 1/ERROR=%lf\n", OD_max_magnitude, OD_min_magnitude, OD_error, 1/OD_error);

            OD_new_stepsize = fmax(0.5, h_fac*(1.0/OD_error));

            if (OD_error>1){
                OD_last_rejected=1;

                OD_RS++;
                // printf("\tOD deceleration (%d)", OD_RS);
                for (jj=0; jj<JJ; jj++) {
                    for (cc=0; cc<CC; cc++) MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(OD_last_jet[jj], cc); 
                }

                t = OD_last_t;
                step_size = OD_last_stepsize;
                OD_position = OD_last_OD_position;
                OD_stepsize = OD_last_OD_stepsize;

                OD_stepsize *= fmin(0.5, OD_new_stepsize);

                if (OD_stepsize<step_size){ 
                    // printf("\n\tOD stepsize reached RK stepsize!!\n\t"); 
                    step_size*=0.95;
                    OD_last_stepsize = step_size;
                    OD_stepsize=1.2*step_size;
                    OD_RK_RS++;
                }

                if (od==0){ OD_first_magnitude_bool=1; }

                if (t+step_size>=OD_position+OD_stepsize && t+step_size<t_F){
                    step_size=(OD_position+OD_stepsize)-t;
                    OD_partition_bool=1;
                }

                // printf(" --> stepsize=%le\n", OD_stepsize);

                continue;
            }  

            for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++){ OD_magnitude[cc+jj*JJ] = fabs(MY_JET_DATA(tmp_2[jj], cc+1)); }}

            OD_last_t = t+step_size;
            OD_last_stepsize = step_size;
            OD_last_OD_position = OD_position+OD_stepsize;
            OD_last_OD_stepsize = OD_stepsize;
        
            for (jj=0; jj<JJ; jj++){
                for (cc=0; cc<JJ; cc++){
                    OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(tmp_2[jj], cc+1);

                    if (jj==cc) MY_JET_DATA(tmp_2[jj], cc+1) = 1.0;
                    else MY_JET_DATA(tmp_2[jj], cc+1) = 0.0;
                }
            }

            for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){  MY_JET_DATA(OD_last_jet[jj], cc) = MY_JET_DATA(tmp_2[jj], cc); }}

            fprintf(OD_part_dat,"%.15lf\n", OD_last_t);

            od++;

            OD_position+=OD_stepsize;

            if (OD_last_rejected==0){
                OD_stepsize *= fmin(2.0, OD_new_stepsize);
            } else {
                OD_stepsize *= fmin(1.0, OD_new_stepsize);
                OD_last_rejected=0;
            }      

            if (OD_stepsize<step_size){ 
                // printf("\n\tOD stepsize reached RK stepsize!!\n\t"); 
                step_size*=0.95;
                OD_last_stepsize = step_size;
                OD_stepsize=1.2*step_size;
                OD_RK_RS++;
            }

            // if (OD_position+OD_stepsize>t_F) OD_stepsize=t_F-OD_position;
            // printf("-->PARTITION! (num %d), OD_stepsize=%le\n", od, OD_stepsize);
        }
        //END 





        //PARTITION BY PERIOD AND MAGNITUDE WITH TIME TRAVEL
        // // printf("t=%.15lf (step_size=%.15le)\n", t+step_size, step_size);
        // if (OD_partition_bool){
        //     printf("OD: position=%.15lf (step_size=%.15le)\n", OD_position+OD_stepsize, OD_stepsize);

        //     OD_partition_bool=0;

        //     /*OD_max_magnitude=0; OD_min_magnitude=1e+200;
        //     for (jj=0; jj<JJ; jj++){
        //         for (cc=0; cc<JJ; cc++){
        //             OD_temp_double = fabs(MY_JET_DATA(tmp_2[jj], cc+1));
        //             if (OD_temp_double>OD_max_magnitude) OD_max_magnitude = OD_temp_double;
        //             if (OD_temp_double<OD_min_magnitude) OD_min_magnitude = OD_temp_double;
        //         }
        //     }*/

        //     OD_max_magnitude=0; OD_min_magnitude=1e+200;
        //     for (jj=0; jj<JJ; jj++){
        //         for (cc=0; cc<JJ; cc++){
        //             OD_temp_double[cc+jj*JJ] = fabs(MY_JET_DATA(tmp_2[jj], cc+1))/OD_magnitude[cc+jj*JJ];
        //             if (OD_temp_double[cc+jj*JJ]>OD_max_magnitude) OD_max_magnitude = OD_temp_double[cc+jj*JJ];
        //             if (OD_temp_double[cc+jj*JJ]<OD_min_magnitude) OD_min_magnitude = OD_temp_double[cc+jj*JJ];
        //         }
        //     }
        //     printf("\tMAGNITUDES: MAX=%le, MIN=%le, RATIO=%le\n", OD_max_magnitude, OD_min_magnitude, (OD_max_magnitude/OD_min_magnitude)/OD_TOL);

        //     for (jj=0; jj<JJ && !OD_magnitude_condition_bool; jj++){
        //         for (cc=0; cc<JJ && !OD_magnitude_condition_bool; cc++){
        //             OD_magnitude_condition_bool = (OD_temp_double[cc+jj*JJ] > OD_MAXTOL) || (OD_temp_double[cc+jj*JJ] < OD_MAXTOL_inv);
        //         }
        //     }


        //     // if (OD_max_magnitude>OD_MAXTOL || OD_min_magnitude<OD_MAXTOL_inv){
        //     if (OD_magnitude_condition_bool){
        //         OD_magnitude_condition_bool = 0;

        //         printf("\tOD deceleration (%d)", OD_RS+1);
        //         for (jj=0; jj<JJ; jj++) {
        //             for (cc=0; cc<CC; cc++) MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(OD_last_jet[jj], cc); 
        //         }

        //         t = OD_last_t;
        //         step_size = OD_last_stepsize;
        //         OD_position = OD_last_OD_position;

        //         OD_stepsize*=0.9;
        //         if (OD_stepsize<step_size){ 
        //             printf("\n\tOD stepsize reached RK stepsize!!\n\t"); 
        //             step_size*=0.95;
        //             OD_last_stepsize = step_size;
        //             OD_stepsize=1.2*step_size;

        //         }
        //         OD_RS++;

        //         if (od==0){ OD_first_magnitude_bool=1; }

        //         printf(" --> stepsize=%le\n", OD_stepsize);

        //         continue;
        //     }

        //     for (jj=0; jj<JJ && !OD_magnitude_condition_bool; jj++){
        //         for (cc=0; cc<JJ && !OD_magnitude_condition_bool; cc++){
        //             OD_magnitude_condition_bool = (OD_temp_double[cc+jj*JJ] > OD_TOL) || (OD_temp_double[cc+jj*JJ] < OD_TOL_inv);
        //         }
        //     }       

        //     // if (OD_max_magnitude>OD_TOL || OD_min_magnitude<OD_TOL_inv){
        //     if (OD_magnitude_condition_bool){
        //         OD_magnitude_condition_bool = 0;

        //         for (jj=0; jj<JJ; jj++){ for (cc=0; cc<JJ; cc++){ OD_magnitude[cc+jj*JJ] = fabs(MY_JET_DATA(tmp_2[jj], cc+1)); }}

        //         printf("-->PARTITION! (num %d)\n", od+1);
        //         for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){  MY_JET_DATA(OD_last_jet[jj], cc) = MY_JET_DATA(tmp_2[jj], cc); }}
        //         OD_last_t = t+step_size;
        //         OD_last_stepsize = step_size;
        //         OD_last_OD_position = OD_position+OD_stepsize;

        //         for (jj=0; jj<JJ; jj++){
        //             for (cc=0; cc<JJ; cc++){
        //                 OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(tmp_2[jj], cc+1);

        //                 if (jj==cc) MY_JET_DATA(tmp_2[jj], cc+1) = 1.0;
        //                 else MY_JET_DATA(tmp_2[jj], cc+1) = 0.0;
        //             }
        //         }

        //         fprintf(OD_part_dat,"%.15lf\n", OD_last_t);

        //         od++;

        //     } else {
        //         OD_magnitude_condition_bool = (OD_max_magnitude < OD_MINTOL) && (OD_min_magnitude > OD_MINTOL_inv);
        //     }

        //     OD_position+=OD_stepsize;      
            
        //     // if (OD_max_magnitude<OD_MINTOL && OD_min_magnitude>OD_MINTOL_inv){
        //     if(OD_magnitude_condition_bool){
        //         OD_magnitude_condition_bool = 0;

        //         OD_stepsize*=1.1;
        //         if (OD_position+OD_stepsize>t_F) OD_stepsize=t_F-OD_position;
        //         OD_AS++;

        //         printf("\tOD acceleration (num %d) --> stepsize=%le\n", OD_AS, OD_stepsize);
        //     }

        // }
        //END 





        //PARTITION BY PERIOD AND MAGNITUDE

        /*// printf("od=%d, t=%lf, t+h=%lf, OD_part_true=%d, (od+1)*OD_part=%lf\n", od, t, t+step_size, OD_part_true, (od+1)*OD_part); 
        // printf("od=%d, t=%lf\n", od, t); 

        // printf("kappa_max=%le (%le/%le)\n kappa_min=%le (%le/%le) \n", cond_num_max/last_cond_num_max , cond_num_max, last_cond_num_max,  last_cond_num_min/cond_num_min, last_cond_num_min, cond_num_min);
        if (OD_part_true){

            if (od==0){
                cond_true=1;
            } else{
                jj=0;
                while(cond_true==0 && jj<JJ){
                    cc=0;
                    while(cond_true==0 && cc<JJ){
                        // printf("a=%le, b=%le, a/b=%le, b/a=%le\n", fabs(MY_JET_DATA(tmp_2[jj], cc+1)), cond_num[cc+jj*JJ], fabs(MY_JET_DATA(tmp_2[jj], cc+1))/cond_num[cc+jj*JJ], cond_num[cc+jj*JJ]/fabs(MY_JET_DATA(tmp_2[jj], cc+1)));
                        cond_temp = fabs(MY_JET_DATA(tmp_2[jj], cc+1));
                        cond_true = (cond_temp > cond_OD) || (cond_temp < inv_cond_OD);
                        // cond_true = (cond_temp/cond_num[cc+jj*JJ] > cond_OD) || (cond_num[cc+jj*JJ]/cond_temp > cond_OD);
                        // printf("%le , %le\n", cond_temp/cond_num[cc+jj*JJ], cond_num[cc+jj*JJ]/cond_temp);
                        cc++;
                    }
                    jj++;
                }
            }
            

            if (cond_true==1){

                for (jj=0; jj<JJ; jj++){
                    for (cc=0; cc<JJ; cc++){
                        cond_temp = MY_JET_DATA(tmp_2[jj], cc+1);

                        OD_JAC[od][cc+jj*JJ] = cond_temp;
                        cond_num[cc+jj*JJ] = fabs(cond_temp);

                        if (jj==cc) MY_JET_DATA(tmp_2[jj], cc+1) = 1.0;
                        else MY_JET_DATA(tmp_2[jj], cc+1) = 0.0;
                    }
                }

                fprintf(OD_part_dat,"%.15lf\n", (od+1)*OD_part);

                od++;
                OD_part_true=0;
                cond_true=0;
            }

        }*/
        //END 

        //PARTITION BY MAGNITUDE
        /*cond_num_max=0; cond_num_min=1e+200;
        for (jj=0; jj<JJ; jj++){
            for (cc=0; cc<JJ; cc++){
                cond_temp = fabs(MY_JET_DATA(tmp_2[jj], cc+1));

                if (cond_temp>cond_num_max) cond_num_max=cond_temp;
                if (cond_temp<cond_num_min) cond_num_min=cond_temp;
            }
        }

        // printf("kappa_max=%le (%le/%le)\n kappa_min=%le (%le/%le) \n", cond_num_max/last_cond_num_max , cond_num_max, last_cond_num_max,  last_cond_num_min/cond_num_min, last_cond_num_min, cond_num_min);
        if (!OD_part_true){

            if (cond_num_max/last_cond_num_max>cond_OD || last_cond_num_min/cond_num_min>cond_OD || last_cond_num_max/cond_num_max>cond_OD || cond_num_min/last_cond_num_min>cond_OD){

                for (jj=0; jj<JJ; jj++){
                    for (cc=0; cc<JJ; cc++){
                        OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(tmp_2[jj], cc+1);
                        if (jj==cc) MY_JET_DATA(tmp_2[jj], cc+1) = 1.0;
                        else MY_JET_DATA(tmp_2[jj], cc+1) = 0.0;
                    }
                }

                // printf("kappa_max=%le, kappa_min=%le \n", cond_num_max/last_cond_num_max,  last_cond_num_min/cond_num_min);

                fprintf(OD_part_dat,"%.15lf\n", t);

                od++;

                last_cond_num_max = cond_num_max;
                last_cond_num_min = cond_num_min;

                OD_part_true=1;
            }

        } else {
            OD_part_true=0;
            last_cond_num_max = cond_num_max;
            last_cond_num_min = cond_num_min;
        }*/
        //END

        // for (jj=0; jj<JJ; jj++){
        //     for (cc=0; cc<JJ; cc++){
        //         last_OD_JAC[cc+jj*JJ] = MY_JET_DATA(tmp_2[jj], cc+1);
        //     }
        // }
        // condition_num = condition(last_OD_JAC);
        // printf("k(A_%d)=%le, t=%f\n", od+1, condition_num, t);

        
        // if (condition_num<1e-10){
        // if (condition_num>1e+8){

        //     condition_num = last_cond_num;
        //     // printf("k(A_%d)=%le, t=%f\n", od+1, condition_num, t);
        //     if (condition_num>max_cond_num) max_cond_num = condition_num;
        //     else if (condition_num<min_cond_num) min_cond_num = condition_num;

        //     for (jj=0; jj<JJ; jj++){
        //         for (cc=0; cc<JJ; cc++){
        //             if (jj==cc) MY_JET_DATA(jet_xx[jj], cc+1) = 1.0;
        //             else MY_JET_DATA(jet_xx[jj], cc+1) = 0.0;
        //         }
        //     }

        //     fprintf(OD_part_dat,"%.15lf\n", last_t_part);

        //     od++;


        //     if (od>=OD_MAX){ printf("CAUTION, OD_MAX<od. Set OD_MAX=%d and try again.\n", od+1); exit(0);}

        //     continue;

        // }     

        //PARTITION BY PERIOD FRACTION ...
        /*if (OD_part_true){
            for (jj=0; jj<JJ; jj++){
                for (cc=0; cc<JJ; cc++){
                    OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(tmp_2[jj], cc+1);
                    if (jj==cc) MY_JET_DATA(tmp_2[jj], cc+1) = 1.0;
                    else MY_JET_DATA(tmp_2[jj], cc+1) = 0.0;
                }
            }

            fprintf(OD_part_dat,"%.15lf\n", (od+1)*OD_part);

            od++;

            OD_part_true=0;
        }  */

        if (od>=OD_MAX){ printf("CAUTION, OD_MAX<od. Increase OD_MAX and try again.\n"); exit(0);} 

        for (jj=0; jj<JJ; jj++){ for (cc=0; cc<CC; cc++){ MY_JET_DATA(jet_xx[jj], cc) = MY_JET_DATA(tmp_2[jj], cc); }}

        t+=step_size;

        AS++; RK_it++;
        // fprintf(h_temp, "%lf %lf %d\n", t, step_size, 1);
        // printf("ACCEPTED h=%le", step_size);
        if (last_rejected==0){
            step_size *= fmin(1.5, new_step_size);
            // printf("--> NEW h=%le\n", step_size);
        } else {
            step_size *= fmin(1, new_step_size);
            // printf("--> NEW h=%le\n", step_size);
            last_rejected=0;
        }

        // printf("t=%.15lf, step_size=%.15le, OD_position=%.15lf, OD_stepsize=%.15le\n", t, step_size, OD_position, OD_stepsize);        
        // printf("t+step_size=%.15lf, OD_position+OD_stepsize=%.15lf\n", t+step_size, OD_position+OD_stepsize);


        //PARTITION BY PERIOD AND MAGNITUDE WITH TIME TRAVEL
        if (t+step_size>=OD_position+OD_stepsize){

            step_size=(OD_position+OD_stepsize)-t;
            OD_partition_bool=1;

        }

        //END

        //PARTITION BY PERIOD FRACTION
        /*if (t+step_size>=(OD_frac_od+1)*OD_part && OD_frac_od+1<OD_part_num){

            step_size=(OD_frac_od+1)*OD_part-t;
            OD_part_true=1;
            OD_frac_od++;

        }*/
        //END

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
        
        // Check if last step
        if (t+step_size>t_F){
            last_step_size = step_size;
            step_size = t_F-t;
            OD_partition_bool = 0;
        } else {
            if (step_size>h_max) h_max=step_size;
            if (step_size<h_min) h_min=step_size;
        }

    }

    fprintf(OD_part_dat,"%.15lf\n", t);
    for (jj=0; jj<JJ; jj++){
        for (cc=0; cc<JJ; cc++){
            OD_JAC[od][cc+jj*JJ] = MY_JET_DATA(jet_xx[jj], cc+1);
            if (jj==cc) MY_JET_DATA(jet_xx[jj], cc+1) = 1.0;
            else MY_JET_DATA(jet_xx[jj], cc+1) = 0.0;
        }
    }
    od++;

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

    for (i=0; i<od; i++){
        for (jj=0; jj<JJ; jj++){
            for(cc=0; cc<JJ; cc++){
                fprintf(OD_A_dat, "%.10le ", OD_JAC[i][cc+jj*JJ]);
            }
            fprintf(OD_A_dat, "\n");
        }
        fprintf(OD_A_dat, "\n");
    }

    // double condition_num, max_cond_num=0, min_cond_num=1e100;

    // for (i=0; i<od; i++){
    //     printf("A_%d\n", i+1);

    //     for (jj=0; jj<JJ; jj++){
    //         for (cc=0; cc<JJ; cc++){
    //             printf("%.15le ", OD_JAC[i][cc+jj*JJ]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");

    //     condition_num = condition(OD_JAC[i]);
    //     printf("k(A_%d)=%le\n\n", i+1, condition_num);

    //     if (condition_num>max_cond_num) max_cond_num = condition_num;
    //     else if (condition_num<min_cond_num) min_cond_num = condition_num;

    // }

    double **OD_Q;
    int PSD_partition;

    if (store_Q){

        OD_Q = malloc((od+3)*sizeof(double *));
        for (i=0; i<od; i++) {
            OD_Q[i] = malloc(JJ*JJ*sizeof(double));
            for (jj=0; jj<JJ; jj++){
                for (cc=0; cc<JJ; cc++){
                    if (jj==cc) OD_Q[i][cc+jj*JJ] = 1.0;
                    else  OD_Q[i][cc+jj*JJ] = 0.0;
                }
            }
        }

        PHTD(OD_JAC, OD_Q, od);

        PSD_partition = PSD(OD_JAC, OD_Q, od, 0, JJ);

        if (PSD_partition == 0) PSD(OD_JAC, OD_Q, od, 1, JJ);
        else if (PSD_partition == 1) PSD(OD_JAC, OD_Q, od, 0, JJ-1);

    } else {

        PHTD(OD_JAC, NULL, od);

        PSD_partition = PSD(OD_JAC, NULL, od, 0, JJ);

        if (PSD_partition == 0) PSD(OD_JAC, NULL, od, 1, JJ);
        else if (PSD_partition == 1) PSD(OD_JAC, NULL, od, 0, JJ-1);

    }

    // printf("max k(A_i)= %le, min k(A_i)=%le\n", max_cond_num, min_cond_num);

    printf("Num of OD:%d, with %d decelerations (%d reached RK stepsize)\n", od, OD_RS, OD_RK_RS);

    double WR[JJ], WI[JJ], W_sign[JJ];
    for (jj=0; jj<JJ; jj++) W_sign[jj]=1.0;

    for (jj=0; jj<JJ; jj++){
        WR[jj]=0.0;
        int W_zero_factors=0;
        for (cc=0; cc<od; cc++){
            if (fabs(OD_JAC[cc][jj+jj*JJ])>1e-200) WR[jj]+=log10(fabs(OD_JAC[cc][jj+jj*JJ]));
            else W_zero_factors++; 
            // printf("eps_(%d,%d)=%le, PI_(%d,%d)=1e%lf\n", cc+1, jj+1, OD_JAC[cc][jj+jj*JJ], cc+1, jj+1, WR[jj]);
            if (OD_JAC[cc][jj+jj*JJ]<0) W_sign[jj]*=-1;
        } 
        if (W_zero_factors>0) printf("WARNING, lambda_%d has %d zero factors (discarded)\n", jj+1, W_zero_factors);
    }

    for (jj=0; jj<JJ; jj++){
        if (W_sign[jj]>0) printf("lambda_%d = 1e%.10lf \n", jj+1, WR[jj]);
        else printf("lambda_%d = -1e%.10lf \n", jj+1, WR[jj]);
    } 
    // for (jj=0; jj<JJ; jj++) printf("lambda_%d = %le + i %le\n", jj+1, WR[jj], WI[jj]);


    // double norm_OD_JAC[od], norm_WR[JJ], norm_WI[JJ], norm_temp;

    // double EV[JJ-1][JJ], last_EV[JJ], norm_EV, norm_prod_exp_EV[JJ-1], error_EV;

    // for (ss=0; ss<JJ-1; ss++){
    //     int it_EV=0, sign_EV;

    //     for (jj=0; jj<JJ; jj++) {
    //         if (jj<ss+2) EV[ss][jj]=1.0;
    //         else EV[ss][jj]=0.0;
    //     }

    //     lapack_int LAPACK_info, *LAPACK_ipiv;
    //     // LAPACK_ipiv = (lapack_int *)malloc(JJ*sizeof(lapack_int)) ;

    //     do{

    //         norm_prod_exp_EV[ss]=0.0;

    //         for (i=od-1; i>=0; i--){

    //             // LAPACK_info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, JJ, 1, OD_JAC[i], JJ, LAPACK_ipiv, EV[ss], 1);
    //             LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_JAC[i], JJ, EV[ss], 1);
    //             if (LAPACK_info != 0) { exit(0); } 

    //             norm_EV = 0.0;
    //             for (jj=0; jj<JJ; jj++) norm_EV+=EV[ss][jj]*EV[ss][jj];
    //             norm_EV = sqrt(norm_EV);

    //             sign_EV=EV[ss][0]/fabs(EV[ss][0]);
    //             for (jj=0; jj<JJ; jj++) EV[ss][jj]/=(sign_EV*norm_EV);

    //             norm_prod_exp_EV[ss] += log10(fabs(norm_EV));
    //         }
    //         printf("E_%d: ", ss+2);
    //         for (jj=0; jj<JJ; jj++) printf("%.15le ", EV[ss][jj]);
    //         printf(", ||E_%d||=1e%lf\n", ss+2, norm_prod_exp_EV[ss]);

    //         error_EV = 0.0;
    //         if (it_EV>0){
    //             for (jj=0; jj<JJ; jj++) error_EV+=fabs(EV[ss][jj]-last_EV[jj]);
    //             printf("error E_%d=%le\n", ss+2, error_EV);
    //         }
    //         for (jj=0; jj<JJ; jj++){
    //             last_EV[jj]=EV[ss][jj];
    //         }

    //         it_EV++;

    //     }while(it_EV<2 || error_EV>EV_TOL);
    // }

    // // free(LAPACK_ipiv);

    // int ind_IEV = 1;
    // double IEV[JJ], IEV_RHS[JJ], norm_IEV, norm_IEV_RHS, norm_prod_exp_IEV, norm_prod_exp_IEV_RHS;
    // IEV_RHS[0]=3.900925780829266e-02;
    // IEV_RHS[1]=-1.142419429875029e-03;
    // IEV_RHS[2]=2.362745844768107e-02;

    // for (jj=0; jj<JJ; jj++) IEV[jj]=EV[ind_IEV][jj];

    // lapack_int LAPACK_info, *LAPACK_ipiv;
    // // LAPACK_ipiv = (lapack_int *)malloc(JJ*sizeof(lapack_int)) ;

    // for (i=od-1; i>=0; i--){

    //     // LAPACK_info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, JJ, 1, OD_JAC[i], JJ, LAPACK_ipiv, EV[ss], 1);
    //     LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_JAC[i], JJ, IEV, 1);
    //     if (LAPACK_info != 0) { printf("IEV LAPACK error %d", LAPACK_info); exit(0); } 

    //     LAPACK_info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', JJ, 1, OD_JAC[i], JJ, IEV_RHS, 1);
    //     if (LAPACK_info != 0) { printf("IEV_RHS LAPACK error %d", LAPACK_info); exit(0); }

    //     norm_IEV = 0.0; norm_IEV_RHS=0.0;
    //     for (jj=0; jj<JJ; jj++) {
    //         norm_IEV+=IEV[jj]*IEV[jj];
    //         norm_IEV_RHS+=IEV_RHS[jj]*IEV_RHS[jj];
    //     }
    //     norm_IEV = sqrt(norm_IEV);
    //     norm_IEV_RHS = sqrt(norm_IEV_RHS);

    //     for (jj=0; jj<JJ; jj++){
    //         IEV[jj]/=norm_IEV;
    //         IEV_RHS[jj]/=norm_IEV_RHS;
    //     } 

    //     // printf("IEV: ");
    //     // for (jj=0; jj<JJ; jj++) printf("%le ", IEV[jj]);
    //     // printf("\n");

    //     // printf("IEV_RHS: ");
    //     // for (jj=0; jj<JJ; jj++) printf("%le ", IEV_RHS[jj]);
    //     // printf("\n");

    //     norm_prod_exp_IEV += log10(fabs(norm_IEV));
    //     norm_prod_exp_IEV_RHS += log10(fabs(norm_IEV_RHS));
    // }

    // printf("IE_%d:\n", ind_IEV);
    // for (jj=0; jj<JJ; jj++) printf("1e%lf 1e%lf %le + 1e%lf %le\n",2*WR[ind_IEV], norm_prod_exp_IEV, IEV[jj], norm_prod_exp_IEV_RHS, IEV_RHS[jj]);

    clock_t CLOCK_end = clock();
    printf("\nExecution time: %lf seconds\n\n\n", (double)(CLOCK_end-CLOCK_begin)/CLOCKS_PER_SEC);

    // printf("STORE OUTPUTS ...");
    for (i=0; i<od; i++){
        for (jj=0; jj<JJ; jj++){
            for(cc=0; cc<JJ; cc++){
                // fprintf(OD_R_dat, "%.10le ", OD_JAC[i][cc+jj*JJ]);
                // fprintf(OD_Q_dat, "%.10le ", OD_Q[i][cc+jj*JJ]);
            }
            // fprintf(OD_R_dat, "\n");
            // fprintf(OD_Q_dat, "\n");
            fprintf(OD_EV_dat, "%.10le ", OD_JAC[i][jj+jj*JJ]);
        }
        // fprintf(OD_R_dat, "\n");
        // fprintf(OD_Q_dat, "\n");
        fprintf(OD_EV_dat, "\n");
    }
    // printf(" ok\n");

    free(OD_JAC);
    if (store_Q) free(OD_Q);
    return 0;
}