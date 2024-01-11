#ifndef RK_RHO_H
#define RK_RHO_H

#include "method.h"
#include "taylor_rho.h"

#define RKS _RKS_
#define JJ _NUMBER_OF_JET_VARS_

void Z_rho(MY_FLOAT rho_DF[RKS*JJ][JJ], MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *state_jet_xx, MY_FLOAT state_stages_in[RKS][JJ], int *init_flag);

// void rho(MY_FLOAT rho_DF[RKS*JJ][JJ], MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *state_jet_xx, MY_FLOAT state_stages_in[RKS][JJ]);

#endif /* RK_RHO_H */