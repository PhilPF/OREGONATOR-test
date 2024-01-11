#ifndef RK_SIGMA_H
#define RK_SIGMA_H

#include "method.h"
#include "taylor_sigma.h"

#define RKS _RKS_
#define JJ _NUMBER_OF_JET_VARS_

void Z_sigma(MY_FLOAT *sigma_F, MY_FLOAT sigma_DF[RKS*JJ][RKS*JJ], MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *state_jet_xx, MY_FLOAT state_stages_in[RKS][JJ], int *init_flag);

// void sigma(MY_FLOAT *sigma_F, MY_FLOAT sigma_DF[RKS*JJ][RKS*JJ], MY_FLOAT step_size, MY_FLOAT t, MY_FLOAT *state_jet_xx, MY_FLOAT state_stages_in[RKS][JJ]);

#endif /* RK_SIGMA_H */