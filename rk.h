#ifndef RK_H
#define RK_H

#include "taylor.h"

void FLOAT_function(MY_FLOAT *out, MY_FLOAT t, MY_FLOAT *xx);

int RK_Implicit(MY_JET *jet_out, MY_FLOAT step_size, MY_FLOAT t, MY_JET *jet_xx);

#endif /* RK_H */