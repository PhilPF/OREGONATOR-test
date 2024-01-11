/* ODE specification:  Oregonator */

extern MY_FLOAT s,w,q,fff;

x' = s*(y-x*y+x-q*x*x);
y' = (-y-x*y+fff*z)/s;
z' = w*(x-z);

jet x,y,z symbols 12 deg 1;