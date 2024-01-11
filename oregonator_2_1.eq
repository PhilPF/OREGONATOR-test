/* ODE specification:  Oregonator */

extern MY_FLOAT s,w,q;

x' = s*(y-x*y+x-q*x*x);
y' = (-y-x*y+z)/s;
z' = w*(x-z);

jet x,y,z symbols 2 deg 1;