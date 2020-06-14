#include <math.h>
#include <stdio.h>

int main(){
    double re = -1;
    double im = 0;

    double arg = atan2(im, re);

    printf("%f\n", arg);
}