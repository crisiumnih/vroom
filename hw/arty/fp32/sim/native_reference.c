/* Cross-check the independent integer oracle against native binary32 on this
 * host. This program is test-only, never synthesized. NaNs are canonicalized.
 * Build with contraction, fast math and constant folding of arithmetic disabled.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <fenv.h>
#include <math.h>
_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24,"requires binary32");
#pragma STDC FENV_ACCESS ON
static uint32_t calculate(uint32_t a, uint32_t b, int mul, unsigned *flags) {
    float x,y; memcpy(&x,&a,4); memcpy(&y,&b,4);
    volatile float vx=x, vy=y;
    feclearexcept(FE_ALL_EXCEPT);
    volatile float z=mul ? vx*vy : vx+vy;
    int f=fetestexcept(FE_ALL_EXCEPT);
    *flags=((f&FE_INVALID)?16:0)|((f&FE_DIVBYZERO)?8:0)|((f&FE_OVERFLOW)?4:0)|
           ((f&FE_UNDERFLOW)?2:0)|((f&FE_INEXACT)?1:0);
    float result=z; uint32_t bits; memcpy(&bits,&result,4);
    if ((bits&0x7fffffff)>0x7f800000) bits=0x7fc00000;
    return bits;
}
int main(int argc, char **argv) {
    if(argc!=2) return 2;
    FILE *f=fopen(argv[1],"r"); if(!f) return 2;
    if(fesetenv(FE_DFL_ENV)||fesetround(FE_TONEAREST)) return 2;
    unsigned a,b,add,af,mul,mf,flags; unsigned long n=0, tininess_differences=0;
    while(fscanf(f,"%x%x%x%x%x%x",&a,&b,&add,&af,&mul,&mf)==6) {
        uint32_t r=calculate(a,b,0,&flags);
        if(r!=add||flags!=af) {printf("ADD mismatch row %lu %08x %08x got %08x/%x expected %08x/%x\n",n,a,b,r,flags,add,af);return 1;}
        r=calculate(a,b,1,&flags);
        /* Some hosts detect tininess before rounding. Permit only this precise
         * policy difference: exact tiny product rounds up to minimum normal.
         * double represents any binary32 product exactly (48 significant bits).
         */
        float x,y; memcpy(&x,&a,4); memcpy(&y,&b,4);
        if(r==mul && (r&0x7fffffff)==0x00800000 && mf==1 && flags==3 &&
           fabs((double)x*(double)y)<0x1p-126) {
            tininess_differences++; flags=mf;
        }
        if(r!=mul||flags!=mf) {printf("MUL mismatch row %lu %08x %08x got %08x/%x expected %08x/%x\n",n,a,b,r,flags,mul,mf);return 1;}
        n++;
    }
    fclose(f); printf("PASS native cross-check: %lu operations and flags\n",2*n);
    printf("Tininess-before versus after rounding differences: %lu\n",tininess_differences); return 0;
}
