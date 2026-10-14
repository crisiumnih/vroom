"""Shift/add CORDIC development model and independent high-precision tanh oracle.
Binary32 I/O, 58-bit signed coordinates/angle with 52 fractional bits internally.
The integer model is for RTL equivalence; Decimal tanh is the accuracy reference.
"""
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import struct

FRAC=52
WIDTH=58
ONE=1<<FRAC
SHIFTS=[j for i in range(1,53) for j in ([i,i] if i in (4,13,40) else [i])]
TINY=0x39000000  # |a| <= 2^-13 rounds tanh(a) back to a.
SAT=0x41200000   # |a| >= 10 rounds tanh(a) to signed one.


def constants():
    with localcontext() as ctx:
        ctx.prec=100
        scale=Decimal(ONE)
        angles=[];gain=Decimal(1)
        for i in SHIFTS:
            s=Decimal(2)**(-i)
            angles.append(int((((1+s)/(1-s)).ln()/2*scale).to_integral_value(rounding=ROUND_HALF_EVEN)))
            gain *= (1-s*s).sqrt()
        initial=int((scale/gain).to_integral_value(rounding=ROUND_HALF_EVEN))
        ln2=int((Decimal(2).ln()*scale).to_integral_value(rounding=ROUND_HALF_EVEN))
    return initial,ln2,angles


INITIAL,LN2,ANGLES=constants()


def value(a):
    sign=a>>31;e=(a>>23)&255;f=a&0x7fffff
    n=f if e==0 else f|0x800000
    with localcontext() as ctx:
        ctx.prec=180
        x=Decimal(n)*(Decimal(2)**(-149 if e==0 else e-150))
    return x.copy_negate() if sign else x


def round_ratio(n,d):
    """RNE a nonnegative exact rational in [0,1] directly to binary32."""
    if not n:return 0
    e=n.bit_length()-d.bit_length()
    if (n<d<<e) if e>=0 else (n<<-e)<d:e-=1
    shift=149 if e < -126 else 23-e
    q,r=divmod(n<<shift,d)
    q+=int(2*r>d or (2*r==d and q&1))
    if e < -126:return q
    if q==(1<<24):q>>=1;e+=1
    return ((e+127)<<23)|(q&0x7fffff)


def reference(a,precision=120):
    sign=a&0x80000000;mag=a&0x7fffffff
    if mag>0x7f800000:return 0x7fc00000,int(not(a&0x400000))
    if mag==0x7f800000:return sign|0x3f800000,0
    if mag==0:return a,0
    with localcontext() as ctx:
        ctx.prec=precision
        # For huge arguments, rounding to one follows from tanh(10), not an
        # enormous exp operation; tiny values are evaluated, not shortcut here.
        x=abs(value(a))
        if x>=10:return sign|0x3f800000,0
        z=(2*x).exp()
        y=(z-1)/(z+1)
        n,d=y.as_integer_ratio()
        return sign|round_ratio(n,d),0


def evaluate(a):
    sign=a&0x80000000;mag=a&0x7fffffff
    if mag>0x7f800000:return 0x7fc00000,int(not(a&0x400000)),1,'nan'
    if mag>=SAT:return sign|0x3f800000,0,1,'saturated'
    if mag<=TINY:return a,0,1,'tiny'
    exp=(mag>>23)&255;sig=(mag&0x7fffff)|0x800000
    z=sig<<(exp-97) # exactly 2*abs(a) in Q52 for this regular domain
    k=0
    while z>=LN2:z-=LN2;k+=1
    x,y=INITIAL,0
    for shift,angle in zip(SHIFTS,ANGLES):
        xs,ys=x>>shift,y>>shift
        if z>=0:x,y,z=x+ys,y+xs,z-angle
        else:x,y,z=x-ys,y-xs,z+angle
    e=(x-y)>>k
    num,den=ONE-e,ONE+e
    assert 0<num<den and 0<e<ONE
    q=(num<<FRAC)//den
    result=sign|round_ratio(q,ONE)
    normal_shifts=FRAC-q.bit_length()
    # Request edge 0; convert 1, range k+1, 2*55 rotations, difference,
    # scale, numerator/denominator, 52 quotient bits, normalize shifts+1,
    # pack, response transfer. Derived independently from the RTL event trace.
    latency=1+(k+1)+2*len(SHIFTS)+3+FRAC+(normal_shifts+1)+2+1
    return result,0,latency,'cordic'


def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
