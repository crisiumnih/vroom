"""Exact-integer binary32 add/multiply oracle, RNE, tininess after rounding.

No host floating-point arithmetic is used by operate/round_exact. Finite values
are signed integers times powers of two; Python integers retain every bit until
one explicit rounding. Flags use the HardFloat invalid/div0/overflow/underflow/
inexact order. Every NaN result is canonical 0x7fc00000.
"""
import random
import struct


def parts(bits):
    return bits >> 31, (bits >> 23) & 255, bits & 0x7fffff


def finite(bits):
    sign, exponent, fraction = parts(bits)
    mantissa = fraction if exponent == 0 else fraction | 0x800000
    return -mantissa if sign else mantissa, -149 if exponent == 0 else exponent-150


def round_exact(value, exponent, zero_sign=0):
    if value == 0:
        return zero_sign << 31, 0
    sign = int(value < 0)
    value = abs(value)
    # IEEE tininess-after is evaluated at 24-bit precision with an UNBOUNDED
    # exponent, before subnormal encoding. Rounding to min-normal can still
    # underflow. See Hauser's SoftFloat FAQ, underflow-threshold example.
    precision_shift = max(value.bit_length()-24, 0)
    precise, tail = divmod(value, 1 << precision_shift)
    if precision_shift:
        half = 1 << (precision_shift-1)
        precise += int(tail > half or (tail == half and precise & 1))
    tiny = exponent+precision_shift+precise.bit_length()-1 < -126
    quantum = max(exponent+value.bit_length()-24, -149)
    shift = quantum-exponent
    if shift > 0:
        rounded, remainder = divmod(value, 1 << shift)
        half = 1 << (shift-1)
        rounded += int(remainder > half or (remainder == half and rounded & 1))
        inexact = int(remainder != 0)
    else:
        rounded, inexact = value << -shift, 0
    flags = inexact | (2 if tiny and inexact else 0)
    highest = quantum+rounded.bit_length()-1 if rounded else -150
    if highest > 127:
        return (sign << 31) | 0x7f800000, 5
    if highest < -126:
        return (sign << 31) | rounded, flags
    adjust = highest-23-quantum
    significand = rounded >> adjust if adjust >= 0 else rounded << -adjust
    return (sign << 31) | ((highest+127) << 23) | (significand & 0x7fffff), flags


def operate(a, b, multiply=False):
    sa, ea, fa = parts(a)
    sb, eb, fb = parts(b)
    nan_a, nan_b = ea == 255 and fa != 0, eb == 255 and fb != 0
    invalid = (nan_a and not fa & 0x400000) or (nan_b and not fb & 0x400000)
    inf_a, inf_b = ea == 255 and fa == 0, eb == 255 and fb == 0
    zero_a, zero_b = ea == 0 and fa == 0, eb == 0 and fb == 0
    invalid |= ((inf_a and zero_b) or (inf_b and zero_a)) if multiply else (inf_a and inf_b and sa != sb)
    if nan_a or nan_b or invalid:
        return 0x7fc00000, 16 if invalid else 0
    if inf_a or inf_b:
        sign = sa ^ sb if multiply else (sa if inf_a else sb)
        return (sign << 31) | 0x7f800000, 0
    na, xa = finite(a)
    nb, xb = finite(b)
    if multiply:
        return round_exact(na*nb, xa+xb, sa ^ sb)
    exponent = min(xa, xb)
    return round_exact((na << (xa-exponent)) + (nb << (xb-exponent)), exponent,
                       int(zero_a and zero_b and sa == sb == 1))


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def generate(out, seed, count):
    assert operate(0x00800000, 0x3f7fffff, True) == (0x00800000,3)
    assert operate(0x007ffffe, 0x3f800002, True) == (0x00800000,1)
    assert operate(1, 0x3f000000, True) == (0,3)
    assert operate(1, 0x3f800000, True) == (1,0)
    assert operate(0x7f7fffff, 0x40000000, True) == (0x7f800000,5)
    assert operate(0x3f800000, 0x33800000) == (0x3f800000,1)
    assert operate(0x3f800001, 0x33800000) == (0x3f800002,1)
    assert operate(0x80000000, 0x80000000) == (0x80000000,0)
    rng = random.Random(seed)
    special = [0,1,2,3,0x3fffff,0x400000,0x7ffffe,0x7fffff,0x800000,0x800001,
               0x00800002,0x33800000,0x3effffff,0x3f000000,0x3f000001,0x3f7fffff,
               0x3f800000,0x3f800001,0x3f800002,0x40000000,0x4b000000,0x7f7ffffe,
               0x7f7fffff,0x7f800000,0x7f800001,0x7fbfffff,0x7fc00000,0x7fffffff]
    special += [v | 0x80000000 for v in special]
    pairs = [(a,b) for a in special for b in special]
    for _ in range(count):
        a, b = rng.getrandbits(32), rng.getrandbits(32)
        pairs.append((a,b))
    # Near cancellation and exponent boundaries are rare in uniform bit draws.
    for _ in range(count//4):
        a = rng.randrange(0x00800000, 0x7f7fffff)
        pairs.append((a, (a+rng.choice([-1,0,1])) ^ 0x80000000))
    with (out/'operators.hex').open('w') as hexfile, (out/'operators.txt').open('w') as native:
        for a,b in pairs:
            add, af = operate(a,b)
            mul, mf = operate(a,b,True)
            hexfile.write(f'{a:08x}{b:08x}{add:08x}{af:02x}{mul:08x}{mf:02x}\n')
            native.write(f'{a:08x} {b:08x} {add:08x} {af:02x} {mul:08x} {mf:02x}\n')
    bias = [bits(rng.uniform(-.1,.1)) for _ in range(64)]
    bias[0] = 0xbf800000
    (out/'bias.hex').write_text(''.join(f'{v:08x}\n' for v in bias))
    cases=[]
    for groups in [1,2,16,22,44]:
        state=bias.copy()
        records=[]
        for col in range(256):
            for group in range(groups):
                a,b=bits(rng.uniform(-.25,.25)),bits(rng.uniform(-.9,.9))
                if col == group == 0:
                    a,b=0x3f800001,0x3f7fffff # Unfused result is zero; FMA is not.
                # Keep long finite accumulation prefixes, then exercise flag
                # propagation through queued products, including NaNs/infinities.
                directed = {128:(0x00000001,0x3f000000),
                            129:(0x00800000,0x3f7fffff),
                            130:(0x80000000,0x3f800000),
                            200:(0x7f7fffff,0x40000000),
                            201:(0xff800000,0x3f800000),
                            202:(0x7f800001,0x3f800000),
                            203:(0x00000000,0x7f800000)}
                if group == 0 and col in directed: a,b=directed[col]
                product, mf=operate(a,b,True)
                result, af=operate(state[group],product)
                state[group]=result
                tag=((col & 1023)<<6)|group
                records.append(f'{tag:04x}{a:08x}{b:08x}{result:08x}{af|mf:02x}\n')
        path=out/f'lane{groups}.hex';path.write_text(''.join(records))
        cases.append({'groups':groups,'operations':len(records),'file':path.name})
    return {'seed':seed,'random_pairs':count,'operator_pairs':len(pairs),
            'directed_cross_pairs':len(special)**2,'cancellation_pairs':count//4,
            'lane_cases':cases,'rounding':'nearest-even','tininess':'after rounding',
            'nan':'canonical 7fc00000','reference':'exact Python integer arithmetic'}
