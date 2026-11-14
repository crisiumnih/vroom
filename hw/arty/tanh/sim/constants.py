#!/usr/bin/env python3
"""Generate/check embedded CORDIC constants; vendor files are never modified."""
import argparse
from pathlib import Path
from model import INITIAL,LN2,ANGLES,SHIFTS


def block():
    lines=[f"    localparam signed [57:0] INITIAL=58'h{INITIAL:015x};",
           f"    localparam signed [57:0] LN2=58'h{LN2:015x};",
           '    function [5:0] shift_index;', '        input [5:0] step;', '        begin case(step)']
    lines += [f"            6'd{j}: shift_index=6'd{i};" for j,i in enumerate(SHIFTS)]
    lines += ["            default: shift_index=6'd52;",'        endcase end','    endfunction',
              '    function signed [57:0] angle;', '        input [5:0] step;', '        begin case(step)']
    lines += [f"            6'd{j}: angle=58'h{v:015x};" for j,v in enumerate(ANGLES)]
    lines += ["            default: angle=58'd1;",'        endcase end','    endfunction']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    path=Path(__file__).resolve().parents[1]/'rtl/vroom_fp32_tanh.v'
    text=path.read_text();start=text.index('    // CONSTANTS_BEGIN\n')+len('    // CONSTANTS_BEGIN\n');end=text.index('    // CONSTANTS_END')
    if args.write:path.write_text(text[:start]+block()+text[end:])
    else:assert text[start:end]==block(),'CORDIC constants differ; regenerate explicitly'
    print('CORDIC constants '+('written' if args.write else 'verified'))
