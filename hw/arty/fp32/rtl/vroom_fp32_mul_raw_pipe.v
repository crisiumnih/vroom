
/*============================================================================

This Verilog source file is part of the Berkeley HardFloat IEEE Floating-Point
Arithmetic Package, Release 1, by John R. Hauser.

Copyright 2019 The Regents of the University of California.  All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

// Vroom derivative: fixed binary32/RNE/canonical-NaN raw multiply with two
// enabled stages. Split 24x24 into 24x17 and 24x7 registered partial products
// to avoid a combinational DSP cascade. Full 48-bit product retained before
// sticky reduction, exactly as mulRecFNToRaw. No numerical precision is lost.
`default_nettype none
module vroom_fp32_mul_raw_pipe (
    input wire clk, enable,
    input wire [32:0] a,b,
    output reg [4:0] special,
    output reg signed [9:0] exponent,
    output reg [26:0] significand
);
    wire nan_a,inf_a,zero_a,sign_a,snan_a;
    wire nan_b,inf_b,zero_b,sign_b,snan_b;
    wire signed [9:0] exp_a,exp_b;
    wire [24:0] sig_a,sig_b;
    recFNToRawFN #(8,24) decode_a(a,nan_a,inf_a,zero_a,sign_a,exp_a,sig_a);
    recFNToRawFN #(8,24) decode_b(b,nan_b,inf_b,zero_b,sign_b,exp_b,sig_b);
    isSigNaNRecFN #(8,24) signaling_a(a,snan_a);
    isSigNaNRecFN #(8,24) signaling_b(b,snan_b);
    reg [40:0] low_product;
    reg [30:0] high_product;
    reg [4:0] special_q;
    reg signed [9:0] exponent_q;
    wire [47:0] product = {7'b0,low_product} + {high_product,17'b0};
    // recFNToRawFN guarantees bit 24 is zero for binary32 significands.
    wire unused_sig_top = sig_a[24] | sig_b[24];
    always @(posedge clk) if (enable) begin
        low_product <= sig_a[23:0] * sig_b[16:0];
        high_product <= sig_a[23:0] * sig_b[23:17];
        special_q <= {snan_a || snan_b || (inf_a && zero_b) || (zero_a && inf_b),
                      nan_a || nan_b,inf_a || inf_b,zero_a || zero_b,sign_a ^ sign_b};
        exponent_q <= exp_a + exp_b - 10'sd256;
        special <= special_q; exponent <= exponent_q;
        significand <= {product[47:22],|product[21:0]};
    end
endmodule
`default_nettype wire
