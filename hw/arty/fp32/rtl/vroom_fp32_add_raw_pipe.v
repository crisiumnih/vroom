
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

// Vroom derivative of addRecFNToRaw, fixed binary32, addition, RNE, canonical
// NaN. Five enabled stages: exponent compare; align/subtract; magnitude/add;
// leading-zero count; normalization/special selection. Raw arithmetic and the
// close/far paths preserve HardFloat's algorithm; the upstream rounder follows.
`default_nettype none
module vroom_fp32_add_raw_pipe (
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
    wire eq_sign = sign_a == sign_b;
    wire signed [9:0] difference = exp_a-exp_b;
    wire [9:0] magnitude = difference[9] ? -difference : difference;
    wire max_align = (difference >>> 5 != 0) &&
                     ((difference >>> 5 != -1) || difference[4:0] == 0);
    wire close_path = !eq_sign && !max_align && magnitude[4:0] <= 1;
    // Metadata indices: invalid12, NaN11, Inf10, both-zero9, equal-sign8,
    // signA7, signB6, InfA5, InfB4, close3, far-sign2, negative-diff1, odd-diff0.
    reg [12:0] meta [1:4];
    reg signed [9:0] base_exp [1:4];
    reg [4:0] distance1;
    reg [23:0] sig_a1,sig_b1;
    wire unused_decoded_bits = sig_a[24] | sig_b[24] | (|magnitude[9:5]);

    wire [26:0] close_aligned1 = meta[1][1] ? {3'b0,sig_a1} :
                                meta[1][0] ? {1'b0,sig_a1,2'b0} : {2'b0,sig_a1,1'b0};
    wire [23:0] larger1 = meta[1][1] ? sig_b1 : sig_a1;
    wire [23:0] smaller1 = meta[1][1] ? sig_a1 : sig_b1;
    wire [28:0] main_aligned1 = {smaller1,5'b0} >> distance1;
    wire [6:0] reduced1,mask1;
    compressBy4 #(26) compress_smaller({smaller1,2'b0},reduced1);
    lowMaskHiLo #(3,7,0) sticky_mask(distance1[4:2],mask1);
    reg signed [26:0] close_difference2;
    reg [26:0] far_aligned2;
    reg [23:0] larger2;

    reg [25:0] close_abs3;
    reg close_negative3;
    reg [27:0] far_sum3;
    wire [27:0] far_operand2 = !meta[2][8] ? {1'b1,~far_aligned2} : {1'b0,far_aligned2};
    wire [12:0] reduced3;
    wire [4:0] leading3;
    compressBy2 #(26) compress_close(close_abs3,reduced3);
    countLeadingZeros #(13,5) count_close(reduced3,leading3);

    reg [25:0] close_abs4;
    reg close_negative4;
    reg [4:0] norm_distance4;
    reg [26:0] far_out4;
    wire [26:0] close_out4 = {close_abs4,1'b0} << norm_distance4;
    wire cancellation4 = !(|close_out4[26:25]);
    wire zero4 = meta[4][9] || (!meta[4][10] && meta[4][3] && cancellation4);
    wire special_case4 = meta[4][10] || meta[4][9];
    // RNE exact cancellation and unlike-signed zero sums have positive sign.
    wire sign4 = (meta[4][8] && meta[4][7]) || (meta[4][5] && meta[4][7]) ||
                 (meta[4][4] && meta[4][6]) ||
                 (!special_case4 && meta[4][3] && !cancellation4 && (meta[4][7] ^ close_negative4)) ||
                 (!special_case4 && !meta[4][3] && meta[4][2]);
    integer i;
    always @(posedge clk) if (enable) begin
        meta[1] <= {snan_a || snan_b || (inf_a && inf_b && !eq_sign),
                    nan_a || nan_b,inf_a || inf_b,zero_a && zero_b,eq_sign,
                    sign_a,sign_b,inf_a,inf_b,close_path,
                    difference[9] ? sign_b : sign_a,difference[9],difference[0]};
        base_exp[1] <= close_path || difference[9] ? exp_b : exp_a;
        distance1 <= max_align ? 5'd31 : magnitude[4:0];
        sig_a1 <= sig_a[23:0]; sig_b1 <= sig_b[23:0];
        for(i=2;i<=4;i=i+1) begin meta[i] <= meta[i-1]; base_exp[i] <= base_exp[i-1]; end
        close_difference2 <= close_aligned1 - {2'b0,sig_b1,1'b0};
        far_aligned2 <= {main_aligned1[28:3],(|main_aligned1[2:0]) || (|(reduced1 & mask1))};
        larger2 <= larger1;
        close_abs3 <= close_difference2[26] ? -close_difference2[25:0] : close_difference2[25:0];
        close_negative3 <= close_difference2[26];
        far_sum3 <= {1'b0,larger2,3'b0} + far_operand2 + {27'b0,!meta[2][8]};
        close_abs4 <= close_abs3; close_negative4 <= close_negative3;
        norm_distance4 <= {leading3[3:0],1'b0};
        far_out4 <= !meta[3][8] ? far_sum3[26:0] :
                    {far_sum3[27:2],far_sum3[1] | far_sum3[0]};
        special <= {meta[4][12:10],zero4,sign4};
        exponent <= base_exp[4] - (meta[4][3] ? {5'b0,norm_distance4} : {9'b0,!meta[4][8]});
        significand <= meta[4][3] ? close_out4 : far_out4;
    end
    wire unused_leading_bit = leading3[4];
endmodule
`default_nettype wire
