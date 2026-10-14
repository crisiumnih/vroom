
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

// Vroom derivative of roundAnyRawFNToRecFN, restricted to binary32 RNE,
// tininess-after and canonical NaN. One internal enabled register after the
// rounded significand separates its carry chain from exponent adjustment.
// The caller registers the combinational recoded output as the second stage.
`default_nettype none
module vroom_fp32_round_pipe #(parameter integer EXACT_SUBNORMALS=0) (
    input wire clk, enable,
    input wire [4:0] special,
    input wire signed [9:0] exponent,
    input wire [26:0] significand,
    output wire [32:0] rounded,
    output wire [4:0] flags
);
    wire shift = significand[26];
    wire [24:0] mask_main;
    vroom_fp32_round_mask make_mask(exponent[8:0],mask_main);
    wire [26:0] mask = EXACT_SUBNORMALS != 0 ? {24'b0,shift,2'b11} :
                      {mask_main | {24'b0,shift},2'b11};
    wire [26:0] shifted_mask = {1'b0,mask[26:1]};
    wire [26:0] position_mask = ~shifted_mask & mask;
    wire position = |(significand & position_mask);
    wire extra = |(significand & shifted_mask);
    wire [26:0] increment_sum = ((significand | mask) >> 2) + 27'd1;
    wire [26:0] rounded_wide = position ? increment_sum & ~(extra ? 27'b0 : shifted_mask) :
                              (significand & ~mask) >> 2;
    reg [25:0] rounded_sig_q;
    reg signed [9:0] exponent_q;
    reg [4:0] special_q;
    reg shift_q,position_q,any_round_q,unbounded_increment_q;
    reg [4:2] mask_q;
    always @(posedge clk) if(enable) begin
        rounded_sig_q <= rounded_wide[25:0]; exponent_q <= exponent; special_q <= special;
        shift_q <= shift; position_q <= position; any_round_q <= position || extra;
        unbounded_increment_q <= shift ? significand[2] : significand[1];
        mask_q <= mask[4:2];
    end
    wire signed [10:0] adjusted_exp = {exponent_q[9],exponent_q} + {9'b0,rounded_sig_q[25:24]};
    wire [22:0] fraction = shift_q ? rounded_sig_q[23:1] : rounded_sig_q[22:0];
    wire common_overflow = (adjusted_exp >>> 7) >= 3;
    wire total_underflow = EXACT_SUBNORMALS != 0 ? 1'b0 : adjusted_exp < 107;
    wire carry = shift_q ? rounded_sig_q[25] : rounded_sig_q[24];
    wire common_underflow = EXACT_SUBNORMALS != 0 ? 1'b0 : total_underflow ||
        (any_round_q && (exponent_q >>> 8 <= 0) && (shift_q ? mask_q[3] : mask_q[2]) &&
         !(!(shift_q ? mask_q[4] : mask_q[3]) && carry && position_q && unbounded_increment_q));
    wire nan_out = special_q[4] || special_q[3];
    wire common_case = !nan_out && !special_q[2] && !special_q[1];
    wire overflow = common_case && common_overflow;
    wire inf_out = special_q[2] || overflow;
    wire [8:0] exp_out = (adjusted_exp[8:0] &
                          ~(special_q[1] || total_underflow ? 9'h1c0 : 9'b0) &
                          ~(inf_out ? 9'h040 : 9'b0)) |
                          (inf_out ? 9'h180 : 9'b0) | (nan_out ? 9'h1c0 : 9'b0);
    wire [22:0] fract_out = nan_out ? 23'h400000 :
                           special_q[1] || total_underflow ? 23'b0 : fraction;
    assign rounded = {nan_out ? 1'b0 : special_q[0],exp_out,fract_out};
    assign flags = {special_q[4],1'b0,overflow,common_case && common_underflow,
                    overflow || (common_case && (total_underflow || any_round_q))};
    wire unused_rounded_top = rounded_wide[26] | (|adjusted_exp[10:9]);
endmodule
`default_nettype wire
