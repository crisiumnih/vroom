`default_nettype none
// Binary32, round to nearest/even, gradual underflow, tininess after rounding.
// Separate multiply and add rounding; no fused MAC. Exception flags
// follow HardFloat: invalid, divide-by-zero, overflow, underflow, inexact.
// Transfer latency: multiply 6, add 9 enabled clock edges; II=1. Global enable
// freezes every data/tag/valid stage. Reset discards all in-flight operations.
module vroom_fp32_pipe #(parameter integer MULTIPLY = 0) (
    input wire clk, reset, enable, valid,
    input wire [15:0] tag,
    input wire [31:0] a, b,
    output wire response_valid,
    output reg [15:0] response_tag,
    output reg [31:0] result,
    output reg [4:0] flags
);
    localparam integer RAW_STAGES = MULTIPLY != 0 ? 2 : 5;
    localparam integer LATENCY = RAW_STAGES + 4;
    localparam integer EXACT_SUBNORMALS = MULTIPLY != 0 ? 0 : 1;
    wire [32:0] rec_a, rec_b, rounded;
    reg [32:0] a_q, b_q, rounded_q;
    wire [4:0] special;
    wire signed [9:0] exponent;
    wire [26:0] significand;
    wire [4:0] rounded_flags;
    reg [4:0] flags_q;
    wire [31:0] ieee_result;
    reg [LATENCY-1:0] valid_q;
    reg [15:0] tags [0:LATENCY-2];
    integer i;
    vroom_fp32_recode recode_a(a,rec_a);
    vroom_fp32_recode recode_b(b,rec_b);
    generate if (MULTIPLY != 0) begin : multiplier
        vroom_fp32_mul_raw_pipe arithmetic(clk,enable,a_q,b_q,special,exponent,significand);
    end else begin : adder
        vroom_fp32_add_raw_pipe arithmetic(clk,enable,a_q,b_q,special,exponent,significand);
    end endgenerate
    vroom_fp32_round_pipe #(EXACT_SUBNORMALS) round_result(
        clk,enable,special,exponent,significand,rounded,rounded_flags);
    recFNToFN #(8,24) decode_result(rounded_q,ieee_result);
    assign response_valid = valid_q[LATENCY-1] && !reset;
    always @(posedge clk) begin
        if (reset) valid_q <= 0;
        else if (enable) valid_q <= {valid_q[LATENCY-2:0],valid};
        if (enable) begin
            a_q <= rec_a; b_q <= rec_b;
            tags[0] <= tag;
            for (i=1;i<LATENCY-1;i=i+1) tags[i] <= tags[i-1];
            rounded_q <= rounded; flags_q <= rounded_flags;
            result <= ieee_result; flags <= flags_q; response_tag <= tags[LATENCY-2];
        end
    end
endmodule
`default_nettype wire
