`default_nettype none
module vroom_fp32_add (
    input wire clk, reset, enable, valid,
    input wire [15:0] tag,
    input wire [31:0] a, b,
    output wire response_valid,
    output wire [15:0] response_tag,
    output wire [31:0] result,
    output wire [4:0] flags
);
    vroom_fp32_pipe #(.MULTIPLY(0)) pipe(
        .clk(clk),.reset(reset),.enable(enable),.valid(valid),.tag(tag),.a(a),.b(b),
        .response_valid(response_valid),.response_tag(response_tag),.result(result),.flags(flags));
endmodule
`default_nettype wire
