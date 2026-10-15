`timescale 1ns/1ps
`default_nettype none
// Test-only in-order SIMD FP32 model. C VPI computes two correctly separated
// native binary32 operations; latency and input stalls are simulated parameters.
module fp32_mac_model #(
    parameter integer LANES=12,
    parameter integer LATENCY=19
) (
    input wire clk, reset,
    input wire force_stall,
    input wire corrupt_tag,
    input wire valid,
    output wire ready,
    input wire [31:0] tag,
    input wire [LANES-1:0] mask,
    input wire [31:0] activation,
    input wire [LANES*32-1:0] weight, accumulator,
    output wire response_valid,
    input wire response_ready,
    output wire [31:0] response_tag,
    output wire [LANES*32-1:0] response_data
);
    reg [31:0] tags[0:63], eligible[0:63];
    reg [LANES*32-1:0] data[0:63];
    reg [5:0] wr=0, rd=0;
    reg [6:0] count=0;
    reg [31:0] cycle=0, rng=1;
    integer seed, lane;
    initial if (!$value$plusargs("seed=%d",seed)) seed=2805;
    assign ready=!reset && !force_stall && count<64 && rng[1:0]!=0;
    assign response_valid=!reset && count!=0 && cycle>=eligible[rd];
    assign response_tag=tags[rd] ^ (corrupt_tag ? 32'h80000000 : 32'b0);
    assign response_data=data[rd];
    always @(posedge clk) begin
        if (reset) begin wr<=0; rd<=0; count<=0; cycle<=0; rng<=seed; end
        else begin
            cycle<=cycle+1;
            rng<={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            if (valid && ready) begin
                tags[wr]<=tag;
                eligible[wr]<=cycle+LATENCY;
                for(lane=0;lane<LANES;lane=lane+1)
                    data[wr][lane*32 +: 32]<=mask[lane] ?
                        $vroom_fp32_mac(weight[lane*32 +: 32],activation,accumulator[lane*32 +: 32]) : 0;
                wr<=wr+1'b1;
            end
            if (response_valid && response_ready) rd<=rd+1'b1;
            case ({valid && ready,response_valid && response_ready})
                2'b10: count<=count+1'b1;
                2'b01: count<=count-1'b1;
                default: ;
            endcase
        end
    end
endmodule
`default_nettype wire
