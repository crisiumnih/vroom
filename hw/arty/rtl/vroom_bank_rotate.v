`default_nettype none
// Combinational rotation of sixteen bank words. Lane i reads bank (i+offset)
// modulo 16; REVERSE routes lane (bank-offset) back to its physical bank.
// Four shared 2:1 mux stages avoid sixteen independent 16:1 lane selectors.
// No arithmetic on payload bits, no state, and no added protocol latency.
module vroom_bank_rotate #(
    parameter integer WIDTH = 32,
    parameter integer REVERSE = 0,
    parameter integer WORDS = 16
) (
    input wire [16*WIDTH-1:0] data_in,
    input wire [3:0] offset,
    output wire [WORDS*WIDTH-1:0] data_out
);
    wire [16*WIDTH-1:0] stage [0:4];
    assign stage[0] = data_in;
    assign data_out = stage[4][WORDS*WIDTH-1:0];
    genvar step, bank;
    generate for (step=0; step<4; step=step+1) begin : stages
        for (bank=0; bank<16; bank=bank+1) begin : banks
            localparam integer SOURCE = (REVERSE != 0) ? (bank+16-(1<<step))%16 : (bank+(1<<step))%16;
            assign stage[step+1][bank*WIDTH +: WIDTH] = offset[step] ?
                stage[step][SOURCE*WIDTH +: WIDTH] : stage[step][bank*WIDTH +: WIDTH];
        end
    end endgenerate
endmodule
`default_nettype wire
