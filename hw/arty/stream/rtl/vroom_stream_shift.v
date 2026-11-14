`default_nettype none
// One synchronous shift enable and one asynchronous tap; maps to SRL16E per bit.
// No payload reset. The owner's valid/credit state protects uninitialized slots.
module vroom_stream_shift #(parameter integer WIDTH=32)(
    input wire clk,enable,
    input wire [WIDTH-1:0] data,
    input wire [3:0] address,
    output wire [WIDTH-1:0] value
);
    genvar bit_number;
    generate for(bit_number=0;bit_number<WIDTH;bit_number=bit_number+1)begin: bits
        reg [15:0] shift;
        always @(posedge clk)if(enable)shift<={shift[14:0],data[bit_number]};
        assign value[bit_number]=shift[address];
    end endgenerate
endmodule
`default_nettype wire
