`default_nettype none
// Existing FP32 inference stream on the actual Arty A7-35T. Only clock and
// USB-UART pins are connected. The host supplies every parameter per job.
module vroom_board_top(input wire clk, input wire uart_rx, output wire uart_tx);
    localparam integer LANES=12;
    reg [7:0] startup=0;
    always @(posedge clk)if(!(&startup))startup<=startup+1'b1;
    wire reset=!(&startup);
    wire rx_valid,rx_error,tx_valid,tx_ready;
    wire [7:0] rx_data,tx_data;
    vroom_board_uart uart(.*,.rx(uart_rx),.tx(uart_tx));
    wire job_valid,job_actor,job_ready,input_valid,input_last,input_ready;
    wire [31:0] job_id,input_data;
    wire bias_valid,bias_last,bias_ready,weight_valid,weight_last,weight_ready;
    wire [LANES*32-1:0] bias_data,weight_data;
    wire [31:0] bias_job,weight_job;
    wire [1:0] bias_layer,weight_layer;
    wire [5:0] bias_group,weight_group;
    wire [7:0] weight_column;
    wire output_valid,output_ready,output_actor;
    wire [63:0] output_data;
    wire [31:0] output_job;
    wire [4:0] arithmetic_flags;
    wire [3:0] fault_code,debug_state;
    wire [1:0] layer;
    vroom_board_bridge #(.LANES(LANES)) bridge(.*);
    vroom_actor_stream #(.LANES(LANES)) actor(.*);
    wire unused_debug=(|debug_state) || (|layer);
endmodule
`default_nettype wire
