`default_nettype none
// 8N1, exact integer divider. Board default: 100 MHz / 100 = 1 Mbaud.
module vroom_board_uart #(parameter integer CLKS_PER_BIT=100) (
    input wire clk, reset, rx,
    output wire tx,
    output reg rx_valid, rx_error,
    output reg [7:0] rx_data,
    input wire tx_valid,
    output wire tx_ready,
    input wire [7:0] tx_data
);
    localparam integer CW=$clog2(CLKS_PER_BIT+1);
    localparam [1:0] IDLE=0, START=1, DATA=2, STOP=3;
    (* ASYNC_REG="TRUE" *) reg rx_meta, rx_sync;
    reg [1:0] state;
    reg [CW-1:0] count;
    reg [2:0] index;
    reg [9:0] frame;
    reg [3:0] left;
    reg [CW-1:0] tx_count;
    assign tx=frame[0];
    assign tx_ready=!reset && left==0;
    always @(posedge clk) begin
        rx_meta<=rx; rx_sync<=rx_meta;
        rx_valid<=0; rx_error<=0;
        case(state)
            IDLE: if(!rx_sync) begin state<=START;count<=CW'(CLKS_PER_BIT/2-1);end
            START: if(count!=0)count<=count-1'b1;
                else if(!rx_sync)begin state<=DATA;count<=CW'(CLKS_PER_BIT-1);index<=0;end
                else state<=IDLE;
            DATA: if(count!=0)count<=count-1'b1;
                else begin
                    rx_data[index]<=rx_sync;count<=CW'(CLKS_PER_BIT-1);
                    if(index==7)state<=STOP;else index<=index+1'b1;
                end
            STOP: if(count!=0)count<=count-1'b1;
                else begin rx_valid<=rx_sync;rx_error<=!rx_sync;state<=IDLE;end
            default: state<=IDLE;
        endcase
        if(reset)begin state<=IDLE;count<=0;index<=0;rx_meta<=1;rx_sync<=1;rx_data<=0;rx_valid<=0;rx_error<=0;end
    end
    always @(posedge clk) begin
        if(left!=0) begin
            if(tx_count==0)begin frame<={1'b1,frame[9:1]};left<=left-1'b1;tx_count<=CW'(CLKS_PER_BIT-1);end
            else tx_count<=tx_count-1'b1;
        end else if(tx_valid)begin frame<={1'b1,tx_data,1'b0};left<=10;tx_count<=CW'(CLKS_PER_BIT-1);end
        if(reset)begin frame<=10'h3ff;left<=0;tx_count<=0;end
    end
endmodule
`default_nettype wire
