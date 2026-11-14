`timescale 1ns/1ps
`default_nettype none
module tb_uart;
    localparam integer CPB=100;
    reg clk=0;always #5 clk=~clk;
    reg reset=1,rx=1,tx_valid=0;
    reg [7:0] tx_data=0;
    wire tx,rx_valid,rx_error,tx_ready;
    wire [7:0] rx_data;
    integer expected=0,received=0,i;
    vroom_board_uart #(.CLKS_PER_BIT(CPB)) dut(.*);
    always @(posedge clk)if(!reset)begin
        if(rx_error)$fatal(1,"unexpected framing error");
        if(rx_valid)begin
            if(rx_data!==8'(expected))$fatal(1,"RX expected %02x got %02x",expected,rx_data);
            received=received+1;
        end
    end
    task cycles(input integer count);repeat(count)@(negedge clk);endtask
    task send_rx(input reg [7:0] value);
        integer bit_index;
        begin
            @(negedge clk);rx=0;cycles(CPB);
            for(bit_index=0;bit_index<8;bit_index=bit_index+1)begin rx=value[bit_index];cycles(CPB);end
            rx=1;cycles(CPB);
        end
    endtask
    task send_tx(input reg [7:0] value);
        begin
            @(negedge clk);tx_data=value;tx_valid=1;
            @(posedge clk);while(!tx_ready)@(posedge clk);
            @(negedge clk);tx_valid=0;
        end
    endtask
    task read_tx(input reg [7:0] value);
        integer bit_index;reg [7:0] result;
        begin
            @(negedge tx);cycles(CPB/2);
            if(tx!==0)$fatal(1,"TX start");
            for(bit_index=0;bit_index<8;bit_index=bit_index+1)begin cycles(CPB);result[bit_index]=tx;end
            cycles(CPB);
            if(tx!==1 || result!==value)$fatal(1,"TX expected %02x got %02x",value,result);
            cycles(CPB+2);
        end
    endtask
    initial begin
        cycles(3);reset=0;
        for(i=0;i<256;i=i+1)begin
            expected=i;
            fork send_rx(8'(i));send_tx(8'(i^165));read_tx(8'(i^165));join
        end
        if(received!=256)$fatal(1,"RX count");
        $display("VROOM_UART_OK rx_bytes=256 tx_bytes=256 baud=1000000");$finish;
    end
    initial begin #20000000;$fatal(1,"UART watchdog");end
endmodule
`default_nettype wire
