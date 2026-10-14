`timescale 1ns/1ps
`default_nettype none
module tb_pinned_cache;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1, boot_valid=0, seal=0;
    reg [31:0] boot_data=0;
    wire boot_ready, sealed, fault;
    wire [15:0] loaded_words;
    reg [10:0] read_a=0, read_b=0;
    reg [131:0] address_a=0, address_b=0;
    wire [10:0] valid_a, valid_b;
    wire [351:0] data_a, data_b;
    integer i,b,checks=0;
    reg [4095:0] wave_path;
    vroom_pinned_cache dut (
        .clk(clk), .reset(reset), .boot_valid(boot_valid), .boot_ready(boot_ready), .boot_data(boot_data),
        .seal(seal), .sealed(sealed), .fault(fault), .loaded_words(loaded_words),
        .read_a(read_a), .read_b(read_b), .address_a(address_a), .address_b(address_b),
        .valid_a(valid_a), .valid_b(valid_b), .data_a(data_a), .data_b(data_b)
    );
    function [31:0] pattern;
        input integer index;
        begin pattern=32'h80000000 ^ (index*32'h9e3779b9); end
    endfunction
    task check;
        input condition;
        input [1023:0] message;
        begin checks=checks+1; if (condition !== 1'b1) $fatal(1,"%0s",message); end
    endtask
    task step;
        begin @(posedge clk); #1; end
    endtask
    initial begin
        if (!$value$plusargs("vcd=%s",wave_path)) wave_path="cache.vcd";
        $dumpfile(wave_path); $dumpvars(0,tb_pinned_cache);
        step; @(negedge clk); reset=0; read_a='h7ff; read_b='h7ff;
        step; check(valid_a==0 && valid_b==0,"unsealed cache cannot be read");
        for(i=0;i<45056;i=i+1) begin
            @(negedge clk); boot_valid=1; boot_data=pattern(i);
            #1; check(boot_ready,"boot stream ready"); step;
        end
        check(loaded_words==45056 && !boot_ready,"complete payload blocks extra writes");
        @(negedge clk); boot_valid=0; seal=1; step; check(sealed && !fault,"seal complete model");
        @(negedge clk); seal=0; boot_valid=1; boot_data=32'hdeadbeef;
        // Exhaustively read both ports with opposing address sequences.
        for(i=0;i<4096;i=i+1) begin
            @(negedge clk);
            for(b=0;b<11;b=b+1) begin address_a[b*12 +: 12]=i; address_b[b*12 +: 12]=4095-i; end
            step; check(valid_a=='h7ff && valid_b=='h7ff && !boot_ready,"two reads per bank while boot locked");
            for(b=0;b<11;b=b+1) begin
                check(data_a[b*32 +: 32]===pattern(i*11+b),"bank A exact data");
                check(data_b[b*32 +: 32]===pattern((4095-i)*11+b),"bank B exact data");
            end
        end
        @(negedge clk); read_a=0; read_b=0; boot_valid=0; step; check(valid_a==0 && valid_b==0,"read enables honored");
        @(negedge clk); reset=1; step;
        @(negedge clk); reset=0; seal=1; step;
        check(fault && !sealed && loaded_words==0,"reset invalidates payload and early seal faults");
        @(negedge clk); seal=0; repeat(3) step; check(fault && !boot_ready,"fault sticky");
        $display("RESULT {\"test\":\"pinned_cache\",\"status\":\"passed\",\"assertions\":%0d,\"loaded_words\":45056,\"verified_reads\":90112}",checks);
        $finish;
    end
    initial begin #1000000; $fatal(1,"watchdog"); end
endmodule
`default_nettype wire
