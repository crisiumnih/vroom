`timescale 1ns/1ps
`default_nettype none
module tb_tanh;
    parameter integer COUNT=1;
    reg clk=0; always #5 clk=~clk;
    reg reset=1,valid=0,response_ready=0;
    reg [31:0] a=0;
    reg [15:0] tag=0;
    wire ready,response_valid,invalid;
    wire [31:0] result;
    wire [15:0] response_tag;
    reg [119:0] vectors[0:COUNT-1];
    integer cycles=0,checks=0,i,j,trace,accepted,latency,delay,seed;
    integer min_latency=9999,max_latency=0,stall_cycles=0;
    integer pair_sent,pair_returned,pair_start,pair_cycles;
    reg sampled_ready,sampled_response;
    reg [31:0] rng;
    reg [48:0] held;
    vroom_fp32_tanh dut(clk,reset,valid,ready,tag,a,response_valid,response_ready,response_tag,result,invalid);
    always @(posedge clk) cycles=cycles+1;
    task check;
        input condition;
        input [767:0] message;
        begin checks=checks+1;if(condition!==1'b1)$fatal(1,"%0s vector=%0d cycle=%0d",message,i,cycles);end
    endtask
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=2805;rng=seed;
        $readmemh("vectors.hex",vectors);
        trace=$fopen("responses.csv","w");
        $fdisplay(trace,"index,input,result,golden,invalid,accepted_edge,first_response_edge,latency,held_cycles");
        $dumpfile("tanh.vcd");$dumpvars(0,tb_tanh.dut);
        repeat(2)@(negedge clk);reset=0;
        for(i=0;i<COUNT;i=i+1)begin
            @(negedge clk);check(ready,"drained unit ready");
            a=vectors[i][119:88];tag=i;valid=1;response_ready=0;
            @(posedge clk);#1;accepted=cycles;
            @(negedge clk);valid=0;
            // Present unrelated live inputs while the request is executing;
            // the accepted payload must be captured internally.
            a=~vectors[i][119:88];tag=~i;
            while(!response_valid)begin
                check(!ready,"single request owns iterative datapath");
                @(negedge clk);
            end
            @(posedge clk);#1;latency=cycles-accepted;
            if(result!==vectors[i][87:56])$display("MISMATCH input=%08x got=%08x model=%08x golden=%08x",vectors[i][119:88],result,vectors[i][87:56],vectors[i][55:24]);
            check(result===vectors[i][87:56],"exact integer-model match");
            check({7'b0,invalid}===vectors[i][23:16],"invalid flag");
            check(response_tag==i%65536,"tag captured with request");
            check(latency==vectors[i][15:0],"first-valid latency matches bounded cycle model");
            check(latency<=199,"worst-case latency bound");
            if(latency<min_latency)min_latency=latency;
            if(latency>max_latency)max_latency=latency;
            rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);delay=rng[2:0];
            held={result,response_tag,invalid};
            for(j=0;j<delay;j=j+1)begin
                @(posedge clk);#1;
                check(response_valid && !ready && {result,response_tag,invalid}===held,"held output stable under backpressure");
                stall_cycles=stall_cycles+1;
            end
            $fdisplay(trace,"%0d,%08x,%08x,%08x,%0d,%0d,%0d,%0d,%0d",i,vectors[i][119:88],result,vectors[i][55:24],invalid,accepted,accepted+latency,latency,delay);
            @(negedge clk);response_ready=1;
            @(posedge clk);#1;check(!response_valid && ready,"one response consumed exactly once");
            if(i==31)$dumpoff;
        end
        // Two worst-latency outputs on the shared unit, continuously ready
        // consumer. Request two stays asserted until the unit accepts it.
        @(negedge clk);valid=1;response_ready=1;a=32'h411fffff;tag=16'h1234;
        pair_sent=0;pair_returned=0;pair_start=0;pair_cycles=0;
        while(pair_returned<2)begin
            @(posedge clk);sampled_ready=ready;sampled_response=response_valid;#1;
            if(sampled_ready && valid)begin
                if(pair_sent==0)pair_start=cycles;
                pair_sent=pair_sent+1;
                if(pair_sent==2)valid=0;
            end
            if(sampled_response)begin
                check(result==32'h3f800000 && response_tag==16'h1234,"worst-case pair output");
                pair_returned=pair_returned+1;
                if(pair_returned==2)pair_cycles=cycles-pair_start;
            end
        end
        check(pair_sent==2 && pair_cycles==399,"two worst-case outputs complete in 399 clocks");
        // Explicit cancellation tests; no failed calculation is restarted.
        @(negedge clk);response_ready=0;valid=1;a=32'h3f800000;tag=16'hbeef;
        @(negedge clk);valid=0;
        repeat(20)@(negedge clk);check(!ready && !response_valid,"cancel active CORDIC");reset=1;
        @(negedge clk);reset=0;
        repeat(205)begin @(posedge clk);#1;check(!response_valid && ready,"no stale result after reset");end
        @(negedge clk);valid=1;a=0;
        @(negedge clk);valid=0;check(response_valid,"hold fast-path output");reset=1;
        @(negedge clk);reset=0;
        @(posedge clk);#1;check(!response_valid && ready,"reset discards blocked response");
        $display("RESULT {\"status\":\"passed\",\"vectors\":%0d,\"assertions\":%0d,\"minimum_latency\":%0d,\"maximum_latency\":%0d,\"response_stall_cycles\":%0d,\"worst_case_pair_cycles\":%0d,\"reset_cancellations\":2}",COUNT,checks,min_latency,max_latency,stall_cycles,pair_cycles);
        $fclose(trace);$finish;
    end
    initial begin #2000000000;$fatal(1,"watchdog");end
endmodule
`default_nettype wire
