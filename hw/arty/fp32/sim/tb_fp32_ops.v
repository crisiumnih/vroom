`timescale 1ns/1ps
`default_nettype none
module tb_fp32_ops;
    parameter integer COUNT=1, ADD_LATENCY=9, MUL_LATENCY=6;
    reg clk=0; always #5 clk=~clk;
    reg reset=1, enable=1, valid=0;
    reg [15:0] tag=0;
    reg [31:0] a=0,b=0;
    wire av,mv;
    wire [15:0] at,mt;
    wire [31:0] ar,mr;
    wire [4:0] af,mf;
    reg [143:0] vectors [0:COUNT-1];
    integer accepted_edge [0:COUNT-1];
    integer edges=0,sent=0,adds=0,muls=0,checks=0,stalls=0;
    integer last_add=-2, last_mul=-2, add_run=0,mul_run=0,max_add_run=0,max_mul_run=0;
    integer trace, seed, i;
    reg checking=1;
    reg [31:0] rng;
    reg [107:0] held;
    wire [107:0] outputs={av,at,ar,af,mv,mt,mr,mf};
    vroom_fp32_add adder(clk,reset,enable,valid,tag,a,b,av,at,ar,af);
    vroom_fp32_mul multiplier(clk,reset,enable,valid,tag,a,b,mv,mt,mr,mf);
    task check;
        input condition;
        input [767:0] message;
        begin checks=checks+1; if(condition!==1'b1) $fatal(1,"%0s sent=%0d adds=%0d muls=%0d",message,sent,adds,muls); end
    endtask
    always @(posedge clk) begin
        if(!reset && checking) begin
            if(enable) begin
                edges=edges+1;
                if(valid) begin accepted_edge[sent]=edges; sent=sent+1; end
                if(av) begin
                    check(adds<sent && at==adds%65536,"add ordering and tag");
                    if(ar!==vectors[adds][79:48] || {3'b0,af}!==vectors[adds][47:40])
                        $display("ADD %0d a=%08x b=%08x got=%08x/%02x expected=%08x/%02x",adds,vectors[adds][143:112],vectors[adds][111:80],ar,af,vectors[adds][79:48],vectors[adds][47:40]);
                    check(ar===vectors[adds][79:48] && {3'b0,af}===vectors[adds][47:40],"integer-reference add value and flags");
                    check(edges-accepted_edge[adds]==ADD_LATENCY,"add enabled-edge latency");
                    add_run=(edges==last_add+1)?add_run+1:1;
                    if(add_run>max_add_run) max_add_run=add_run;
                    last_add=edges;
                    $fdisplay(trace,"%0d,add,%0d,%08x,%02x",edges,adds,ar,af); adds=adds+1;
                end
                if(mv) begin
                    check(muls<sent && mt==muls%65536,"multiply ordering and tag");
                    if(mr!==vectors[muls][39:8] || {3'b0,mf}!==vectors[muls][7:0])
                        $display("MUL %0d a=%08x b=%08x got=%08x/%02x expected=%08x/%02x",muls,vectors[muls][143:112],vectors[muls][111:80],mr,mf,vectors[muls][39:8],vectors[muls][7:0]);
                    check(mr===vectors[muls][39:8] && {3'b0,mf}===vectors[muls][7:0],"integer-reference multiply value and flags");
                    check(edges-accepted_edge[muls]==MUL_LATENCY,"multiply enabled-edge latency");
                    mul_run=(edges==last_mul+1)?mul_run+1:1;
                    if(mul_run>max_mul_run) max_mul_run=mul_run;
                    last_mul=edges;
                    $fdisplay(trace,"%0d,multiply,%0d,%08x,%02x",edges,muls,mr,mf); muls=muls+1;
                end
            end else begin
                held=outputs; stalls=stalls+1;
                #1; check(outputs===held,"global enable holds all payload and valid bits");
            end
        end
    end
    initial begin
        if(!$value$plusargs("seed=%d",seed)) seed=2805;
        rng=seed;
        $readmemh("operators.hex",vectors);
        trace=$fopen("operator-events.csv","w");
        $fdisplay(trace,"enabled_edge,operation,index,result,flags");
        $dumpfile("operators.vcd"); $dumpvars(1,tb_fp32_ops);
        repeat(2) @(negedge clk); reset=0;
        for(i=0;i<COUNT;i=i+1) begin
            @(negedge clk);
            rng=rng^(rng<<13); rng=rng^(rng>>17); rng=rng^(rng<<5);
            if(i>1024 && rng[2:0]==0) begin
                enable=0; valid=1; repeat(2) @(negedge clk);
            end
            enable=1; valid=1; tag=i; a=vectors[i][143:112]; b=vectors[i][111:80];
            if(i==1024) $dumpoff; // All transfers remain in the full CSV trace.
        end
        @(negedge clk); valid=0;
        while(adds<COUNT || muls<COUNT) @(negedge clk);
        check(sent==COUNT && max_add_run>=512 && max_mul_run>=512,"II=1 sustained independent operations");
        check(stalls>0,"clock-enable stall coverage");
        // Deliberately cancel in-flight requests; this is a named reset test,
        // not silent recovery from a numerical failure.
        checking=0; valid=1; a=32'h3f800000;b=32'h40000000;tag=16'hffff;
        repeat(2) @(negedge clk); reset=1;
        @(posedge clk); #1; check(!av && !mv,"reset invalidates responses");
        @(negedge clk); valid=0;reset=0;
        repeat(12) begin @(posedge clk); #1; check(!av && !mv,"no stale response after reset"); end
        $display("RESULT {\"status\":\"passed\",\"pairs\":%0d,\"assertions\":%0d,\"add_latency\":%0d,\"mul_latency\":%0d,\"max_add_run\":%0d,\"max_mul_run\":%0d,\"stall_cycles\":%0d,\"reset_cancelled_requests\":2}",COUNT,checks,ADD_LATENCY,MUL_LATENCY,max_add_run,max_mul_run,stalls);
        $fclose(trace); $finish;
    end
    initial begin #20000000; $fatal(1,"operator watchdog"); end
endmodule
`default_nettype wire
