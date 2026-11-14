`timescale 1ns/1ps
`default_nettype none
module tb_rate_monitor;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1,start=0,inner_done=0,outer_done=0;
    reg [31:0] inner_done_job=0,outer_done_job=0;
    wire running,inner_release,outer_release,inner_pending,outer_pending,fault,fault_actor;
    wire [31:0] inner_job,outer_job,fault_job;
    wire [63:0] elapsed_cycles,fault_cycle;
    wire [1:0] fault_code;
    integer i,checks=0,inner_count=0,outer_count=0;
    reg [4095:0] wave_path;
    vroom_rate_monitor dut (
        .clk(clk),.reset(reset),.start(start),.inner_done(inner_done),.inner_done_job(inner_done_job),
        .outer_done(outer_done),.outer_done_job(outer_done_job),.running(running),
        .inner_release(inner_release),.outer_release(outer_release),.inner_job(inner_job),.outer_job(outer_job),
        .inner_pending(inner_pending),.outer_pending(outer_pending),.elapsed_cycles(elapsed_cycles),
        .fault(fault),.fault_actor(fault_actor),.fault_code(fault_code),.fault_job(fault_job),.fault_cycle(fault_cycle)
    );
    task check;
        input condition;
        input [1023:0] message;
        begin checks=checks+1; if(condition !== 1'b1) $fatal(1,"%0s",message); end
    endtask
    task step;
        begin @(posedge clk); #1; end
    endtask
    task restart;
        begin
            @(negedge clk); reset=1; start=0; inner_done=0; outer_done=0; step;
            @(negedge clk); reset=0; start=1; step;
            check(inner_release && outer_release && inner_job==0 && outer_job==0,"initial releases at cycle zero");
            @(negedge clk); start=0;
        end
    endtask
    initial begin
        if(!$value$plusargs("vcd=%s",wave_path)) wave_path="rates.vcd";
        $dumpfile(wave_path); $dumpvars(0,tb_rate_monitor);
        restart;
        $display("EVENT 0,1,1");
        // Finish every job exactly at its deadline, including simultaneous boundaries.
        for(i=1;i<=200000;i=i+1) begin
            inner_done=(i%10000==0); outer_done=(i%100000==0);
            inner_done_job=inner_job; outer_done_job=outer_job; step;
            check(!fault && running,"exact-deadline completion accepted");
            check(inner_release==(i%10000==0),"10 kHz release spacing");
            check(outer_release==(i%100000==0),"1 kHz release spacing");
            if(inner_release) inner_count=inner_count+1;
            if(outer_release) outer_count=outer_count+1;
            if(inner_release || outer_release) $display("EVENT %0d,%0d,%0d",elapsed_cycles,inner_release,outer_release);
            @(negedge clk);
        end
        check(inner_count==20 && outer_count==2,"20:2 release counts over 2 ms after initial release");
        // An incomplete inner job fails on cycle 10,000, without a replacement release.
        restart; inner_done=0; outer_done=0; repeat(10000) step;
        check(fault && !fault_actor && fault_code==1 && fault_job==0 && fault_cycle==10000,"inner deadline fault evidence");
        check(!inner_release && !outer_release && !running,"overrun stops releases");
        @(negedge clk); start=1; inner_done=1; inner_done_job=0; repeat(2) step;
        check(fault && fault_cycle==10000 && !running,"late completion cannot hide a failure");
        // Inner jobs continue normally while an unfinished outer job reaches 1 ms.
        restart;
        for(i=1;i<=100000;i=i+1) begin
            inner_done=(i%10000==0); inner_done_job=inner_job; outer_done=0; step; @(negedge clk);
        end
        check(fault && fault_actor && fault_code==1 && fault_cycle==100000,"outer deadline fault evidence");
        restart; inner_done=1; inner_done_job=99; step;
        check(fault && !fault_actor && fault_code==2,"wrong completion generation faults");
        restart; inner_done=1; inner_done_job=0; step;
        @(negedge clk); inner_done=0; step;
        @(negedge clk); inner_done=1; step;
        check(fault && fault_code==2,"duplicate completion faults");
        $display("RESULT {\"test\":\"rate_monitor\",\"status\":\"passed\",\"assertions\":%0d,\"core_hz\":100000000,\"inner_period_cycles\":10000,\"outer_period_cycles\":100000,\"observed_inner_releases\":20,\"observed_outer_releases\":2}",checks);
        $finish;
    end
    initial begin #4000000; $fatal(1,"watchdog"); end
endmodule
`default_nettype wire
