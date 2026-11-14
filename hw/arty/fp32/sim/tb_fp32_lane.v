`timescale 1ns/1ps
`default_nettype none
module tb_fp32_lane;
    parameter integer GROUPS=1, COUNT=256, FEEDBACK=11;
    reg clk=0; always #5 clk=~clk;
    reg reset=1,init_valid=0,valid=0;
    wire init_ready,ready,response_valid;
    reg [5:0] init_group=0;
    reg [31:0] init_value=0,a=0,b=0;
    reg [15:0] tag=0;
    wire [15:0] response_tag;
    wire [31:0] result;
    wire [4:0] flags;
    reg [119:0] vectors[0:COUNT-1];
    reg [31:0] biases[0:63];
    reg [4095:0] vector_file;
    integer checks=0,sent=0,returned=0,cycles=0,trace,i,last_return=-1,max_gap=0,min_gap=999999;
    integer run_length=0,max_run=0,blocked=0,peak_credits=0,first_issue=0,last_write=0;
    reg checking=1;
    vroom_fp32_lane dut(clk,reset,init_valid,init_ready,init_group,init_value,
                       valid,ready,tag,a,b,response_valid,response_tag,result,flags);
    task check;
        input condition;
        input [767:0] message;
        begin checks=checks+1;if(condition!==1'b1)$fatal(1,"%0s sent=%0d returned=%0d",message,sent,returned);end
    endtask
    always @(posedge clk) begin
        cycles=cycles+1;
        if(!reset && checking) begin
            if(valid && ready) begin
                if(sent==0) first_issue=cycles;
                $fdisplay(trace,"%0d,issue,%0d,%04x,%08x,%02x",cycles,sent,tag,0,0);
                sent=sent+1;
            end
            if(valid && !ready) blocked=blocked+1;
            if(dut.credits>peak_credits) peak_credits=dut.credits;
            check(dut.credits<=16 && dut.queued<=16,"bounded product credits and queue");
            if(response_valid) begin
                check(returned<sent && response_tag===vectors[returned][119:104],"lane result ordering and tag");
                if(result!==vectors[returned][39:8] || {3'b0,flags}!==vectors[returned][7:0])
                    $display("LANE %0d got=%08x/%02x expected=%08x/%02x",returned,result,flags,vectors[returned][39:8],vectors[returned][7:0]);
                check(result===vectors[returned][39:8] && {3'b0,flags}===vectors[returned][7:0],"unfused sequential FP32 accumulation and flags");
                if(last_return>=0) begin
                    if(cycles-last_return>max_gap)max_gap=cycles-last_return;
                    if(cycles-last_return<min_gap)min_gap=cycles-last_return;
                    if(GROUPS==1)check(cycles-last_return==FEEDBACK,"single-row feedback interval");
                end
                run_length=(cycles==last_return+1)?run_length+1:1;
                if(run_length>max_run)max_run=run_length;
                last_return=cycles;last_write=cycles;
                $fdisplay(trace,"%0d,retire,%0d,%04x,%08x,%02x",cycles,returned,response_tag,result,flags);
                returned=returned+1;
            end
        end
    end
    initial begin
        if(!$value$plusargs("vectors=%s",vector_file))$fatal(1,"missing vectors");
        $readmemh(vector_file,vectors);$readmemh("bias.hex",biases);
        trace=$fopen("lane-events.csv","w");$fdisplay(trace,"cycle,event,index,tag,result,flags");
        $dumpfile("lane.vcd");$dumpvars(1,tb_fp32_lane);
        repeat(2)@(negedge clk);reset=0;
        check(!ready,"reset requires explicit accumulator initialization");
        for(i=0;i<64;i=i+1)begin
            @(negedge clk);init_valid=1;init_group=i;init_value=biases[i];
            #1;check(init_ready,"drained lane accepts initialization");
        end
        @(negedge clk);init_valid=0;
        for(i=0;i<COUNT;i=i+1)begin
            @(negedge clk);valid=1;tag=vectors[i][119:104];a=vectors[i][103:72];b=vectors[i][71:40];
            @(posedge clk);while(!ready)@(posedge clk);
            if(i==1024)$dumpoff;
        end
        @(negedge clk);valid=0;
        while(returned<COUNT)@(negedge clk);
        check(sent==COUNT,"every accepted request retired");
        if(GROUPS>=16)check(max_run>=128,"lane sustains one result per clock for independent rows");
        if(GROUPS==1)check(blocked>0 && peak_credits==16,"same-row recurrence backpressures a full product queue");
        @(negedge clk);check(init_ready,"all local writes drained");
        // Flush pending products and busy rows, then verify reinitialization is
        // required. There is no automatic restart after arithmetic failures.
        checking=0;valid=1;tag=0;a=32'h3f800000;b=32'h3f800000;
        repeat(2)@(negedge clk);valid=0;
        repeat(10)@(negedge clk);
        check(dut.busy!=0,"reset test reaches an in-flight accumulator update");
        reset=1;
        @(negedge clk);reset=0;valid=0;
        repeat(16)begin @(posedge clk);#1;check(!response_valid && !ready,"reset flushes products/adder and invalidates accumulators");end
        $display("RESULT {\"status\":\"passed\",\"groups\":%0d,\"operations\":%0d,\"assertions\":%0d,\"cycles_first_issue_to_last_write\":%0d,\"min_result_gap\":%0d,\"max_result_gap\":%0d,\"max_consecutive_results\":%0d,\"backpressure_cycles\":%0d,\"peak_product_credits\":%0d,\"reset_cancelled_requests\":2}",GROUPS,returned,checks,last_write-first_issue,min_gap,max_gap,max_run,blocked,peak_credits);
        $fclose(trace);$finish;
    end
    initial begin #20000000;$fatal(1,"lane watchdog");end
endmodule
`default_nettype wire
