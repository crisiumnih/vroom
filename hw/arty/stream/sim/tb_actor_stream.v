`timescale 1ns/1ps
`default_nettype none
module tb_actor_stream;
    parameter integer LANES=12;
    reg clk=0;always #5 clk=~clk;
    reg reset=1,job_valid=0,job_actor=0;
    reg [31:0] job_id=0;
    wire job_ready;
    reg input_valid=0,input_last=0;
    reg [31:0] input_data=0;
    wire input_ready;
    reg bias_valid=0,bias_last=0,weight_valid=0,weight_last=0;
    reg [LANES*32-1:0] bias_data=0,weight_data=0;
    reg [31:0] bias_job=0,weight_job=0;
    reg [1:0] bias_layer=0,weight_layer=0;
    reg [5:0] bias_group=0,weight_group=0;
    reg [7:0] weight_column=0;
    wire bias_ready,weight_ready,output_valid;
    reg output_ready=0;
    wire [63:0] output_data;
    wire output_actor;
    wire [31:0] output_job;
    wire [4:0] arithmetic_flags;
    wire [3:0] fault_code,debug_state;
    wire [1:0] layer;
    vroom_actor_stream #(.LANES(LANES)) dut(
        clk,reset,job_valid,job_actor,job_id,job_ready,input_valid,input_data,input_last,input_ready,
        bias_valid,bias_data,bias_job,bias_layer,bias_group,bias_last,bias_ready,
        weight_valid,weight_data,weight_job,weight_layer,weight_column,weight_group,weight_last,weight_ready,
        output_valid,output_ready,output_data,output_actor,output_job,arithmetic_flags,fault_code,debug_state,layer);
    reg [31:0] inputs[0:24],weights[0:72447],biases[0:513],pre[0:513],post[0:513],outputs[0:1];
    reg [7:0] expected_flags[0:0];
    reg [4095:0] filename;
    integer cycles=0,checks=0,job=0,li,col,g,l,rows,cols,groups,weight_base,bias_base,nin,mode,seed;
    integer event_file,activation_file,job_file,jobs_done=0,layer_checks=0,blocked=0,source_bubbles=0;
    integer start_cycle,first_cycle,stall_length,i,protocol_cases=0,protocol_file;
    reg check_job=0;
    reg [31:0] rng;
    reg [101:0] held;
    always @(posedge clk)cycles=cycles+1;
    task check;
        input condition;input [767:0] message;
        begin checks=checks+1;if(condition!==1'b1)$fatal(1,"%0s job=%0d layer=%0d cycle=%0d state=%0d fault=%0d",message,job,layer,cycles,debug_state,fault_code);end
    endtask
    task bubble;
        begin
            rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);
            if(mode!=0 && rng[2:0]==0)begin
                @(negedge clk);bias_valid=0;weight_valid=0;input_valid=0;
                repeat(2)@(negedge clk);source_bubbles=source_bubbles+3;
            end
        end
    endtask
    task reset_all;
        begin
            @(negedge clk);reset=1;job_valid=0;input_valid=0;bias_valid=0;weight_valid=0;output_ready=0;check_job=0;
            @(negedge clk);reset=0;
            #1;check(job_ready && !output_valid && fault_code==0,"explicit reset restores idle");
        end
    endtask
    task probe_inputs;
        input [31:0] value;
        integer n;
        begin
            reset_all();
            @(negedge clk);job_valid=1;job_actor=0;job_id=55;
            @(negedge clk);job_valid=0;
            for(n=0;n<25;n=n+1)begin
                input_valid=1;input_data=value;input_last=n==24;
                @(posedge clk);check(input_ready,"probe input accepted");
                @(negedge clk);
            end
            input_valid=0;
        end
    endtask
    task probe_biases;
        input integer which_layer;
        integer n,count;
        begin
            count=which_layer==2 ? 1 : (256+LANES-1)/LANES;
            for(n=0;n<count;n=n+1)begin
                @(negedge clk);bias_valid=1;bias_data=0;bias_job=55;
                bias_layer=which_layer;bias_group=n;bias_last=n==count-1;
                @(posedge clk);while(!bias_ready)@(posedge clk);
            end
            @(negedge clk);bias_valid=0;
        end
    endtask
    task expect_fault;
        input [3:0] expected;
        input [767:0] description;
        begin
            repeat(20)begin
                @(negedge clk);
                check(fault_code==expected && !output_valid && !job_ready &&
                      !input_ready && !bias_ready && !weight_ready,description);
            end
            protocol_cases=protocol_cases+1;
            $fdisplay(protocol_file,"%0d,%0s,%0d,passed",cycles,description,expected);
            reset_all();
        end
    endtask
    task protocol_tests;
        integer test_index,n,cl,gr,ly,nc,ng,timeout_cycles;
        begin
            // Every invalid accepted transfer faults, suppresses all handshakes,
            // and requires an explicit joint source/DUT reset.
            for(test_index=0;test_index<2;test_index=test_index+1)begin
                @(negedge clk);job_valid=1;job_actor=0;job_id=55;
                @(negedge clk);job_valid=0;input_valid=1;input_last=test_index==0;
                input_data=test_index==0 ? 0 : 32'h7f800000;
                @(negedge clk);input_valid=0;
                expect_fault(1,test_index==0 ? "input last mismatch" : "nonfinite input");
            end
            for(test_index=0;test_index<5;test_index=test_index+1)begin
                probe_inputs(32'h3f800000);
                @(negedge clk);bias_valid=1;bias_data=0;bias_job=55;bias_layer=0;bias_group=0;bias_last=0;
                case(test_index)
                    0:bias_job=54;
                    1:bias_layer=1;
                    2:bias_group=1;
                    3:bias_last=1;
                    4:bias_data[31:0]=32'h7f800001;
                endcase
                @(posedge clk);while(!bias_ready)@(posedge clk);
                @(negedge clk);bias_valid=0;
                expect_fault(2,"bad bias metadata or active value");
            end
            for(test_index=0;test_index<6;test_index=test_index+1)begin
                probe_inputs(32'h3f800000);probe_biases(0);
                @(negedge clk);weight_valid=1;weight_data=0;weight_job=55;
                weight_layer=0;weight_column=0;weight_group=0;weight_last=0;
                case(test_index)
                    0:weight_job=54;
                    1:weight_layer=1;
                    2:weight_column=1;
                    3:weight_group=1;
                    4:weight_last=1;
                    5:weight_data[31:0]=32'hff800000;
                endcase
                @(posedge clk);while(!weight_ready)@(posedge clk);
                @(negedge clk);weight_valid=0;
                expect_fault(3,"bad weight metadata or active value");
            end
            // All operands are finite, but their rounded product overflows.
            probe_inputs(32'h7f7fffff);probe_biases(0);
            @(negedge clk);weight_valid=1;weight_data={LANES{32'h40000000}};weight_job=55;
            weight_layer=0;weight_column=0;weight_group=0;weight_last=0;
            @(posedge clk);while(!weight_ready)@(posedge clk);
            @(negedge clk);weight_valid=0;timeout_cycles=0;
            while(fault_code==0 && timeout_cycles<100)begin @(negedge clk);timeout_cycles=timeout_cycles+1;end
            expect_fault(4,"finite arithmetic overflow invalidates job");
            // Reset while real adders own live accumulators, not just while idle.
            probe_inputs(32'h3f800000);probe_biases(0);
            for(n=0;n<4;n=n+1)begin
                @(negedge clk);weight_valid=1;weight_data={LANES{32'h3f800000}};weight_job=55;
                weight_layer=0;weight_column=0;weight_group=n;weight_last=0;
                @(posedge clk);while(!weight_ready)@(posedge clk);
            end
            @(negedge clk);weight_valid=0;
            repeat(10)@(negedge clk);
            check(dut.engine.busy!=0,"reset probe reaches live FP32 accumulation");
            reset_all();
            repeat(40)begin @(negedge clk);check(job_ready && !output_valid && fault_code==0 && dut.lane_valid==0,"no stale arithmetic after reset");end
            protocol_cases=protocol_cases+1;$fdisplay(protocol_file,"%0d,reset during FP32 accumulation,0,passed",cycles);
            // Supplemental zero network: cancel a valid output held by its
            // consumer, then recover with independently checked jobs 2 and 3.
            probe_inputs(0);
            for(ly=0;ly<3;ly=ly+1)begin
                probe_biases(ly);nc=ly==0 ? 25 : 256;ng=ly==2 ? 1 : (256+LANES-1)/LANES;
                for(cl=0;cl<nc;cl=cl+1)for(gr=0;gr<ng;gr=gr+1)begin
                    @(negedge clk);weight_valid=1;weight_data=0;weight_job=55;
                    weight_layer=ly;weight_column=cl;weight_group=gr;weight_last=cl==nc-1 && gr==ng-1;
                    @(posedge clk);while(!weight_ready)@(posedge clk);
                end
                @(negedge clk);weight_valid=0;
            end
            while(!output_valid)begin @(negedge clk);check(fault_code==0,"zero network remains valid");end
            check(output_data==0 && arithmetic_flags==0 && output_job==55,"zero network yields exact zero packet");
            repeat(5)@(negedge clk);
            reset_all();
            repeat(40)begin @(negedge clk);check(job_ready && !output_valid && fault_code==0,"held output cancelled by reset");end
            protocol_cases=protocol_cases+1;$fdisplay(protocol_file,"%0d,reset with output backpressure,0,passed",cycles);
        end
    endtask
    // Observe full layer outputs directly at the real arithmetic writeback.
    genvar lane;
    generate for(lane=0;lane<LANES;lane=lane+1)begin: monitor
        integer row,index;
        always @(posedge clk)begin
            if(!reset && check_job && dut.lane_valid[lane] && dut.lanes[lane].final_column)begin
                row=dut.lane_tag[lane*16+:6]*LANES+lane;
                index=(layer==0 ? 0 : layer==1 ? 256 : 512)+row;
                check(row<(layer==2 ? 2 : 256),"no retirement from masked tail lanes");
                if(dut.lane_result[lane*32+:32]!==pre[index])
                    $display("ACTIVATION row=%0d got=%08x expected=%08x",row,dut.lane_result[lane*32+:32],pre[index]);
                check(dut.lane_result[lane*32+:32]===pre[index],"every dense-layer output matches exact RN32 reference");
                if(layer<2)check(dut.lanes[lane].relu===post[index],"ReLU output matches reference");
                layer_checks=layer_checks+1;
                $fdisplay(activation_file,"%0d,%0d,%0d,%0d,%08x",cycles,job,layer,row,dut.lane_result[lane*32+:32]);
            end
        end
    end endgenerate
    always @(posedge clk)begin
        if(!reset && !dut.lane_reset)begin
            check(dut.engine.credits<=16,"SIMD credits include multiplications in flight");
            check(dut.engine.queued<=dut.engine.credits,"visible queue cannot exceed reserved credits");
            check(dut.engine.front==dut.engine.queued[3:0]-4'd1,"registered SRL tap equals queue count minus one");
            check(dut.engine.head_count<=2,"registered queue head capacity");
            check(!dut.engine.pop || (dut.engine.queued!=0 && dut.engine.head_count<2),"registered pop reserves available data and head space");
            check(!dut.engine.init_ready || (dut.engine.credits==0 && dut.engine.head_count==0 && dut.engine.busy==0),"initialization ready never overlaps live arithmetic");
        end
        if(!reset && check_job)begin
            check(fault_code==0,"valid job must not fault");
            if(weight_valid && !weight_ready)blocked=blocked+1;
            if(job_valid && job_ready)$fdisplay(event_file,"%0d,job,%0d,0,0,0",cycles,job);
            if(input_valid && input_ready)$fdisplay(event_file,"%0d,input,%0d,0,0,0",cycles,job);
            if(bias_valid && bias_ready)$fdisplay(event_file,"%0d,bias,%0d,%0d,0,%0d",cycles,job,bias_layer,bias_group);
            if(weight_valid && weight_ready)$fdisplay(event_file,"%0d,weight,%0d,%0d,%0d,%0d",cycles,job,weight_layer,weight_column,weight_group);
            if(dut.issue)$fdisplay(event_file,"%0d,issue,%0d,%0d,%0d,%0d",cycles,job,layer,dut.tag_q[13:6],dut.tag_q[5:0]);
            if(output_valid && output_ready)$fdisplay(event_file,"%0d,output,%0d,2,0,0",cycles,job);
        end
    end
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=2805;rng=seed;
        event_file=$fopen("events.csv","w");$fdisplay(event_file,"cycle,event,job,layer,column,group");
        activation_file=$fopen("activations.csv","w");$fdisplay(activation_file,"cycle,job,layer,row,value");
        job_file=$fopen("jobs.csv","w");$fdisplay(job_file,"job,actor,mode,accepted_edge,first_output_edge,latency,output0,output1,flags,output_stall_cycles");
        protocol_file=$fopen("protocol.csv","w");$fdisplay(protocol_file,"cycle,case,fault_code,status");
        $dumpfile("stream.vcd");$dumpvars(1,tb_actor_stream);$dumpvars(1,dut);
        repeat(2)@(negedge clk);reset=0;
        for(job=0;job<4;job=job+1)begin
            nin=job%2 ? 19 : 25;mode=job/2;
            $sformat(filename,"job%0d-input.hex",job);$readmemh(filename,inputs,0,nin-1);
            $sformat(filename,"job%0d-weight.hex",job);$readmemh(filename,weights,0,nin*256+65536+512-1);
            $sformat(filename,"job%0d-bias.hex",job);$readmemh(filename,biases);
            $sformat(filename,"job%0d-pre.hex",job);$readmemh(filename,pre);
            $sformat(filename,"job%0d-post.hex",job);$readmemh(filename,post);
            $sformat(filename,"job%0d-output.hex",job);$readmemh(filename,outputs);
            $sformat(filename,"job%0d-flags.hex",job);$readmemh(filename,expected_flags);
            @(negedge clk);check(job_ready,"new job ready after previous output");
            job_valid=1;job_actor=job%2;job_id=32'h12340000+job;check_job=1;
            @(posedge clk);#1;start_cycle=cycles;
            @(negedge clk);job_valid=0;
            for(i=0;i<nin;i=i+1)begin
                bubble();@(negedge clk);input_valid=1;input_data=inputs[i];input_last=i==nin-1;
                @(posedge clk);while(!input_ready)@(posedge clk);
            end
            @(negedge clk);input_valid=0;weight_base=0;bias_base=0;
            for(li=0;li<3;li=li+1)begin
                rows=li==2 ? 2 : 256;cols=li==0 ? nin : 256;groups=(rows+LANES-1)/LANES;
                for(g=0;g<groups;g=g+1)begin
                    bubble();@(negedge clk);bias_valid=1;bias_job=job_id;bias_layer=li;bias_group=g;bias_last=g==groups-1;
                    // Poison inactive slots: they cannot publish results/flags.
                    bias_data={LANES{32'h7f800001}};
                    for(l=0;l<LANES;l=l+1)if(g*LANES+l<rows)bias_data[l*32+:32]=biases[bias_base+g*LANES+l];
                    @(posedge clk);while(!bias_ready)@(posedge clk);
                end
                @(negedge clk);bias_valid=0;
                for(col=0;col<cols;col=col+1)begin
                    for(g=0;g<groups;g=g+1)begin
                        bubble();@(negedge clk);weight_valid=1;weight_job=job_id;weight_layer=li;weight_column=col;weight_group=g;
                        weight_last=(col==cols-1 && g==groups-1);weight_data={LANES{32'h7f800001}};
                        for(l=0;l<LANES;l=l+1)if(g*LANES+l<rows)weight_data[l*32+:32]=weights[weight_base+col*rows+g*LANES+l];
                        @(posedge clk);while(!weight_ready)@(posedge clk);
                    end
                end
                @(negedge clk);weight_valid=0;weight_base=weight_base+cols*rows;bias_base=bias_base+rows;
            end
            while(!output_valid)@(negedge clk);
            @(posedge clk);#1;first_cycle=cycles;
            check(output_data==={outputs[1],outputs[0]} && output_job==job_id && output_actor==job_actor,"complete tagged actor output");
            check(arithmetic_flags===expected_flags[0][4:0],"job-wide arithmetic exception flags");
            held={output_data,output_actor,output_job,arithmetic_flags};stall_length=mode ? 37 : 0;
            // A queued job cannot overwrite the current packet while stalled.
            job_valid=1;
            repeat(stall_length)begin
                @(posedge clk);#1;
                check(output_valid && !job_ready && {output_data,output_actor,output_job,arithmetic_flags}===held,"output packet held under backpressure");
            end
            @(negedge clk);job_valid=0;output_ready=1;
            @(posedge clk);#1;check(job_ready && !output_valid,"output accepted once");
            @(negedge clk);output_ready=0;check_job=0;jobs_done=jobs_done+1;
            $fdisplay(job_file,"%0d,%0d,%0d,%0d,%0d,%0d,%08x,%08x,%02x,%0d",job,job_actor,mode,start_cycle,first_cycle,first_cycle-start_cycle,output_data[31:0],output_data[63:32],arithmetic_flags,stall_length);
            $display("JOB %0d actor=%0d mode=%0d cycles=%0d",job,job_actor,mode,first_cycle-start_cycle);
            if(job==0)$dumpoff;
            if(job==1)protocol_tests();
        end
        check(layer_checks==4*514,"all intermediate layer values checked");
        check(source_bubbles>0 && blocked>0,"source stalls and lane credit backpressure covered");
        check(protocol_cases==16,"malformed transfers arithmetic faults and reset cases covered");
        $display("RESULT {\"status\":\"passed\",\"lanes\":%0d,\"jobs\":%0d,\"layer_values_checked\":%0d,\"protocol_cases\":%0d,\"assertions\":%0d,\"source_bubble_cycles\":%0d,\"weight_backpressure_cycles\":%0d}",LANES,jobs_done,layer_checks,protocol_cases,checks,source_bubbles,blocked);
        $fclose(event_file);$fclose(activation_file);$fclose(job_file);$fclose(protocol_file);$finish;
    end
    initial begin #50000000;$fatal(1,"watchdog");end
endmodule
`default_nettype wire
