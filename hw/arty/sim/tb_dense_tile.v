`timescale 1ns/1ps
`default_nettype none
module tb_dense_tile;
    parameter integer LANES=12;
    parameter integer MAC_LATENCY=19;
    reg clk=0;
    reg reset=1;
    reg context_valid=0;
    wire context_accept;
    reg context_actor=0;
    reg [31:0] context_job=0;
    reg [1:0] context_layer=0;
    reg [8:0] context_inputs=0;
    reg [8:0] context_outputs=0;
    reg input_valid=0;
    wire input_ready;
    reg [31:0] input_data=0;
    reg bias_valid=0;
    wire bias_ready;
    reg [31:0] bias_data=0;
    wire [1:0] context_ready;
    wire [1:0] context_complete;
    reg command_valid=0;
    wire command_ready;
    reg command_actor=0;
    reg [31:0] command_job=0;
    reg [1:0] command_layer=0;
    reg [1:0] command_buffer=0;
    reg [31:0] command_tag=0;
    reg [7:0] command_row=0;
    reg [7:0] command_rows=0;
    reg [7:0] command_column=0;
    reg [2:0] command_columns=0;
    wire done_valid;
    reg done_ready=0;
    wire active_actor;
    wire [31:0] active_job;
    wire [1:0] active_layer;
    wire mac_valid;
    wire mac_ready;
    wire [31:0] mac_tag;
    wire [LANES-1:0] mac_mask;
    wire [31:0] mac_input;
    wire [LANES*32-1:0] mac_weight;
    wire [LANES*32-1:0] mac_accumulator;
    wire mac_response_valid;
    wire mac_response_ready;
    wire [31:0] mac_response_tag;
    wire [LANES*32-1:0] mac_response_data;
    reg result_valid=0;
    wire result_ready;
    reg result_actor=0;
    reg [7:0] result_row=0;
    reg [LANES-1:0] result_mask=0;
    wire result_response_valid;
    wire [LANES-1:0] result_response_mask;
    wire [LANES*32-1:0] result_response_data;
    wire [31:0] result_response_job;
    wire [1:0] result_response_layer;
    wire [3:0] state;
    wire [3:0] fault_code;
    reg load_valid=0;
    wire load_ready;
    reg [1:0] load_buffer=0;
    reg [31:0] load_tag=0;
    reg [9:0] load_words=0;
    reg dma_valid=0;
    wire dma_ready;
    reg [1:0] dma_buffer=0;
    reg [31:0] dma_tag=0;
    reg [127:0] dma_data=0;
    reg dma_last=0;
    reg status_valid=0;
    wire status_ready;
    reg [1:0] status_buffer=0;
    reg [31:0] status_tag=0;
    reg status_ok=0;
    reg [11:0] status_bytes=0;
    wire [7:0] buffer_states;
    wire [3:0] buffer_faults;
    wire [15:0] buffer_fault_codes;
    wire fault;
    always #5 clk=~clk;
    vroom_dense_tile_core #(.LANES(LANES)) dut (
        .clk(clk),
        .reset(reset),
        .context_valid(context_valid),
        .context_accept(context_accept),
        .context_actor(context_actor),
        .context_job(context_job),
        .context_layer(context_layer),
        .context_inputs(context_inputs),
        .context_outputs(context_outputs),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .input_data(input_data),
        .bias_valid(bias_valid),
        .bias_ready(bias_ready),
        .bias_data(bias_data),
        .context_ready(context_ready),
        .context_complete(context_complete),
        .command_valid(command_valid),
        .command_ready(command_ready),
        .command_actor(command_actor),
        .command_job(command_job),
        .command_layer(command_layer),
        .command_buffer(command_buffer),
        .command_tag(command_tag),
        .command_row(command_row),
        .command_rows(command_rows),
        .command_column(command_column),
        .command_columns(command_columns),
        .done_valid(done_valid),
        .done_ready(done_ready),
        .active_actor(active_actor),
        .active_job(active_job),
        .active_layer(active_layer),
        .mac_valid(mac_valid),
        .mac_ready(mac_ready),
        .mac_tag(mac_tag),
        .mac_mask(mac_mask),
        .mac_input(mac_input),
        .mac_weight(mac_weight),
        .mac_accumulator(mac_accumulator),
        .mac_response_valid(mac_response_valid),
        .mac_response_ready(mac_response_ready),
        .mac_response_tag(mac_response_tag),
        .mac_response_data(mac_response_data),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_actor(result_actor),
        .result_row(result_row),
        .result_mask(result_mask),
        .result_response_valid(result_response_valid),
        .result_response_mask(result_response_mask),
        .result_response_data(result_response_data),
        .result_response_job(result_response_job),
        .result_response_layer(result_response_layer),
        .state(state),
        .fault_code(fault_code),
        .load_valid(load_valid),
        .load_ready(load_ready),
        .load_buffer(load_buffer),
        .load_tag(load_tag),
        .load_words(load_words),
        .dma_valid(dma_valid),
        .dma_ready(dma_ready),
        .dma_buffer(dma_buffer),
        .dma_tag(dma_tag),
        .dma_data(dma_data),
        .dma_last(dma_last),
        .status_valid(status_valid),
        .status_ready(status_ready),
        .status_buffer(status_buffer),
        .status_tag(status_tag),
        .status_ok(status_ok),
        .status_bytes(status_bytes),
        .buffer_states(buffer_states),
        .buffer_faults(buffer_faults),
        .buffer_fault_codes(buffer_fault_codes),
        .fault(fault)
    );
    reg force_stall=0, corrupt_tag=0, inject_response=0;
    wire model_response_valid;
    wire [31:0] model_response_tag;
    wire [LANES*32-1:0] model_response_data;
    assign mac_response_valid=model_response_valid || inject_response;
    assign mac_response_tag=inject_response ? 32'hdeadbeef : model_response_tag;
    assign mac_response_data=inject_response ? 0 : model_response_data;
    fp32_mac_model #(.LANES(LANES),.LATENCY(MAC_LATENCY)) operators (
        .clk(clk),.reset(reset),.force_stall(force_stall),.corrupt_tag(corrupt_tag),
        .valid(mac_valid),.ready(mac_ready),.tag(mac_tag),.mask(mac_mask),
        .activation(mac_input),.weight(mac_weight),.accumulator(mac_accumulator),
        .response_valid(model_response_valid),.response_ready(mac_response_ready),
        .response_tag(model_response_tag),.response_data(model_response_data)
    );
    reg [31:0] fixture_inputs[0:1023], fixture_bias[0:1023], fixture_expected[0:1023];
    reg [31:0] fixture_weights[0:262143], small_expected[0:1];
    reg [31:0] alignment_expected[0:255];
    reg [4095:0] wave_path;
    integer seed, checks=0, cycles=0, accepted=0, returned=0;
    integer scalar_macs=0, peak_inflight=0, overlap_beats=0, stalled_cycles=0, full_cycles=0;
    integer verified_outputs=0, completed_tiles=0, numerical_macs=0, negative_cases=0;
    integer alignment_macs=0, result_windows=0;
    reg held=0, done_previous=0;
    reg [LANES*65+63:0] held_payload;
    wire [LANES*65+63:0] payload={mac_tag,mac_mask,mac_input,mac_weight,mac_accumulator};
    integer monitor_lane;
    always @(posedge clk) begin
        cycles=cycles+1;
        if(reset) begin accepted=0; returned=0; held=0; end
        else begin
            if(held) check(mac_valid && payload===held_payload,"MAC request stable under backpressure");
            held=mac_valid && !mac_ready;
            held_payload=payload;
            if(mac_valid && mac_ready) begin
                accepted=accepted+1;
                for(monitor_lane=0;monitor_lane<LANES;monitor_lane=monitor_lane+1)
                    if(mac_mask[monitor_lane]) scalar_macs=scalar_macs+1;
            end
            if(mac_response_valid && mac_response_ready) returned=returned+1;
            if(accepted-returned>peak_inflight) peak_inflight=accepted-returned;
            if(dma_valid && dma_ready && (state==4 || state==5)) overlap_beats=overlap_beats+1;
            if(mac_valid && !mac_ready) stalled_cycles=stalled_cycles+1;
            if(dut.engine.queue_count==4) full_cycles=full_cycles+1;
        end
        #1;
        if(!reset && done_valid && !done_previous)
            $display("DENSE_COMPLETE %0d,%0d,%0d,%0d",cycles,active_actor,active_layer,command_tag);
        done_previous=done_valid;
    end
    task check;
        input condition;
        input [1023:0] message;
        begin checks=checks+1; if(condition !== 1'b1) $fatal(1,"%0s (state=%0d fault=%0d code=%0d)",message,state,fault,fault_code); end
    endtask
    task step;
        begin @(posedge clk); #1; end
    endtask
    task idle;
        begin
            context_valid=0; input_valid=0; bias_valid=0; command_valid=0; result_valid=0;
            done_ready=0; load_valid=0; dma_valid=0; status_valid=0;
            force_stall=0; corrupt_tag=0; inject_response=0;
        end
    endtask
    task restart;
        begin
            @(negedge clk); idle; reset=1; repeat(2) step;
            @(negedge clk); reset=0; step;
            check(!fault && context_ready==0 && buffer_states==0,"joint reset invalidates contexts and buffers");
        end
    endtask
    task load_context;
        input integer actor_id, fixture, nin, nout, job_id, layer_id;
        integer index;
        begin
            @(negedge clk); context_valid=1; context_actor=actor_id; context_job=job_id;
            context_layer=layer_id; context_inputs=nin; context_outputs=nout;
            #1; check(context_accept,"context header accepted"); step;
            @(negedge clk); context_valid=0;
            for(index=0;index<nin || index<nout;index=index+1) begin
                @(negedge clk);
                input_valid=(index<nin); bias_valid=(index<nout);
                input_data=fixture_inputs[fixture*256+index]; bias_data=fixture_bias[fixture*256+index];
                #1; if(input_valid) check(input_ready,"activation initialization ready");
                if(bias_valid) check(bias_ready,"bias initialization ready"); step;
            end
            @(negedge clk); input_valid=0; bias_valid=0;
            repeat(2) step;
            check(context_ready[actor_id] && !context_complete[actor_id] && !fault,"initialized independent actor context");
        end
    endtask
    task fill_tile;
        input integer fixture, buffer_id, row, rows, column, columns, tag_id;
        integer word_id, lane_id, linear;
        begin
            @(negedge clk); load_valid=1; load_buffer=buffer_id; load_tag=tag_id; load_words=rows*columns;
            #1; check(load_ready,"DMA allocation ready"); step;
            @(negedge clk); load_valid=0;
            for(word_id=0;word_id<rows*columns;word_id=word_id+4) begin
                if((word_id/4)%7==3) begin @(negedge clk); dma_valid=0; step; end
                @(negedge clk); dma_valid=1; dma_buffer=buffer_id; dma_tag=tag_id;
                dma_last=(word_id+4==rows*columns);
                for(lane_id=0;lane_id<4;lane_id=lane_id+1) begin
                    linear=word_id+lane_id;
                    dma_data[lane_id*32 +: 32]=fixture_weights[fixture*65536+(row+linear%rows)*256+column+linear/rows];
                end
                #1; check(dma_ready,"DMA payload ready while other actor computes"); step;
                check(!fault,"valid payload");
            end
            @(negedge clk); dma_valid=0; status_valid=1; status_buffer=buffer_id;
            status_tag=tag_id; status_bytes=rows*columns*4; status_ok=1;
            #1; check(status_ready,"DMA completion ready"); step;
            @(negedge clk); status_valid=0;
        end
    endtask
    task launch;
        input integer actor_id, job_id, layer_id, buffer_id, row, rows, column, columns, tag_id;
        begin
            @(negedge clk); command_valid=1; command_actor=actor_id; command_job=job_id; command_layer=layer_id;
            command_buffer=buffer_id; command_tag=tag_id; command_row=row; command_rows=rows;
            command_column=column; command_columns=columns;
            #1; check(command_ready,"tile descriptor ready"); step;
            $display("DENSE_EVENT %0d,start,%0d,%0d,%0d,%0d,%0d,%0d",cycles,actor_id,tag_id,row,rows,column,columns);
            @(negedge clk); command_valid=0;
        end
    endtask
    task finish_tile;
        input integer actor_id, tag_id;
        integer guard;
        begin
            guard=0;
            while(!done_valid && guard<10000) begin step; check(!fault,"tile executing without fault"); guard=guard+1; end
            check(done_valid && active_actor==actor_id,"microtile completed for requested actor");
            check(accepted==returned,"all FP32 results drained before actor switch");
            check(buffer_states[command_buffer*2 +: 2]==0,"tile released only after arithmetic drains");
            repeat(3) begin step; check(done_valid && active_actor==actor_id && !command_ready,"completion held under backpressure"); end
            $display("DENSE_EVENT %0d,done,%0d,%0d,%0d,%0d,%0d,%0d",cycles,actor_id,tag_id,command_row,command_rows,command_column,command_columns);
            @(negedge clk); done_ready=1; step;
            @(negedge clk); done_ready=0;
            completed_tiles=completed_tiles+1;
        end
    endtask
    task verify_results;
        input integer actor_id, fixture, count, job_id, layer_id;
        integer base, nvalid, lane_id;
        reg [31:0] expected_word;
        begin
            check(context_complete[actor_id],"all layer rows complete");
            for(base=0;base<count;base=base+LANES) begin
                nvalid=(count-base<LANES)?count-base:LANES;
                @(negedge clk); result_valid=1; result_actor=actor_id; result_row=base;
                result_mask=({LANES{1'b1}} >> (LANES-nvalid));
                #1; check(result_ready,"completed layer readable"); step;
                check(result_response_valid && result_response_mask==result_mask,"result response and tail mask");
                check(result_response_job==job_id && result_response_layer==layer_id,"result context identity");
                for(lane_id=0;lane_id<nvalid;lane_id=lane_id+1) begin
                    expected_word=(fixture==5) ? alignment_expected[base+lane_id] :
                                  (fixture==4) ? small_expected[base+lane_id] : fixture_expected[fixture*256+base+lane_id];
                    if(result_response_data[lane_id*32 +: 32]!==expected_word)
                        $display("Mismatch fixture=%0d row=%0d got=%08x expected=%08x",fixture,base+lane_id,result_response_data[lane_id*32 +: 32],expected_word);
                    check(result_response_data[lane_id*32 +: 32]===expected_word,"FP32 result bit-exact against Python reference");
                    $display("DENSE_RESULT %0d,%0d,%08x",fixture,base+lane_id,result_response_data[lane_id*32 +: 32]);
                    verified_outputs=verified_outputs+1;
                end
            end
            @(negedge clk); result_valid=0;
        end
    endtask
    task verify_alignment_windows;
        integer base, lane_id, mode;
        reg [31:0] expected_word;
        begin
            for(base=0;base<256;base=base+1) begin
                for(mode=0;mode<2;mode=mode+1) begin
                    @(negedge clk); result_valid=1; result_actor=1; result_row=base;
                    for(lane_id=0;lane_id<LANES;lane_id=lane_id+1)
                        result_mask[lane_id]=(base+lane_id<256) && (mode==0 || (lane_id%2)==(base%2));
                    #1; check(result_ready,"unaligned result accepted"); step;
                    check(result_response_valid && result_response_mask==result_mask,"unaligned result response");
                    for(lane_id=0;lane_id<LANES;lane_id=lane_id+1) begin
                        expected_word=result_mask[lane_id] ? alignment_expected[base+lane_id] : 32'b0;
                        check(result_response_data[lane_id*32 +: 32]===expected_word,"all result alignments and sparse masks");
                    end
                    result_windows=result_windows+1;
                end
            end
            @(negedge clk); result_valid=0;
        end
    endtask
    task expect_fault;
        input integer code;
        integer guard;
        begin
            guard=0;
            while(!fault && guard<1000) begin step; guard=guard+1; end
            check(fault && fault_code==code,"expected engine fault");
            check(!command_ready && !mac_valid && !done_valid && !load_ready,"fault halts all commands and DMA");
            @(negedge clk); idle; repeat(3) step;
            check(fault && fault_code==code,"fault sticky until explicit reset");
            negative_cases=negative_cases+1;
        end
    endtask
    integer column, part, row, rows, inner_cols, outer_cols, inner_buf, outer_buf, inner_tag, outer_tag, next_tag=1;
    initial begin
        if(!$value$plusargs("vcd=%s",wave_path)) wave_path="dense.vcd";
        if(!$value$plusargs("seed=%d",seed)) seed=2805;
        $dumpfile(wave_path); $dumpvars(0,tb_dense_tile);
        $readmemh("dense-inputs.hex",fixture_inputs);
        $readmemh("dense-bias.hex",fixture_bias);
        $readmemh("dense-weights.hex",fixture_weights);
        $readmemh("dense-expected.hex",fixture_expected);
        $readmemh("dense-small-expected.hex",small_expected);
        $readmemh("dense-alignment-expected.hex",alignment_expected);
        $display("CASE independent contexts, interleaved tiles, prefetch overlap, tails, and stalls");
        restart;
        // Model canaries: a fused implementation gives a different result for
        // the first case. Preserve subnormal arithmetic in the second case.
        check($vroom_fp32_mac(32'h3f800001,32'h3f7fffff,32'hbf800000)==0,"multiply and add must round separately");
        check($vroom_fp32_mac(32'h00000001,32'h3f800000,32'h00000001)==32'h00000002,"gradual underflow model");
        load_context(0,0,25,256,17,0);
        load_context(1,1,19,256,29,0);
        for(column=0;column<25;column=column+4) begin
            inner_cols=(25-column<4)?25-column:4;
            outer_cols=(19-column<4)?19-column:4;
            inner_buf=(column/4)%2; outer_buf=2+inner_buf;
            for(part=0;part<3;part=part+1) begin
                case(part)
                    0: begin row=0; rows=124; end
                    1: begin row=124; rows=128; end
                    default: begin row=252; rows=4; end
                endcase
                inner_tag=next_tag; next_tag=next_tag+1;
                if(column<19) begin
                    outer_tag=next_tag; next_tag=next_tag+1;
                    fill_tile(1,outer_buf,row,rows,column,outer_cols,outer_tag);
                    if(column==0 && part==0) force_stall=1;
                    launch(1,29,0,outer_buf,row,rows,column,outer_cols,outer_tag);
                    fill_tile(0,inner_buf,row,rows,column,inner_cols,inner_tag);
                    force_stall=0;
                    finish_tile(1,outer_tag);
                end else fill_tile(0,inner_buf,row,rows,column,inner_cols,inner_tag);
                launch(0,17,0,inner_buf,row,rows,column,inner_cols,inner_tag);
                finish_tile(0,inner_tag);
            end
        end
        verify_results(1,1,256,29,0); verify_results(0,0,256,17,0);
        $display("CASE full 256 by 256 middle layer, all column addresses and ring wraparound");
        load_context(0,3,256,256,17,1);
        for(column=0;column<256;column=column+4) begin
            for(part=0;part<2;part=part+1) begin
                row=part*128; inner_tag=next_tag; next_tag=next_tag+1; inner_buf=(column/4)%2;
                fill_tile(3,inner_buf,row,128,column,4,inner_tag);
                launch(0,17,1,inner_buf,row,128,column,4,inner_tag); finish_tile(0,inner_tag);
            end
        end
        verify_results(0,3,256,17,1);
        $display("CASE two-output recurrence and completed-context replacement");
        load_context(0,2,256,2,17,2);
        for(column=0;column<256;column=column+4) begin
            inner_tag=next_tag; next_tag=next_tag+1; inner_buf=(column/4)%2;
            fill_tile(2,inner_buf,0,2,column,4,inner_tag);
            launch(0,17,2,inner_buf,0,2,column,4,inner_tag); finish_tile(0,inner_tag);
        end
        verify_results(0,2,2,17,2);
        numerical_macs=scalar_macs;
        check(numerical_macs==77312 && verified_outputs==770,"all requested numerical work executed exactly once");
        check(peak_inflight>1 && full_cycles>0 && overlap_beats>0 && stalled_cycles>0,"pipeline, queue saturation, prefetch, and stall coverage");
        $display("CASE invalid result range");
        @(negedge clk); result_valid=1; result_actor=0; result_row=1; result_mask=3; step; expect_fault(7);
        $display("CASE stale descriptor job");
        restart; load_context(0,2,8,2,17,0); launch(0,99,0,0,0,2,0,4,1); expect_fault(2);
        $display("CASE wrong actor buffer");
        restart; load_context(0,2,8,2,17,0); launch(0,17,0,2,0,2,0,4,1); expect_fault(1);
        $display("CASE skipped input columns");
        restart; load_context(0,2,8,2,17,0); launch(0,17,0,0,0,2,4,4,1); expect_fault(3);
        $display("CASE duplicate input columns");
        restart; load_context(0,2,8,2,17,0);
        fill_tile(2,0,0,2,0,4,1); launch(0,17,0,0,0,2,0,4,1); finish_tile(0,1);
        launch(0,17,0,0,0,2,0,4,2); expect_fault(3);
        $display("CASE wrong arithmetic response tag");
        restart; load_context(0,2,4,2,17,0); fill_tile(2,0,0,2,0,4,1);
        corrupt_tag=1; launch(0,17,0,0,0,2,0,4,1); expect_fault(5);
        $display("CASE unsolicited arithmetic response");
        restart; @(negedge clk); inject_response=1; step; expect_fault(5);
        $display("CASE reset while arithmetic is in flight, then clean reload");
        restart; load_context(0,2,4,2,17,0); fill_tile(2,0,0,2,0,4,1);
        launch(0,17,0,0,0,2,0,4,1);
        while(accepted==0) step;
        restart; load_context(0,2,4,2,18,0); fill_tile(2,0,0,2,0,4,2);
        launch(0,18,0,0,0,2,0,4,2); finish_tile(0,2); verify_results(0,4,2,18,0);
        $display("CASE every accumulator bank alignment and row tail");
        restart; load_context(1,3,4,256,31,0); alignment_macs=scalar_macs;
        row=0;
        // Prefix sums of lengths 1..16 visit every starting bank exactly once.
        for(part=1;part<=17;part=part+1) begin
            rows=(part==17) ? 120 : part;
            inner_tag=next_tag; next_tag=next_tag+1;
            fill_tile(3,2,row,rows,0,4,inner_tag);
            launch(1,31,0,2,row,rows,0,4,inner_tag); finish_tile(1,inner_tag);
            row=row+rows;
        end
        alignment_macs=scalar_macs-alignment_macs;
        check(row==256 && alignment_macs==1024,"alignment test computes each row exactly once");
        verify_results(1,5,256,31,0); verify_alignment_windows;
        @(negedge clk); result_valid=1; result_actor=1; result_row=255; result_mask=2; step; expect_fault(7);
        $display("RESULT {\"test\":\"dense_tile\",\"status\":\"passed\",\"lanes\":%0d,\"mac_latency\":%0d,\"seed\":%0d,\"assertions\":%0d,\"numerical_scalar_macs\":%0d,\"verified_outputs\":%0d,\"completed_tiles\":%0d,\"peak_inflight_vectors\":%0d,\"overlapped_dma_beats\":%0d,\"operator_stall_cycles\":%0d,\"operand_fifo_full_cycles\":%0d,\"negative_cases\":%0d,\"alignment_scalar_macs\":%0d,\"result_window_reads\":%0d}",LANES,MAC_LATENCY,seed,checks,numerical_macs,verified_outputs,completed_tiles,peak_inflight,overlap_beats,stalled_cycles,full_cycles,negative_cases,alignment_macs,result_windows);
        $finish;
    end
    initial begin #10000000; $fatal(1,"dense watchdog"); end
endmodule
`default_nettype wire
