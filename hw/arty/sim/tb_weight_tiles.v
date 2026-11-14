`timescale 1ns/1ps
`default_nettype none
module tb_weight_tiles;
    parameter integer LANES=12;
    reg clk=0;
    always #5 clk=~clk;
    reg reset=1;
    reg load_valid=0, dma_valid=0, status_valid=0, acquire_valid=0, release_valid=0, read_valid=0;
    wire load_ready, dma_ready, status_ready, acquire_ready, release_ready, read_ready;
    reg [1:0] load_buffer=0, dma_buffer=0, status_buffer=0, acquire_buffer=0, release_buffer=0, read_buffer=0;
    reg [31:0] load_tag=0, dma_tag=0, status_tag=0, acquire_tag=0, release_tag=0;
    reg [9:0] load_words=0;
    reg [127:0] dma_data=0;
    reg dma_last=0, status_ok=1;
    reg [11:0] status_bytes=0;
    reg [8:0] read_base=0;
    reg [LANES-1:0] read_mask=0;
    wire response_valid, fault;
    wire [LANES-1:0] response_mask;
    wire [LANES*32-1:0] response_data;
    wire [7:0] states;
    wire [3:0] fault_buffers;
    wire [15:0] fault_codes;
    integer checks=0, overlap_cycles=0, reads=0, window_reads=0, seed;
    reg [31:0] rng;
    reg [4095:0] wave_path;
    vroom_weight_tiles #(.LANES(LANES)) dut (
        .clk(clk), .reset(reset),
        .load_valid(load_valid), .load_ready(load_ready), .load_buffer(load_buffer), .load_tag(load_tag), .load_words(load_words),
        .dma_valid(dma_valid), .dma_ready(dma_ready), .dma_buffer(dma_buffer), .dma_tag(dma_tag), .dma_data(dma_data), .dma_last(dma_last),
        .status_valid(status_valid), .status_ready(status_ready), .status_buffer(status_buffer), .status_tag(status_tag), .status_ok(status_ok), .status_bytes(status_bytes),
        .acquire_valid(acquire_valid), .acquire_ready(acquire_ready), .acquire_buffer(acquire_buffer), .acquire_tag(acquire_tag),
        .release_valid(release_valid), .release_ready(release_ready), .release_buffer(release_buffer), .release_tag(release_tag),
        .read_valid(read_valid), .read_ready(read_ready), .read_buffer(read_buffer), .read_base(read_base), .read_mask(read_mask),
        .response_valid(response_valid), .response_mask(response_mask), .response_data(response_data),
        .states(states), .fault(fault), .fault_buffers(fault_buffers), .fault_codes(fault_codes)
    );
    function [31:0] pattern;
        input integer buffer_id, word_id;
        begin
            // Exercise exceptional IEEE bit patterns as bits, never arithmetic.
            case (word_id % 17)
                0: pattern=32'h80000000;
                1: pattern=32'h7fc12345;
                2: pattern=32'h00000001;
                3: pattern=32'h7f800000;
                default: pattern=32'h3f800000 ^ (32'h9e3779b9 * word_id) ^ buffer_id;
            endcase
        end
    endfunction
    task check;
        input condition;
        input [1023:0] message;
        begin checks=checks+1; if (condition !== 1'b1) $fatal(1,"%0s",message); end
    endtask
    task verify_windows;
        input integer buf_id, count;
        integer base, mode, k;
        reg [31:0] expected_word;
        begin
            // Actual RAM contents are checked at every base, including bank
            // wrap and the final partial vector. Sparse masks must zero holes.
            for(base=0;base<512;base=base+1) begin
                for(mode=0;mode<2;mode=mode+1) begin
                    @(negedge clk); read_valid=1; read_buffer=buf_id; read_base=base;
                    for(k=0;k<LANES;k=k+1)
                        read_mask[k]=(base+k<count) && (mode==0 || (k%2)==(base%2));
                    #1; check(read_ready,"sliding window accepted"); step;
                    check(response_valid && response_mask==read_mask,"sliding window response");
                    for(k=0;k<LANES;k=k+1) begin
                        expected_word=read_mask[k] ? pattern(buf_id,base+k) : 32'b0;
                        check(response_data[k*32 +: 32]===expected_word,"every alignment and mask hole preserves payload");
                    end
                    window_reads=window_reads+1;
                end
            end
            @(negedge clk); read_valid=0;
        end
    endtask
    task step;
        begin @(posedge clk); #1; end
    endtask
    task idle;
        begin
            load_valid=0; dma_valid=0; status_valid=0; acquire_valid=0; release_valid=0; read_valid=0;
        end
    endtask
    task restart;
        begin @(negedge clk); idle; reset=1; step; @(negedge clk); reset=0; step; check(!fault && states==0,"reset ownership"); end
    endtask
    task allocate;
        input integer buf_id, count;
        input [31:0] transaction;
        begin
            @(negedge clk); load_valid=1; load_buffer=buf_id; load_tag=transaction; load_words=count;
            #1; check(load_ready,"empty buffer accepts allocation"); step;
            @(negedge clk); load_valid=0;
        end
    endtask
    task send_status;
        input integer buf_id, count;
        input [31:0] transaction;
        input success;
        begin
            @(negedge clk); status_valid=1; status_buffer=buf_id; status_tag=transaction;
            status_ok=success; status_bytes=count*4;
            #1; check(status_ready,"status accepted"); step;
            @(negedge clk); status_valid=0;
        end
    endtask
    task stream_tile;
        input integer buf_id, count, concurrent_read;
        input [31:0] transaction;
        integer word_id, k, base, nvalid;
        begin
            for(word_id=0; word_id<count; word_id=word_id+4) begin
                // Deterministic source bubbles exercise data/status ordering.
                rng=rng ^ (rng<<13); rng=rng ^ (rng>>17); rng=rng ^ (rng<<5);
                if (rng[1:0]==0) begin @(negedge clk); dma_valid=0; read_valid=0; step; end
                @(negedge clk); dma_valid=1; dma_buffer=buf_id; dma_tag=transaction; dma_last=(word_id+4==count);
                for(k=0;k<4;k=k+1) dma_data[k*32 +: 32]=pattern(buf_id,word_id+k);
                if(concurrent_read>=0) begin
                    base=(word_id*3)%512; nvalid=(512-base<LANES)?512-base:LANES;
                    read_valid=1; read_buffer=concurrent_read; read_base=base; read_mask=({LANES{1'b1}} >> (LANES-nvalid));
                end
                #1; check(dma_ready,"filling buffer accepts beat");
                if(concurrent_read>=0) check(read_ready,"compute remains ready during DMA");
                step;
                if(concurrent_read>=0) begin
                    check(response_valid && response_mask==read_mask,"concurrent read response");
                    for(k=0;k<nvalid;k=k+1) check(response_data[k*32 +: 32]===pattern(concurrent_read,base+k),"overlap preserves FP32 payload");
                    overlap_cycles=overlap_cycles+1;
                end
                check(!fault,"valid DMA/read transaction");
            end
            @(negedge clk); dma_valid=0; read_valid=0;
        end
    endtask
    task acquire;
        input integer buf_id;
        input [31:0] transaction;
        begin
            repeat(2) step;
            @(negedge clk); acquire_valid=1; acquire_buffer=buf_id; acquire_tag=transaction;
            #1; check(acquire_ready,"completed tile available"); step;
            @(negedge clk); acquire_valid=0;
        end
    endtask
    task verify_tile;
        input integer buf_id, count;
        integer base, nvalid, k;
        begin
            for(base=0;base<count;base=base+LANES) begin
                nvalid=(count-base<LANES)?count-base:LANES;
                @(negedge clk); read_valid=1; read_buffer=buf_id; read_base=base;
                read_mask=({LANES{1'b1}} >> (LANES-nvalid));
                #1; check(read_ready,"owned buffer readable"); step;
                check(response_valid && response_mask==read_mask,"one-cycle response");
                for(k=0;k<nvalid;k=k+1) check(response_data[k*32 +: 32]===pattern(buf_id,base+k),"all tile words bit exact");
                reads=reads+nvalid;
            end
            @(negedge clk); read_valid=0;
        end
    endtask
    task expect_fault;
        input integer buf_id, code;
        begin
            check(fault && fault_buffers[buf_id] && fault_codes[buf_id*4 +: 4]==code,"expected tagged fault");
            check(!load_ready && !dma_ready && !acquire_ready && !read_ready && !response_valid,"fault stops all buffers");
            @(negedge clk); idle; repeat(3) step; check(fault,"fault is sticky");
        end
    endtask
    integer b;
    initial begin
        if (!$value$plusargs("vcd=%s",wave_path)) wave_path="tiles.vcd";
        if (!$value$plusargs("seed=%d",seed)) seed=2805;
        rng=seed;
        $dumpfile(wave_path); $dumpvars(0,tb_weight_tiles);
        restart;
        // READY requires both completion channels; test data-first and status-first.
        allocate(0,512,32'h1000); stream_tile(0,512,-1,32'h1000);
        check(states[1:0]==1,"data without status remains FILLING");
        send_status(0,512,32'h1000,1); acquire(0,32'h1000);
        for(b=1;b<4;b=b+1) begin
            allocate(b,512,32'h1000+b);
            send_status(b,512,32'h1000+b,1);
            check(states[b*2 +: 2]==1,"status without data remains FILLING");
            stream_tile(b,512,0,32'h1000+b); acquire(b,32'h1000+b);
        end
        // All four contexts survive interleaved engine selection.
        for(b=3;b>=0;b=b-1) verify_tile(b,512);
        verify_windows(0,512);
        @(negedge clk); load_valid=1; load_buffer=0; load_tag=99; load_words=512;
        dma_valid=1; dma_buffer=0; dma_tag=99; dma_data=0;
        repeat(3) begin #1; check(!load_ready && !dma_ready,"occupied buffer backpressures overwrite"); step; @(negedge clk); end
        idle; verify_tile(0,512);
        @(negedge clk); release_valid=1; release_buffer=1; release_tag=32'h1001; step;
        @(negedge clk); release_valid=0;
        allocate(1,8,32'h2001); stream_tile(1,8,0,32'h2001); send_status(1,8,32'h2001,1); acquire(1,32'h2001); verify_tile(1,8);
        verify_windows(1,8);
        // Base 511 plus lane 1 crosses the physical 512-word boundary.
        @(negedge clk); read_valid=1; read_buffer=0; read_base=511; read_mask=2; step; expect_fault(0,7);
        // Reset with a transfer in flight; a new epoch rejects its late data.
        restart; allocate(2,8,32'h2002); restart; allocate(2,8,32'h3002);
        @(negedge clk); dma_valid=1; dma_buffer=2; dma_tag=32'h2002; dma_last=0; step; expect_fault(2,2);
        restart; allocate(0,7,1); expect_fault(0,1);
        restart; allocate(0,8,1);
        @(negedge clk); dma_valid=1; dma_buffer=0; dma_tag=1; dma_last=1; step; expect_fault(0,3);
        restart; allocate(3,8,3); send_status(3,8,3,0); expect_fault(3,4);
        restart; allocate(3,8,3); send_status(3,4,3,1); expect_fault(3,4);
        restart; allocate(3,8,3); send_status(3,8,2,1); expect_fault(3,4);
        restart; allocate(3,8,3); send_status(3,8,3,1); send_status(3,8,3,1); expect_fault(3,4);
        restart; allocate(0,8,1); stream_tile(0,8,-1,1); send_status(0,8,1,1); acquire(0,2); expect_fault(0,5);
        restart; allocate(0,8,1); stream_tile(0,8,-1,1); send_status(0,8,1,1); acquire(0,1);
        @(negedge clk); release_valid=1; release_buffer=0; release_tag=2; step; expect_fault(0,6);
        restart; allocate(0,8,1); stream_tile(0,8,-1,1); send_status(0,8,1,1); acquire(0,1);
        @(negedge clk); read_valid=1; read_buffer=0; read_base=7; read_mask=3; step; expect_fault(0,7);
        $display("RESULT {\"test\":\"weight_tiles\",\"status\":\"passed\",\"lanes\":%0d,\"seed\":%0d,\"assertions\":%0d,\"overlap_cycles\":%0d,\"verified_words\":%0d,\"sliding_window_reads\":%0d}",LANES,seed,checks,overlap_cycles,reads,window_reads);
        $finish;
    end
    initial begin #1000000; $fatal(1,"watchdog"); end
endmodule
`default_nettype wire
