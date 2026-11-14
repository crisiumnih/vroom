`timescale 1ns/1ps
`default_nettype none
// Transport isolation: check every scalar/vector word and all generated metadata
// against complete real checkpoint request packets, with a bounded stalled sink.
// Arithmetic is checked independently by the existing full-layer RTL test.
module tb_bridge;
    localparam integer LANES=12;
    reg clk=0;always #5 clk=~clk;
    reg reset=1,rx_valid=0,rx_error=0;
    reg [7:0] rx_data=0;
    wire tx_valid;wire [7:0] tx_data;
    integer cooldown=0;
    wire tx_ready=cooldown==0;
    wire job_valid,job_actor,input_valid,input_last,bias_valid,bias_last,weight_valid,weight_last;
    wire [31:0] job_id,input_data,bias_job,weight_job;
    wire [LANES*32-1:0] bias_data,weight_data;
    wire [1:0] bias_layer,weight_layer;
    wire [5:0] bias_group,weight_group;
    wire [7:0] weight_column;
    reg [7:0] ticks=0;
    wire job_ready=1;
    wire input_ready=ticks[2:0]!=0,bias_ready=ticks[2:0]!=1,weight_ready=ticks[2:0]!=2;
    reg output_valid=0,output_actor=0;
    wire output_ready;
    reg [31:0] output_job=0;
    wire [63:0] output_data=64'hbf0000003e800000;
    wire [4:0] arithmetic_flags=1;
    wire [3:0] fault_code=0;
    vroom_board_bridge #(.LANES(LANES)) dut(.*);
    reg [7:0] packet[0:400000];
    integer file,nbytes,pointer,phase,ni,idx,ly,gr,col,groups,cols,l,j,n,reply_file;
    integer replies=0,acks=0,words=0,expected_job;
    reg expected_actor;
    reg [1023:0] filename;
    function [31:0] packet_word(input integer offset);
        packet_word={packet[offset+3],packet[offset+2],packet[offset+1],packet[offset]};
    endfunction
    task check(input reg ok,input [511:0] message);
        if(ok!==1'b1)$fatal(1,"%0s job=%0d phase=%0d layer=%0d group=%0d col=%0d pointer=%0d",message,j,phase,ly,gr,col,pointer);
    endtask
    task send_byte(input reg [7:0] value);
        begin
            @(negedge clk);rx_data=value;rx_valid=1;
            @(negedge clk);rx_valid=0;
            repeat(15)@(negedge clk);
        end
    endtask
    always @(posedge clk)begin
        ticks<=ticks+1'b1;
        if(cooldown>0)cooldown<=cooldown-1;
        if(tx_valid && tx_ready)begin
            $fwrite(reply_file,"%c",tx_data);replies=replies+1;cooldown<=12;
        end
        if(!reset)begin
            if(job_valid && job_ready)begin
                check(job_actor==expected_actor && job_id==32'(expected_job),"job identity");
                ni=job_actor ? 19 : 25;phase=0;idx=0;ly=0;gr=0;col=0;pointer=9;
                output_job<=job_id;output_actor<=job_actor;
            end
            if(input_valid && input_ready)begin
                check(phase==0 && input_data===packet_word(pointer) && input_last==(idx==ni-1),"prepared input word/last");
                pointer=pointer+4;words=words+1;idx=idx+1;
                if(idx==ni)phase=1;
            end
            if(bias_valid && bias_ready)begin
                groups=ly==2 ? 1 : (256+LANES-1)/LANES;
                check(phase==1 && bias_job==32'(expected_job) && bias_layer==2'(ly) && bias_group==6'(gr) && bias_last==(gr==groups-1),"bias metadata");
                for(l=0;l<LANES;l=l+1)begin
                    check(bias_data[l*32+:32]===packet_word(pointer),"bias bits/tail mask transport");
                    pointer=pointer+4;words=words+1;
                end
                gr=gr+1;if(gr==groups)begin phase=2;gr=0;col=0;end
            end
            if(weight_valid && weight_ready)begin
                groups=ly==2 ? 1 : (256+LANES-1)/LANES;cols=ly==0 ? ni : 256;
                check(phase==2 && weight_job==32'(expected_job) && weight_layer==2'(ly) && weight_group==6'(gr) && weight_column==8'(col) && weight_last==(gr==groups-1 && col==cols-1),"weight metadata");
                for(l=0;l<LANES;l=l+1)begin
                    check(weight_data[l*32+:32]===packet_word(pointer),"column/group/lane weight bits");
                    pointer=pointer+4;words=words+1;
                end
                gr=gr+1;
                if(gr==groups)begin
                    gr=0;col=col+1;
                    if(col==cols)begin
                        col=0;ly=ly+1;phase=1;
                        if(ly==3)begin
                            phase=3;check(pointer==nbytes-4,"exact complete payload length");output_valid<=1;
                        end
                    end
                end
            end
            if(output_ready && output_valid)begin output_valid<=0;acks=acks+1;end
        end else output_valid<=0;
    end
    initial begin
        reply_file=$fopen("replies.bin","wb");
        repeat(3)@(negedge clk);reset=0;
        for(j=0;j<3;j=j+1)begin
            $sformat(filename,"request%0d.bin",j==1 ? 1 : 0);
            file=$fopen(filename,"rb");check(file!=0,"request open");
            nbytes=$fread(packet,file);$fclose(file);
            expected_actor=packet[4][0];expected_job=int'(packet_word(5));
            if(j==2)packet[nbytes-1]=packet[nbytes-1]^8'h01;
            for(n=0;n<nbytes;n=n+1)send_byte(packet[n]);
            wait(replies==32*(j+1));repeat(20)@(negedge clk);
            check(acks==(j==0 ? 1 : 2),"good packet consumed once; bad CRC never committed");
            $display("BRIDGE_FRAME_OK index=%0d bytes=%0d total_words=%0d",j,nbytes,words);
        end
        check(dut.state==10 && output_valid,"CRC failure stays halted with uncommitted output");
        $fclose(reply_file);
        $display("VROOM_BRIDGE_OK packets=3 good=2 crc_rejected=1 words_checked=%0d",words);
        $finish;
    end
    initial begin #250000000;$fatal(1,"transport watchdog");end
endmodule
`default_nettype wire
