`default_nettype none
// Byte transport for one serial actor job. At 1 Mbaud each byte spans 1000
// core clocks; real core transitions/credit stalls are shorter than that gap.
// A held word/vector at the next byte is an explicit fault, never an overwrite.
// CRC32 authenticates transport bits; host separately verifies checkpoint SHA.
module vroom_board_bridge #(parameter integer LANES=12) (
    input wire clk, reset,
    input wire rx_valid, rx_error, input wire [7:0] rx_data,
    output wire tx_valid, input wire tx_ready, output wire [7:0] tx_data,
    output wire job_valid, output reg job_actor, output reg [31:0] job_id,
    input wire job_ready,
    output wire input_valid, output reg [31:0] input_data,
    output wire input_last, input wire input_ready,
    output wire bias_valid, output wire [LANES*32-1:0] bias_data,
    output wire [31:0] bias_job, output wire [1:0] bias_layer,
    output wire [5:0] bias_group, output wire bias_last, input wire bias_ready,
    output wire weight_valid, output wire [LANES*32-1:0] weight_data,
    output wire [31:0] weight_job, output wire [1:0] weight_layer,
    output wire [7:0] weight_column, output wire [5:0] weight_group,
    output wire weight_last, input wire weight_ready,
    input wire output_valid, output wire output_ready,
    input wire [63:0] output_data, input wire output_actor,
    input wire [31:0] output_job, input wire [4:0] arithmetic_flags,
    input wire [3:0] fault_code
);
    localparam integer GROUPS=(256+LANES-1)/LANES;
    localparam [3:0] SCAN=0,HEADER=1,JOB=2,INPUTS=3,BIASES=4,WEIGHTS=5,
                     CRC=6,WAIT_OUTPUT=7,SEND=8,FINISH=9,HALT=10;
    reg [3:0] state;
    reg [1:0] magic_index,byte_index;
    reg [2:0] header_index;
    reg [4:0] lane_index;
    reg [7:0] input_index,column_index;
    reg [5:0] group_index;
    reg [1:0] current_layer;
    reg [23:0] word_q;
    reg [31:0] received_crc;
    reg [LANES*32-1:0] vector_q;
    reg scalar_pending,vector_pending;
    reg [7:0] bridge_fault;
    wire [8:0] input_columns=job_actor ? 9'd19 : 9'd25;
    wire [8:0] columns=current_layer==0 ? input_columns : 9'd256;
    wire [5:0] groups=current_layer==2 ? 6'd1 : 6'(GROUPS);
    wire last_group=group_index+6'd1==groups;
    wire last_column={1'b0,column_index}+9'd1==columns;
    wire payload_state=state==INPUTS || state==BIASES || state==WEIGHTS;
    wire crc_byte=rx_valid && (state==HEADER || payload_state);
    reg [31:0] rx_crc,tx_crc;
    reg [3:0] rx_crc_left,tx_crc_left;
    reg [223:0] reply;
    reg [5:0] tx_index;
    reg running;
    reg [31:0] elapsed_cycles;
    wire tx_fire=tx_valid && tx_ready;
    wire [31:0] final_tx_crc=~tx_crc;

    assign job_valid=state==JOB && bridge_fault==0 && !reset;
    assign input_valid=state==INPUTS && scalar_pending && !reset;
    assign input_last={1'b0,input_index}+9'd1==input_columns;
    assign bias_valid=state==BIASES && vector_pending && !reset;
    assign bias_data=vector_q;
    assign bias_job=job_id;
    assign bias_layer=current_layer;
    assign bias_group=group_index;
    assign bias_last=last_group;
    assign weight_valid=state==WEIGHTS && vector_pending && !reset;
    assign weight_data=vector_q;
    assign weight_job=job_id;
    assign weight_layer=current_layer;
    assign weight_column=column_index;
    assign weight_group=group_index;
    assign weight_last=last_group && last_column;
    assign tx_valid=state==SEND && !reset;
    assign tx_data=tx_index<28 ? reply[tx_index*8+:8] : final_tx_crc[(tx_index-6'd28)*8+:8];
    assign output_ready=state==FINISH && tx_ready && bridge_fault==0 && fault_code==0 && !reset;

    // Bit-serial CRC update avoids an eight-XOR-stage path in the core clock.
    always @(posedge clk)begin
        if(crc_byte)begin rx_crc<=rx_crc^{24'b0,rx_data};rx_crc_left<=8;end
        else if(rx_crc_left!=0)begin
            rx_crc<=(rx_crc>>1)^(rx_crc[0] ? 32'hedb88320 : 32'b0);
            rx_crc_left<=rx_crc_left-1'b1;
        end
        if(tx_fire && tx_index<28)begin tx_crc<=tx_crc^{24'b0,tx_data};tx_crc_left<=8;end
        else if(tx_crc_left!=0)begin
            tx_crc<=(tx_crc>>1)^(tx_crc[0] ? 32'hedb88320 : 32'b0);
            tx_crc_left<=tx_crc_left-1'b1;
        end
        if(state==SCAN)begin rx_crc<=32'hffffffff;rx_crc_left<=0;end
        if(state==WAIT_OUTPUT)begin tx_crc<=32'hffffffff;tx_crc_left<=0;end
        if(reset)begin rx_crc<=32'hffffffff;tx_crc<=32'hffffffff;rx_crc_left<=0;tx_crc_left<=0;end
    end

    always @(posedge clk)begin
        if(running && !output_valid)elapsed_cycles<=elapsed_cycles+1'b1;
        if(output_valid)running<=0;
        case(state)
            SCAN: if(rx_valid)begin
                case(magic_index)
                    0:magic_index<=rx_data==8'h56 ? 1 : 0;
                    1:magic_index<=rx_data==8'h52 ? 2 : 0;
                    2:magic_index<=rx_data==8'h4d ? 3 : 0;
                    3:begin
                        magic_index<=0;
                        if(rx_data==8'h31)begin state<=HEADER;header_index<=0;end
                    end
                endcase
            end
            HEADER: if(rx_valid)begin
                if(header_index==0)begin
                    job_actor<=rx_data[0];
                    if(rx_data>1)begin bridge_fault<=8'h83;state<=WAIT_OUTPUT;end
                end else job_id[(header_index-3'd1)*8+:8]<=rx_data;
                if(header_index==4)state<=JOB;
                else header_index<=header_index+1'b1;
            end
            JOB: if(job_ready)begin
                state<=INPUTS;input_index<=0;byte_index<=0;current_layer<=0;
                group_index<=0;column_index<=0;lane_index<=0;
                scalar_pending<=0;vector_pending<=0;running<=1;elapsed_cycles<=0;
            end
            INPUTS: if(input_valid && input_ready)begin
                scalar_pending<=0;
                if(input_last)begin state<=BIASES;group_index<=0;end
                else input_index<=input_index+1'b1;
            end
            BIASES: if(bias_valid && bias_ready)begin
                vector_pending<=0;
                if(last_group)begin state<=WEIGHTS;group_index<=0;column_index<=0;end
                else group_index<=group_index+1'b1;
            end
            WEIGHTS: if(weight_valid && weight_ready)begin
                vector_pending<=0;
                if(last_group)begin
                    group_index<=0;
                    if(last_column)begin
                        if(current_layer==2)begin state<=CRC;byte_index<=0;end
                        else begin state<=BIASES;current_layer<=current_layer+1'b1;end
                    end else column_index<=column_index+1'b1;
                end else group_index<=group_index+1'b1;
            end
            CRC: if(rx_valid)begin
                received_crc[byte_index*8+:8]<=rx_data;
                if(byte_index==3)begin
                    state<=WAIT_OUTPUT;
                    if({rx_data,received_crc[23:0]}!=~rx_crc || rx_crc_left!=0)bridge_fault<=8'h82;
                end else byte_index<=byte_index+1'b1;
            end
            WAIT_OUTPUT: if(output_valid || bridge_fault!=0 || fault_code!=0)begin
                // Reply VRO1/version/actor/fault/flags/job/two FP32/cycles/CRC.
                reply<={received_crc,elapsed_cycles,(output_valid ? output_data : 64'b0),
                        job_id,{3'b0,arithmetic_flags},
                        (bridge_fault!=0 ? bridge_fault : {4'b0,fault_code}),
                        {7'b0,job_actor},8'd1,32'h314f5256};
                if(output_valid && (output_job!=job_id || output_actor!=job_actor))begin
                    reply[55:48]<=8'h85;bridge_fault<=8'h85;
                end
                tx_index<=0;state<=SEND;
            end
            SEND: if(tx_fire)begin
                if(tx_index==31)state<=FINISH;
                else tx_index<=tx_index+1'b1;
            end
            FINISH: if(tx_ready)begin
                if(bridge_fault!=0 || fault_code!=0)state<=HALT;
                else begin state<=SCAN;magic_index<=0;end
            end
            default:;
        endcase
        if(rx_valid && payload_state)begin
            if(scalar_pending || vector_pending)begin bridge_fault<=8'h81;state<=WAIT_OUTPUT;end
            else begin
                word_q<={rx_data,word_q[23:8]};
                if(byte_index==3)begin
                    byte_index<=0;
                    if(state==INPUTS)begin input_data<={rx_data,word_q[23:0]};scalar_pending<=1;end
                    else begin
                        vector_q[lane_index*32+:32]<={rx_data,word_q[23:0]};
                        if(lane_index==5'(LANES-1))begin lane_index<=0;vector_pending<=1;end
                        else lane_index<=lane_index+1'b1;
                    end
                end else byte_index<=byte_index+1'b1;
            end
        end
        if(rx_error || (crc_byte && rx_crc_left!=0))begin bridge_fault<=8'h80;state<=WAIT_OUTPUT;end
        if(rx_valid && state==JOB)begin bridge_fault<=8'h81;state<=WAIT_OUTPUT;end
        if(reset)begin
            state<=SCAN;magic_index<=0;byte_index<=0;header_index<=0;
            job_actor<=0;job_id<=0;lane_index<=0;input_index<=0;column_index<=0;
            group_index<=0;current_layer<=0;scalar_pending<=0;vector_pending<=0;
            bridge_fault<=0;running<=0;elapsed_cycles<=0;received_crc<=0;
            input_data<=0;word_q<=0;vector_q<=0;reply<=0;tx_index<=0;
        end
    end
endmodule
`default_nettype wire
