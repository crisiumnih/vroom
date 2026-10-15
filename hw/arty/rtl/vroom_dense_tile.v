`default_nettype none
// Dense microtile executor. Arithmetic is an external, in-order FP32 SIMD port:
// result[lane] = RN32(RN32(weight[lane] * input) + accumulator[lane]).
// No integer arithmetic is substituted for FP32. See dense-engine.md for protocol.
module vroom_dense_tile #(
    parameter integer LANES = 12
) (
    input wire clk,
    input wire reset,
    input wire context_valid,
    output wire context_accept,
    input wire context_actor,
    input wire [31:0] context_job,
    input wire [1:0] context_layer,
    input wire [8:0] context_inputs,
    input wire [8:0] context_outputs,
    input wire input_valid,
    output wire input_ready,
    input wire [31:0] input_data,
    input wire bias_valid,
    output wire bias_ready,
    input wire [31:0] bias_data,
    output reg [1:0] context_ready,
    output reg [1:0] context_complete,
    input wire command_valid,
    output wire command_ready,
    input wire command_actor,
    input wire [31:0] command_job,
    input wire [1:0] command_layer,
    input wire [1:0] command_buffer,
    input wire [31:0] command_tag,
    input wire [7:0] command_row,
    input wire [7:0] command_rows,
    input wire [7:0] command_column,
    input wire [2:0] command_columns,
    output wire done_valid,
    input wire done_ready,
    output reg active_actor,
    output reg [31:0] active_job,
    output reg [1:0] active_layer,
    output wire tile_acquire_valid,
    input wire tile_acquire_ready,
    output reg [1:0] tile_buffer,
    output reg [31:0] tile_tag,
    output wire tile_release_valid,
    input wire tile_release_ready,
    output wire tile_read_valid,
    input wire tile_read_ready,
    output wire [8:0] tile_read_base,
    output wire [LANES-1:0] tile_read_mask,
    input wire tile_response_valid,
    input wire [LANES-1:0] tile_response_mask,
    input wire [LANES*32-1:0] tile_response_data,
    input wire tile_fault,
    output wire mac_valid,
    input wire mac_ready,
    output wire [31:0] mac_tag,
    output wire [LANES-1:0] mac_mask,
    output wire [31:0] mac_input,
    output wire [LANES*32-1:0] mac_weight,
    output wire [LANES*32-1:0] mac_accumulator,
    input wire mac_response_valid,
    output wire mac_response_ready,
    input wire [31:0] mac_response_tag,
    input wire [LANES*32-1:0] mac_response_data,
    input wire result_valid,
    output wire result_ready,
    input wire result_actor,
    input wire [7:0] result_row,
    input wire [LANES-1:0] result_mask,
    output wire result_response_valid,
    output reg [LANES-1:0] result_response_mask,
    output reg [LANES*32-1:0] result_response_data,
    output reg [31:0] result_response_job,
    output reg [1:0] result_response_layer,
    output reg [3:0] state,
    output reg fault,
    output reg [3:0] fault_code
);
    localparam [3:0] IDLE=0, LOAD=1, CHECK=2, ACQUIRE=3, ISSUE=4, DRAIN=5, RELEASE=6, DONE=7;
    localparam [8:0] LANE_STEP = LANES[8:0];
    reg [1:0] context_live;
    reg [31:0] jobs [0:1];
    reg [1:0] layers [0:1];
    reg [8:0] inputs [0:1], outputs [0:1], completed_rows [0:1];
    reg load_actor;
    reg [8:0] input_count, bias_count;
    // A single input broadcasts to every lane. 2 x 256 binary32 input words.
    (* ram_style = "block" *) reg [31:0] input_memory [0:511];
    reg [7:0] first_row, row_count, first_column;
    reg [2:0] column_count, issue_column;
    reg [8:0] issue_row, check_row;
    reg [127:0] busy_rows;
    wire live = !reset && !fault && !tile_fault;
    wire context_fire = context_valid && context_accept;
    wire input_fire = input_valid && input_ready;
    wire bias_fire = bias_valid && bias_ready;
    wire command_fire = command_valid && command_ready;
    wire result_fire = result_valid && result_ready;
    wire [8:0] current_column = {1'b0,first_column} + {6'b0,issue_column};
    wire [8:0] command_row_end = {1'b0,command_row} + {1'b0,command_rows};
    wire [8:0] command_column_end = {1'b0,command_column} + {6'b0,command_columns};
    wire [10:0] command_words = {3'b0,command_rows} * {8'b0,command_columns};

    // Four-entry operand FIFO absorbs a one-cycle, non-stallable tile response.
    reg [LANES*32-1:0] queue_weight [0:3], queue_accumulator [0:3];
    reg [31:0] queue_input [0:3], queue_tag [0:3];
    reg [LANES-1:0] queue_mask [0:3];
    reg [1:0] queue_write, queue_read;
    reg [2:0] queue_count;
    reg request_pending;
    reg [31:0] request_tag, request_input;
    reg [LANES-1:0] request_mask;
    reg [LANES*32-1:0] request_accumulator;

    // Sixteen outstanding vector reservations. Retire in issue order.
    reg [31:0] ticket;
    reg [31:0] slot_tag [0:15];
    reg [8:0] slot_row [0:15], slot_column [0:15];
    reg [LANES-1:0] slot_mask [0:15];
    reg [3:0] reserve_pointer, retire_pointer;
    reg [4:0] reserved_count, accepted_count;
    wire read_fire = tile_read_valid && tile_read_ready;
    wire mac_fire = mac_valid && mac_ready;
    wire retire_fire = mac_response_valid && mac_response_ready;
    wire retire_good = retire_fire && accepted_count != 0 &&
                       mac_response_tag == slot_tag[retire_pointer];
    wire push_operand = tile_response_valid && request_pending &&
                        tile_response_mask == request_mask && live;

    // Banked accumulator and progress state. A row's next input column is
    // advanced only on successful retirement, never on issue or DMA completion.
    wire selected_actor = (state == CHECK || state == ISSUE) ? active_actor : result_actor;
    wire [8:0] selected_row = state == CHECK ? check_row :
                                state == ISSUE ? {1'b0,first_row}+issue_row : {1'b0,result_row};
    wire [655:0] bank_payload;
    wire [LANES*41-1:0] selected_payload;
    wire [LANES*32-1:0] selected_data;
    wire [LANES-1:0] lane_progress_bad, lane_busy, lane_result_bounds_bad;
    wire progress_bad = |lane_progress_bad;
    wire rows_busy = |lane_busy;
    wire result_bounds_bad = |lane_result_bounds_bad;
    wire [9:0] issue_lane_row [0:LANES-1];
    wire [9:0] retire_lane_row [0:LANES-1];
    wire [7:0] retirement_row = first_row + slot_row[retire_pointer][7:0];
    wire [15:0] retirement_mask = {{(16-LANES){1'b0}}, slot_mask[retire_pointer]};
    wire [511:0] retirement_data = {{((16-LANES)*32){1'b0}}, mac_response_data};
    wire [511:0] write_bank_data;
    wire [15:0] write_bank_mask;
    reg [8:0] retire_rows;
    integer count_lane;

    assign context_accept = state == IDLE && live &&
                            (!context_live[context_actor] || context_complete[context_actor]);
    assign input_ready = state == LOAD && live && input_count < inputs[load_actor];
    assign bias_ready = state == LOAD && live && bias_count < outputs[load_actor];
    assign command_ready = state == IDLE && live && !context_fire;
    assign result_ready = state == IDLE && live && !context_fire && !command_valid &&
                          context_ready[result_actor] && context_complete[result_actor];
    assign done_valid = state == DONE && live;
    assign tile_acquire_valid = state == ACQUIRE && live;
    assign tile_release_valid = state == RELEASE && live;
    assign tile_read_valid = state == ISSUE && live && !rows_busy && reserved_count < 16 &&
                             ({1'b0,queue_count}+{3'b0,request_pending}) < 4;
    assign tile_read_base = ({6'b0,issue_column} * {1'b0,row_count}) + issue_row;
    assign mac_valid = queue_count != 0 && live;
    assign mac_tag = queue_tag[queue_read];
    assign mac_mask = queue_mask[queue_read];
    assign mac_input = queue_input[queue_read];
    assign mac_weight = queue_weight[queue_read];
    assign mac_accumulator = queue_accumulator[queue_read];
    // Operator outputs must be registered (at least one cycle after acceptance).
    assign mac_response_ready = live;
    reg result_response_v;
    assign result_response_valid = result_response_v && live;

    always @* begin
        retire_rows=0;
        if (!bias_fire && retire_good)
            for (count_lane=0; count_lane<LANES; count_lane=count_lane+1)
                retire_rows=retire_rows+{8'b0,slot_mask[retire_pointer][count_lane]};
    end
    vroom_bank_rotate #(.WIDTH(41), .WORDS(LANES)) read_rotation (
        .data_in(bank_payload), .offset(selected_row[3:0]), .data_out(selected_payload));
    vroom_bank_rotate #(.REVERSE(1)) write_rotation (
        .data_in(retirement_data), .offset(retirement_row[3:0]), .data_out(write_bank_data));
    vroom_bank_rotate #(.WIDTH(1), .REVERSE(1)) mask_rotation (
        .data_in(retirement_mask), .offset(retirement_row[3:0]), .data_out(write_bank_mask));
    genvar bank, lane;
    generate for (bank=0; bank<16; bank=bank+1) begin : contexts
        localparam [3:0] BANK = bank;
        // Each vector touches a bank at most once (LANES <= 16). Derive
        // bank-local row groups directly; no variable writes into packed
        // address buses and no 32-bit signed bank-offset arithmetic.
        wire read_wrap, write_wrap;
        if (bank == 15) begin : last_bank
            assign read_wrap = 1'b0;
            assign write_wrap = 1'b0;
        end else begin : wrapping_bank
            assign read_wrap = BANK < selected_row[3:0];
            assign write_wrap = BANK < retirement_row[3:0];
        end
        wire [3:0] read_group = selected_row[7:4] + {3'b0, read_wrap};
        wire [4:0] read_address = {selected_actor, read_group};
        wire [3:0] write_group = retirement_row[7:4] + {3'b0, write_wrap};
        wire write_enable = bias_fire ? bias_count[3:0] == BANK :
                                        retire_good && write_bank_mask[bank];
        wire [4:0] write_address = bias_fire ? {load_actor,bias_count[7:4]} :
                                                              {active_actor,write_group};
        wire [31:0] write_data = bias_fire ? bias_data : write_bank_data[bank*32 +: 32];
        wire [8:0] write_column = bias_fire ? 9'b0 : slot_column[retire_pointer]+9'd1;
        (* ram_style = "distributed" *) reg [31:0] accumulator [0:31];
        (* ram_style = "distributed" *) reg [8:0] next_column [0:31];
        assign bank_payload[bank*41 +: 41]={next_column[read_address],accumulator[read_address]};
        always @(posedge clk) begin
            if (write_enable) begin
                accumulator[write_address]<=write_data;
                next_column[write_address]<=write_column;
            end
        end
    end
    for (lane=0; lane<LANES; lane=lane+1) begin : read_lanes
        localparam [9:0] OFFSET = lane;
        wire [9:0] row_index = {1'b0,selected_row} + OFFSET;
        assign issue_lane_row[lane] = {1'b0,issue_row} + OFFSET;
        assign retire_lane_row[lane] = {1'b0,slot_row[retire_pointer]} + OFFSET;
        assign selected_data[lane*32 +: 32] = selected_payload[lane*41 +: 32];
        assign tile_read_mask[lane] = issue_lane_row[lane] < {2'b0,row_count};
        // Command validation limits the tile to 128 rows; masked lanes cannot
        // address the scoreboard. Preserve full-width sums for bounds checks.
        assign lane_busy[lane] = tile_read_mask[lane] && busy_rows[issue_lane_row[lane][6:0]];
        assign lane_progress_bad[lane] = row_index < ({2'b0,first_row}+{2'b0,row_count}) &&
                                          selected_payload[lane*41+32 +: 9] != {1'b0,first_column};
        assign lane_result_bounds_bad[lane] = result_mask[lane] && row_index >= {1'b0,outputs[result_actor]};
    end endgenerate
    always @(posedge clk) begin
        if (input_fire) input_memory[{load_actor,input_count[7:0]}]<=input_data;
        if (read_fire) request_input<=input_memory[{active_actor,current_column[7:0]}];
    end

    integer seq_lane, actor;
    always @(posedge clk) begin
        result_response_v<=1'b0;
        if (reset) begin
            state<=IDLE;
            fault<=0;
            fault_code<=0;
            context_live<=0;
            context_ready<=0;
            context_complete<=0;
            load_actor<=0;
            input_count<=0;
            bias_count<=0;
            active_actor<=0;
            active_job<=0;
            active_layer<=0;
            first_row<=0;
            row_count<=0;
            first_column<=0;
            column_count<=0;
            issue_row<=0;
            issue_column<=0;
            check_row<=0;
            tile_buffer<=0;
            tile_tag<=0;
            busy_rows<=0;
            request_pending<=0;
            request_tag<=0;
            request_mask<=0;
            request_accumulator<=0;
            queue_write<=0;
            queue_read<=0;
            queue_count<=0;
            ticket<=0;
            reserve_pointer<=0;
            retire_pointer<=0;
            reserved_count<=0;
            accepted_count<=0;
            result_response_mask<=0;
            result_response_job<=0;
            result_response_layer<=0;
            for (actor=0; actor<2; actor=actor+1) begin
                jobs[actor]<=0;
                layers[actor]<=0;
                inputs[actor]<=0;
                outputs[actor]<=0;
                completed_rows[actor]<=0;
            end
        end else if (!fault) begin
            request_pending<=read_fire;
            if (context_fire) begin
                if (context_inputs==0 || context_inputs>256 || context_outputs==0 || context_outputs>256) begin
                    fault<=1; fault_code<=1;
                end else begin
                    load_actor<=context_actor;
                    context_live[context_actor]<=1;
                    context_ready[context_actor]<=0;
                    context_complete[context_actor]<=0;
                    jobs[context_actor]<=context_job;
                    layers[context_actor]<=context_layer;
                    inputs[context_actor]<=context_inputs;
                    outputs[context_actor]<=context_outputs;
                    completed_rows[context_actor]<=0;
                    input_count<=0; bias_count<=0;
                    state<=LOAD;
                end
            end
            if (input_fire) input_count<=input_count+1'b1;
            if (bias_fire) bias_count<=bias_count+1'b1;
            if (state==LOAD && input_count==inputs[load_actor] && bias_count==outputs[load_actor]) begin
                context_ready[load_actor]<=1;
                state<=IDLE;
            end
            if (command_fire) begin
                if (!context_ready[command_actor] || context_complete[command_actor] ||
                    jobs[command_actor]!=command_job || layers[command_actor]!=command_layer) begin
                    fault<=1; fault_code<=2;
                end else if (command_rows==0 || command_rows>128 || command_columns==0 || command_columns>4 ||
                             command_row_end>outputs[command_actor] || command_column_end>inputs[command_actor] ||
                             command_words>512 || command_words[1:0]!=0 || command_buffer[1]!=command_actor) begin
                    fault<=1; fault_code<=1;
                end else begin
                    active_actor<=command_actor; active_job<=command_job; active_layer<=command_layer;
                    first_row<=command_row; row_count<=command_rows;
                    first_column<=command_column; column_count<=command_columns;
                    tile_buffer<=command_buffer; tile_tag<=command_tag;
                    issue_row<=0; issue_column<=0; check_row<={1'b0,command_row};
                    busy_rows<=0;
                    state<=CHECK;
                end
            end
            if (state==CHECK) begin
                if (progress_bad) begin fault<=1; fault_code<=3; end
                else if (check_row+LANE_STEP >= {1'b0,first_row}+{1'b0,row_count}) state<=ACQUIRE;
                else check_row<=check_row+LANE_STEP;
            end
            if (tile_acquire_valid && tile_acquire_ready) state<=ISSUE;
            if (read_fire) begin
                request_tag<=ticket;
                request_mask<=tile_read_mask;
                request_accumulator<=selected_data;
                slot_tag[reserve_pointer]<=ticket;
                slot_row[reserve_pointer]<=issue_row;
                slot_column[reserve_pointer]<=current_column;
                slot_mask[reserve_pointer]<=tile_read_mask;
                ticket<=ticket+1'b1;
                reserve_pointer<=reserve_pointer+1'b1;
                for (seq_lane=0; seq_lane<LANES; seq_lane=seq_lane+1)
                    if (tile_read_mask[seq_lane]) busy_rows[issue_lane_row[seq_lane][6:0]]<=1;
                if (issue_row+LANE_STEP >= {1'b0,row_count}) begin
                    issue_row<=0;
                    if (issue_column+3'd1 == column_count) state<=DRAIN;
                    else issue_column<=issue_column+1'b1;
                end else issue_row<=issue_row+LANE_STEP;
            end
            if (push_operand) begin
                queue_weight[queue_write]<=tile_response_data;
                queue_accumulator[queue_write]<=request_accumulator;
                queue_input[queue_write]<=request_input;
                queue_mask[queue_write]<=request_mask;
                queue_tag[queue_write]<=request_tag;
                queue_write<=queue_write+1'b1;
            end
            if (mac_fire) queue_read<=queue_read+1'b1;
            case ({push_operand,mac_fire})
                2'b10: queue_count<=queue_count+1'b1;
                2'b01: queue_count<=queue_count-1'b1;
                default: ;
            endcase
            case ({read_fire,retire_good})
                2'b10: reserved_count<=reserved_count+1'b1;
                2'b01: reserved_count<=reserved_count-1'b1;
                default: ;
            endcase
            case ({mac_fire,retire_good})
                2'b10: accepted_count<=accepted_count+1'b1;
                2'b01: accepted_count<=accepted_count-1'b1;
                default: ;
            endcase
            if (retire_good) begin
                retire_pointer<=retire_pointer+1'b1;
                for (seq_lane=0; seq_lane<LANES; seq_lane=seq_lane+1)
                    if (slot_mask[retire_pointer][seq_lane]) busy_rows[retire_lane_row[seq_lane][6:0]]<=0;
                if (slot_column[retire_pointer]+9'd1 == inputs[active_actor]) begin
                    completed_rows[active_actor]<=completed_rows[active_actor]+retire_rows;
                    if (completed_rows[active_actor]+retire_rows == outputs[active_actor]) context_complete[active_actor]<=1;
                end
            end
            if (state==DRAIN && reserved_count==0) state<=RELEASE;
            if (tile_release_valid && tile_release_ready) state<=DONE;
            if (done_valid && done_ready) state<=IDLE;
            if (result_fire) begin
                if (result_bounds_bad) begin fault<=1; fault_code<=7; end
                else begin
                    result_response_v<=1;
                    result_response_mask<=result_mask;
                    result_response_job<=jobs[result_actor];
                    result_response_layer<=layers[result_actor];
                    for (seq_lane=0; seq_lane<LANES; seq_lane=seq_lane+1)
                        result_response_data[seq_lane*32 +: 32]<=result_mask[seq_lane] ? selected_data[seq_lane*32 +: 32] : 32'b0;
                end
            end
            // Infrastructure/protocol failures invalidate the whole episode.
            if (tile_response_valid != request_pending ||
                (tile_response_valid && (tile_response_mask!=request_mask || queue_count>=4))) begin
                fault<=1; fault_code<=4;
            end
            if (retire_fire && !retire_good) begin fault<=1; fault_code<=5; end
            if (tile_fault) begin fault<=1; fault_code<=6; end
        end
    end
`ifndef SYNTHESIS
    initial if (LANES<1 || LANES>16) $fatal(1,"LANES must be 1..16");
`endif
endmodule
`default_nettype wire
