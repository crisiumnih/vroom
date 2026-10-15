`default_nettype none
// One complete actor per job: (25 or 19)->256->256->2, separate RN32 multiply
// and add, ReLU after hidden layers, shared CORDIC tanh after the final layer.
// The source supplies column-major output-group weight vectors and group-major
// bias vectors, lane 0 in bits [31:0]. Tail masks are generated internally.
// One job at a time; all streams must flush jointly on reset. See ../README.md.
module vroom_actor_stream #(parameter integer LANES=12) (
    input wire clk, reset,
    input wire job_valid, input wire job_actor, input wire [31:0] job_id,
    output wire job_ready,
    input wire input_valid, input wire [31:0] input_data, input wire input_last,
    output wire input_ready,
    input wire bias_valid, input wire [LANES*32-1:0] bias_data,
    input wire [31:0] bias_job, input wire [1:0] bias_layer,
    input wire [5:0] bias_group, input wire bias_last,
    output wire bias_ready,
    input wire weight_valid, input wire [LANES*32-1:0] weight_data,
    input wire [31:0] weight_job, input wire [1:0] weight_layer,
    input wire [7:0] weight_column, input wire [5:0] weight_group,
    input wire weight_last, output wire weight_ready,
    output wire output_valid, input wire output_ready,
    output reg [63:0] output_data, output reg output_actor,
    output reg [31:0] output_job, output reg [4:0] arithmetic_flags,
    output reg [3:0] fault_code,
    output wire [3:0] debug_state, output reg [1:0] layer
);
    localparam integer GROUPS=(256+LANES-1)/LANES;
    localparam integer GROUP_AW=$clog2(GROUPS);
    localparam integer ACTIVATION_AW=GROUP_AW+1;
    localparam integer LAST_LANE_NUMBER=LANES-1;
    localparam [5:0] HIDDEN_GROUPS=GROUPS[5:0];
    localparam [4:0] LAST_LANE=LAST_LANE_NUMBER[4:0];
    localparam [3:0] IDLE=0,INPUTS=1,LAYER_RESET=2,BIASES=3,FETCH=4,
                     RUN=5,DRAIN=6,TANH_SEND=7,TANH_WAIT=8,OUTPUTS=9,FAILED=10;
    // Preserve the binary debug interface but let synthesis use direct state
    // bits for enables; exporting debug_state otherwise inhibits FSM extraction.
    (* fsm_encoding="one-hot" *) reg [3:0] state;
    reg failed;
    reg source_bank;
    reg [8:0] columns;
    reg [7:0] input_count,stream_column;
    reg [5:0] init_group,stream_group;
    reg [GROUP_AW-1:0] read_group;
    reg [4:0] read_lane;
    reg [31:0] input_hold;
    reg pending;
    reg [LANES*32-1:0] weights_q;
    reg [31:0] operand_q;
    reg [15:0] tag_q;
    reg [LANES-1:0] mask_q;
    wire [LANES-1:0] lane_valid,lane_done;
    wire engine_ready,engine_init_ready,engine_valid;
    wire [15:0] engine_tag;
    wire [LANES-1:0] engine_mask;
    wire [LANES*16-1:0] lane_tag;
    wire [LANES*32-1:0] lane_result,bank_reads;
    wire [LANES*5-1:0] lane_flags;
    wire lane_reset=reset || state==LAYER_RESET || failed;
    wire [5:0] active_groups=layer==2 ? 6'd1 : HIDDEN_GROUPS;
    wire last_input=({1'b0,input_count}+9'd1)==columns;
    wire last_column=({1'b0,stream_column}+9'd1)==columns;
    wire last_group=(stream_group+6'd1)==active_groups;
    wire last_bias=(init_group+6'd1)==active_groups;
    // Reset suppresses external handshakes and resets all ownership state.
    // Internal payload enables need not depend combinationally on reset: a
    // write on that edge is invalidated by the synchronous state reset.
    wire input_fire=input_valid && state==INPUTS;
    wire bias_fire=bias_valid && state==BIASES && engine_init_ready;
    wire issue=pending && engine_ready;
    wire weight_fire=weight_valid && state==RUN && (!pending || issue);
    // Predict the next group for distributed read-address registers. The bank
    // changes only between layers; the bias phase hides its one-clock delay.
    // During RUN the prediction updates on the same edge as read_group, so
    // even the one-group output layer retains one column per eligible clock.
    wire advance_read_group=state==RUN && weight_fire && last_group && !last_column && read_lane==LAST_LANE;
    wire [GROUP_AW-1:0] next_read_group=state==LAYER_RESET ? {GROUP_AW{1'b0}} :
                                      read_group+{{(GROUP_AW-1){1'b0}},advance_read_group};
    assign job_ready=!reset && !failed && state==IDLE;
    assign input_ready=!reset && !failed && state==INPUTS;
    assign bias_ready=!reset && !failed && state==BIASES && engine_init_ready;
    assign weight_ready=!reset && !failed && state==RUN && (!pending || issue);
    assign output_valid=!reset && !failed && state==OUTPUTS;
    assign debug_state=failed ? FAILED : state;

    wire [LANES-1:0] bias_bad,weight_bad,response_bad;
    wire [LANES-1:0] stream_mask;
    vroom_fp32_simd #(.LANES(LANES),.GROUP_AW($clog2(GROUPS))) engine(
        .clk(clk),.reset(lane_reset),.init_valid(bias_fire),.init_ready(engine_init_ready),
        .init_group(init_group),.init_values(bias_data),.valid(issue),.ready(engine_ready),
        .tag(tag_q),.mask(mask_q),.weights(weights_q),.operand(operand_q),
        .response_valid(engine_valid),.response_tag(engine_tag),.response_mask(engine_mask),
        .results(lane_result),.flags(lane_flags));
    genvar l;
    generate for(l=0;l<LANES;l=l+1)begin: lanes
        localparam integer ROW_GROUPS=(256+LANES-1-l)/LANES;
        localparam [4:0] LANE=l;
        localparam [5:0] ROW_GROUP_COUNT=ROW_GROUPS[5:0];
        wire [5:0] group_count=layer==2 ? (l<2 ? 6'd1 : 6'd0) : ROW_GROUP_COUNT;
        wire init_active=init_group<group_count;
        assign stream_mask[l]=stream_group<group_count;
        wire [31:0] result=lane_result[l*32+:32];
        wire [15:0] returned_tag=lane_tag[l*16+:16];
        reg [5:0] expected_group,completed;
        reg [7:0] expected_column;
        reg done;
        wire final_column=({1'b0,returned_tag[13:6]}+9'd1)==columns;
        wire final_write=lane_valid[l] && final_column && !failed;
        assign lane_done[l]=done;
        wire bad_return=lane_valid[l] &&
            (returned_tag!={2'b0,expected_column,expected_group} ||
             completed==group_count || result[30:23]==8'hff);
        reg bad_return_q,bad_bias_q,bad_weight_q;
        assign response_bad[l]=bad_return_q;
        assign bias_bad[l]=bad_bias_q;
        assign weight_bad[l]=bad_weight_q;
        assign lane_valid[l]=engine_valid && engine_mask[l];
        assign lane_tag[l*16+:16]=engine_tag;
        // Two activation banks, one physical write port per lane. External
        // inputs write bank 0; final hidden sums write the opposite source bank.
        // The banks are not reset. Job/layer sequencing owns initialized data.
        // Power-of-two bank stride makes bank selection one address bit,
        // avoiding an address adder before the wide RAM-to-lane gather path.
        (* ram_style="distributed" *) reg [31:0] activations [0:2*(1<<GROUP_AW)-1];
        wire input_write=input_fire && read_lane==LANE;
        wire activation_write=final_write && layer!=2;
        wire write_enable=input_write || activation_write;
        wire [ACTIVATION_AW-1:0] input_address={1'b0,read_group};
        wire [ACTIVATION_AW-1:0] output_address={!source_bank,returned_tag[GROUP_AW-1:0]};
        (* keep="true" *) reg [ACTIVATION_AW-1:0] read_address;
        // ReLU is x<0 ? +0 : x, retaining signed zero; nonfinite sums fault.
        wire [31:0] relu=(result[31] && (|result[30:0])) ? 32'b0 : result;
        always @(posedge clk)begin
            read_address<={source_bank,next_read_group};
            bad_bias_q<=init_active && bias_data[l*32+23+:8]==8'hff;
            bad_weight_q<=stream_mask[l] && weight_data[l*32+23+:8]==8'hff;
            if(reset)begin bad_bias_q<=0;bad_weight_q<=0;end
            // Validate beside the lane before crossing the array to the FSM.
            // DRAIN waits for retirement; a bad final return is therefore
            // caught before tanh admission or publishing a result.
            if(lane_reset)bad_return_q<=0;else bad_return_q<=bad_return;
            if(write_enable)activations[input_write ? input_address : output_address] <= input_write ? input_data : relu;
            if(lane_reset)begin expected_group<=0;expected_column<=0;completed<=0;done<=group_count==0;end
            else if(lane_valid[l])begin
                if(final_column)begin completed<=completed+1'b1;done<=completed+6'd1==group_count;end
                if(expected_group+6'd1==group_count)begin expected_group<=0;expected_column<=expected_column+1'b1;end
                else expected_group<=expected_group+1'b1;
            end
        end
        assign bank_reads[l*32+:32]=activations[read_address];
    end endgenerate

    reg [31:0] final0,final1;
    reg tanh_index;
    wire tanh_ready,tanh_response,tanh_invalid;
    wire [15:0] tanh_tag;
    wire [31:0] tanh_result;
    vroom_fp32_tanh #(.RESET_GATES_HANDSHAKES(0)) activation(
        .clk(clk),.reset(reset || failed),.valid(state==TANH_SEND),.ready(tanh_ready),
        .tag({15'b0,tanh_index}),.a(tanh_index ? final1 : final0),
        .response_valid(tanh_response),.response_ready(state==TANH_WAIT),
        .response_tag(tanh_tag),.result(tanh_result),.invalid(tanh_invalid));
    reg [4:0] response_flags;
    integer j;
    always @*begin
        response_flags=0;
        for(j=0;j<LANES;j=j+1)if(lane_valid[j])response_flags=response_flags | lane_flags[j*5+:5];
    end
    // Validate each accepted stream independently, then report the registered
    // result to the controller. This cuts ready/issue/metadata comparison out
    // of the state feedback path. A fault flushes ownership one clock later;
    // the pipeline cannot publish that job in this interval.
    reg input_error,bias_error,weight_error,bias_accepted_q,weight_accepted_q;
    always @(posedge clk)begin
        bias_accepted_q<=bias_fire;weight_accepted_q<=weight_fire;
        input_error<=input_fire && (input_last!=last_input || input_data[30:23]==8'hff);
        bias_error<=bias_fire && (bias_job!=output_job || bias_layer!=layer || bias_group!=init_group || bias_last!=last_bias);
        weight_error<=weight_fire && (weight_job!=output_job || weight_layer!=layer || weight_column!=stream_column || weight_group!=stream_group || weight_last!=(last_group && last_column));
        if(reset)begin input_error<=0;bias_error<=0;weight_error<=0;bias_accepted_q<=0;weight_accepted_q<=0;end
    end
    wire bias_failure=bias_error || (bias_accepted_q && (|bias_bad));
    wire weight_failure=weight_error || (weight_accepted_q && (|weight_bad));
    wire return_failure=|response_bad;
    wire tanh_failure=state==TANH_WAIT && tanh_response && (tanh_tag!={15'b0,tanh_index} || tanh_invalid);
    // Failure owns a separate sticky latch, not a priority override of every
    // normal state transition. Once latched, external transfers are suppressed
    // and both arithmetic units are reset. Internal payload/state is then
    // irrelevant until reset; it cannot publish a failed job.
    always @(posedge clk)begin
        if(input_error || bias_failure || weight_failure || return_failure || tanh_failure)failed<=1;
        if(!failed)begin
            if(input_error)fault_code<=1;
            else if(bias_failure)fault_code<=2;
            else if(weight_failure)fault_code<=3;
            else if(return_failure)fault_code<=4;
            else if(tanh_failure)fault_code<=5;
        end
        if(reset)begin failed<=0;fault_code<=0;end
    end
    always @(posedge clk)begin
            // An empty/consumed elastic slot may sample arbitrary payload:
            // pending alone marks validity. Keep valid/state decoding out of
            // the wide payload enable while preserving a blocked live slot.
            if(!pending || engine_ready)begin
                weights_q<=weight_data;operand_q<=input_hold;
                tag_q<={2'b0,stream_column,stream_group};mask_q<=stream_mask;
            end
            if(issue)pending<=0;
            arithmetic_flags<=arithmetic_flags | response_flags;
            if(layer==2)begin
                if(lane_valid[0] && ({1'b0,lane_tag[13:6]}+9'd1)==columns)final0<=lane_result[31:0];
                if(lane_valid[1] && ({1'b0,lane_tag[29:22]}+9'd1)==columns)final1<=lane_result[63:32];
            end
            case(state)
                IDLE: if(job_valid)begin
                    output_actor<=job_actor;output_job<=job_id;columns<=job_actor ? 9'd19 : 9'd25;
                    layer<=0;source_bank<=0;input_count<=0;read_lane<=0;read_group<=0;
                    arithmetic_flags<=0;state<=INPUTS;
                end
                INPUTS: if(input_fire)begin
                    if(last_input)state<=LAYER_RESET;
                    else input_count<=input_count+1'b1;
                    if(read_lane==LAST_LANE)begin read_lane<=0;read_group<=read_group+1'b1;end
                    else read_lane<=read_lane+1'b1;
                end
                LAYER_RESET: begin
                    init_group<=0;stream_column<=0;stream_group<=0;pending<=0;
                    read_lane<=0;read_group<=0;state<=BIASES;
                end
                BIASES: if(bias_fire)begin
                    if(last_bias)state<=FETCH;else init_group<=init_group+1'b1;
                end
                FETCH: begin input_hold<=bank_reads[31:0];read_lane<=1;state<=RUN;end
                RUN: if(weight_fire)begin
                    pending<=1;
                    if(last_group)begin
                        stream_group<=0;
                        if(last_column)state<=DRAIN;
                        else begin
                            stream_column<=stream_column+1'b1;
                            input_hold<=bank_reads[read_lane*32+:32];
                            if(read_lane==LAST_LANE)begin read_lane<=0;read_group<=read_group+1'b1;end
                            else read_lane<=read_lane+1'b1;
                        end
                    end else stream_group<=stream_group+1'b1;
                end
                DRAIN: if(&lane_done)begin
                    if(layer==2)begin tanh_index<=0;state<=TANH_SEND;end
                    else begin layer<=layer+1'b1;columns<=256;source_bank<=!source_bank;state<=LAYER_RESET;end
                end
                TANH_SEND: if(tanh_ready)state<=TANH_WAIT;
                TANH_WAIT: if(tanh_response)begin
                    if(tanh_index)begin output_data[63:32]<=tanh_result;state<=OUTPUTS;end
                    else begin output_data[31:0]<=tanh_result;tanh_index<=1;state<=TANH_SEND;end
                end
                OUTPUTS: if(output_ready)state<=IDLE;
                default: ;
            endcase
        if(reset)begin
            state<=IDLE;layer<=0;source_bank<=0;pending<=0;mask_q<=0;
            tag_q<=0;read_group<=0;read_lane<=0;init_group<=0;input_count<=0;
            stream_column<=0;stream_group<=0;columns<=25;tanh_index<=0;
            output_data<=0;output_actor<=0;output_job<=0;arithmetic_flags<=0;
        end
    end
endmodule
`default_nettype wire
