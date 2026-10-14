`default_nettype none
// One multiply-ahead lane, 16 reserved product slots, 64 local accumulators.
// tag[5:0] selects a local accumulator; tag[15:6] is returned opaque metadata.
// Initialization is accepted only when all operations have drained. Accepted
// products retire in input order. A row cannot read until its previous write
// completed on an earlier edge (no read-during-write or forwarding assumption).
// Responses have no backpressure; every response is also stored locally.
module vroom_fp32_lane (
    input wire clk, reset,
    input wire init_valid,
    output wire init_ready,
    input wire [5:0] init_group,
    input wire [31:0] init_value,
    input wire valid,
    output wire ready,
    input wire [15:0] tag,
    input wire [31:0] a, b,
    output wire response_valid,
    output wire [15:0] response_tag,
    output wire [31:0] result,
    output wire [4:0] flags
);
    (* ram_style="distributed" *) reg [31:0] accumulator [0:63];
    reg [63:0] initialized, busy;
    (* ram_style="distributed" *) reg [31:0] products [0:15];
    (* ram_style="distributed" *) reg [20:0] metadata [0:15];
    reg [3:0] head, tail;
    reg [4:0] queued, credits;
    wire mul_valid;
    wire [15:0] mul_tag;
    wire [31:0] mul_result;
    wire [4:0] mul_flags;
    wire [20:0] front_metadata = metadata[head];
    wire [15:0] front_tag = front_metadata[15:0];
    wire add_fire = !reset && queued != 0 && !busy[front_tag[5:0]];
    // Register the local RAM read before binary32 recoding. Otherwise the
    // critical path crosses FIFO tag RAM, accumulator RAM and normalization.
    reg add_valid_q;
    reg [31:0] add_a_q,add_b_q;
    reg [15:0] add_tag_q;
    reg [4:0] add_mul_flags_q;
    wire [4:0] add_flags;
    localparam integer ADD_LATENCY = 9;
    reg [4:0] mul_flag_pipe [0:ADD_LATENCY-1];
    integer stage;
    assign init_ready = !reset && credits == 0 && busy == 0;
    assign ready = !reset && !init_valid && initialized[tag[5:0]] && credits < 16;
    wire fire = valid && ready;
    wire init_fire = init_valid && init_ready;
    vroom_fp32_mul multiplier(
        .clk(clk),.reset(reset),.enable(1'b1),.valid(fire),.tag(tag),.a(a),.b(b),
        .response_valid(mul_valid),.response_tag(mul_tag),.result(mul_result),.flags(mul_flags));
    vroom_fp32_add adder(
        .clk(clk),.reset(reset),.enable(1'b1),.valid(add_valid_q),.tag(add_tag_q),
        .a(add_a_q),.b(add_b_q),
        .response_valid(response_valid),.response_tag(response_tag),.result(result),.flags(add_flags));
    assign flags = add_flags | mul_flag_pipe[ADD_LATENCY-1];
    // One physical write port; initialization and retirement are exclusive.
    wire acc_write = init_fire || response_valid;
    wire [5:0] acc_address = response_valid ? response_tag[5:0] : init_group;
    wire [31:0] acc_data = response_valid ? result : init_value;
    always @(posedge clk) begin
        if (acc_write) accumulator[acc_address] <= acc_data;
        add_valid_q <= !reset && add_fire;
        add_a_q <= accumulator[front_tag[5:0]]; add_b_q <= products[head];
        add_tag_q <= front_tag; add_mul_flags_q <= front_metadata[20:16];
        mul_flag_pipe[0] <= add_mul_flags_q;
        for (stage=1;stage<ADD_LATENCY;stage=stage+1) mul_flag_pipe[stage] <= mul_flag_pipe[stage-1];
        if (reset) begin
            initialized <= 0; busy <= 0; head <= 0; tail <= 0;
            queued <= 0; credits <= 0;
        end else begin
            if (init_fire) initialized[init_group] <= 1;
            if (mul_valid) begin
                products[tail] <= mul_result;
                metadata[tail] <= {mul_flags,mul_tag};
                tail <= tail+1'b1;
            end
            if (add_fire) begin
                busy[front_tag[5:0]] <= 1;
                head <= head+1'b1;
            end
            if (response_valid) busy[response_tag[5:0]] <= 0;
            case ({mul_valid,add_fire})
                2'b10: queued <= queued+1'b1;
                2'b01: queued <= queued-1'b1;
                default: ;
            endcase
            case ({fire,add_fire})
                2'b10: credits <= credits+1'b1;
                2'b01: credits <= credits-1'b1;
                default: ;
            endcase
        end
    end
endmodule
`default_nettype wire
