`default_nettype none
// Lockstep binary32 lanes with one shared credit/tag/hazard controller. The
// caller initializes every row group before issuing that layer. All lanes
// advance, but masks suppress unused row results. Payloads use SRL-friendly
// shift queues; their depth counts multiplications still in flight as well.
module vroom_fp32_simd #(
    parameter integer LANES=12,
    parameter integer GROUP_AW=5
) (
    input wire clk,reset,
    input wire init_valid,
    output wire init_ready,
    input wire [5:0] init_group,
    input wire [LANES*32-1:0] init_values,
    input wire valid,
    output wire ready,
    input wire [15:0] tag,
    input wire [LANES-1:0] mask,
    input wire [LANES*32-1:0] weights,
    input wire [31:0] operand,
    output wire response_valid,
    output wire [15:0] response_tag,
    output wire [LANES-1:0] response_mask,
    output wire [LANES*32-1:0] results,
    output wire [LANES*5-1:0] flags
);
    localparam integer GROUPS=1<<GROUP_AW;
    localparam integer MUL_LATENCY=6,ADD_LATENCY=9;
    reg [GROUPS-1:0] busy;
    reg [4:0] queued,credits;
    reg init_idle;
    wire fire=valid && ready;
    reg [MUL_LATENCY-1:0] mul_valid_pipe;
    reg [15:0] mul_tags [0:MUL_LATENCY-1];
    reg [LANES-1:0] mul_masks [0:MUL_LATENCY-1];
    // Payload queues need no reset gating: synchronous reset clears their
    // occupancy/valid state. An old payload write on the reset edge is harmless.
    wire mul_valid=mul_valid_pipe[MUL_LATENCY-1];
    // Maintain count-1 directly, avoiding a carry chain before the wide SRL tap.
    reg [3:0] front;
    wire [15:0] front_tag;
    wire [LANES-1:0] front_mask;
    vroom_stream_shift #(.WIDTH(16+LANES)) metadata_queue(
        .clk(clk),.enable(mul_valid),.data({mul_tags[MUL_LATENCY-1],mul_masks[MUL_LATENCY-1]}),
        .address(front),.value({front_tag,front_mask}));
    // Two head slots absorb one cycle of registered pop control. The next
    // pop reserves a free slot using post-edge occupancy, so no combinational
    // row-hazard signal drives every lane's payload clock enable.
    reg pop,head_read,head_write;
    reg [1:0] head_count;
    wire head_valid=head_count!=0;
    reg [15:0] head_tags [0:1];
    reg [LANES-1:0] head_masks [0:1];
    reg [GROUPS-1:0] head_rows [0:1];
    wire [15:0] head_tag=head_tags[head_read];
    wire [LANES-1:0] head_mask=head_masks[head_read];
    wire [GROUPS-1:0] head_row=head_rows[head_read];
    wire add_fire=head_valid && !(|(busy & head_row));
    wire [4:0] queued_next=queued+{4'b0,mul_valid}-{4'b0,pop};
    wire [1:0] head_next=head_count+{1'b0,pop}-{1'b0,add_fire};
    reg add_valid_q;
    reg [15:0] add_tag_q;
    reg [LANES-1:0] add_mask_q;
    reg [ADD_LATENCY-1:0] add_valid_pipe;
    reg [15:0] add_tags [0:ADD_LATENCY-1];
    reg [LANES-1:0] add_masks [0:ADD_LATENCY-1];
    assign response_valid=add_valid_pipe[ADD_LATENCY-1];
    assign response_tag=add_tags[ADD_LATENCY-1];
    assign response_mask=add_masks[ADD_LATENCY-1];
    // A multiply request has priority over initialization. The caller normally
    // uses separate phases, but even simultaneous valids cannot accept both.
    // Keep the wide initialization/hazard predicate out of MAC readiness.
    assign ready=credits<16;
    assign init_ready=!valid && init_idle;
    wire init_fire=init_valid && init_ready;
    integer i;
    always @(posedge clk)begin
        mul_tags[0]<=tag;mul_masks[0]<=mask;
        for(i=1;i<MUL_LATENCY;i=i+1)begin mul_tags[i]<=mul_tags[i-1];mul_masks[i]<=mul_masks[i-1];end
        if(pop)begin
            head_tags[head_write]<=front_tag;head_masks[head_write]<=front_mask;
            head_rows[head_write]<={{(GROUPS-1){1'b0}},1'b1} << front_tag[GROUP_AW-1:0];
        end
        add_tag_q<=head_tag;add_mask_q<=head_mask;
        add_tags[0]<=add_tag_q;add_masks[0]<=add_mask_q;
        for(i=1;i<ADD_LATENCY;i=i+1)begin add_tags[i]<=add_tags[i-1];add_masks[i]<=add_masks[i-1];end
        if(reset)begin
            busy<=0;queued<=0;front<=15;credits<=0;head_count<=0;pop<=0;head_read<=0;head_write<=0;mul_valid_pipe<=0;add_valid_q<=0;add_valid_pipe<=0;init_idle<=1;
        end else begin
            // Conservative registered drain status. New work clears it on the
            // accepting edge; becoming idle may take one extra clock. Layer
            // reset makes it immediately available for the next bias stream.
            init_idle<=!fire && !pop && !add_fire && credits==0 && !head_valid && busy==0;
            mul_valid_pipe<={mul_valid_pipe[MUL_LATENCY-2:0],fire};
            add_valid_q<=add_fire;add_valid_pipe<={add_valid_pipe[ADD_LATENCY-2:0],add_valid_q};
            head_count<=head_next;
            pop<=queued_next!=0 && head_next<2;
            if(pop)head_write<=!head_write;
            if(add_fire)head_read<=!head_read;
            busy<=(busy | (add_fire ? head_row : {GROUPS{1'b0}})) &
                ~(response_valid ? ({{(GROUPS-1){1'b0}},1'b1} << response_tag[GROUP_AW-1:0]) : {GROUPS{1'b0}});
            case({mul_valid,pop})
                2'b10:begin queued<=queued+1'b1;front<=front+1'b1;end
                2'b01:begin queued<=queued-1'b1;front<=front-1'b1;end
                default:;
            endcase
            case({fire,pop})
                2'b10:credits<=credits+1'b1;
                2'b01:credits<=credits-1'b1;
                default:;
            endcase
        end
    end
    genvar l;
    generate for(l=0;l<LANES;l=l+1)begin: lanes
        wire [31:0] product;
        wire [4:0] product_flags,add_flags;
        // Shared valid/tag pipelines above control identical fixed-latency FPUs.
        wire unused_mul_valid,unused_add_valid;
        wire [15:0] unused_mul_tag,unused_add_tag;
        vroom_fp32_mul multiplier(.clk(clk),.reset(reset),.enable(1'b1),.valid(fire),.tag(16'b0),
            .a(weights[l*32+:32]),.b(operand),.response_valid(unused_mul_valid),
            .response_tag(unused_mul_tag),.result(product),.flags(product_flags));
        (* ram_style="distributed" *) reg [31:0] accumulator [0:GROUPS-1];
        reg [31:0] add_a,add_b;
        reg [4:0] flags_q;
        reg [4:0] flag_pipe [0:ADD_LATENCY-1];
        wire [36:0] front_product;
        reg [36:0] head_products [0:1];
        wire [36:0] head_product=head_products[head_read];
        vroom_stream_shift #(.WIDTH(37)) product_queue(.clk(clk),.enable(mul_valid),
            .data({product_flags,product}),.address(front),.value(front_product));
        wire write_enable=init_fire || response_valid;
        wire [GROUP_AW-1:0] write_address=response_valid ? response_tag[GROUP_AW-1:0] : init_group[GROUP_AW-1:0];
        wire [31:0] write_value=response_valid ? results[l*32+:32] : init_values[l*32+:32];
        integer stage;
        always @(posedge clk)begin
            if(write_enable)accumulator[write_address]<=write_value;
            if(pop)head_products[head_write]<=front_product;
            add_a<=accumulator[head_tag[GROUP_AW-1:0]];add_b<=head_product[31:0];flags_q<=head_product[36:32];
            flag_pipe[0]<=flags_q;
            for(stage=1;stage<ADD_LATENCY;stage=stage+1)flag_pipe[stage]<=flag_pipe[stage-1];
        end
        vroom_fp32_add adder(.clk(clk),.reset(reset),.enable(1'b1),.valid(add_valid_q),.tag(16'b0),
            .a(add_a),.b(add_b),.response_valid(unused_add_valid),
            .response_tag(unused_add_tag),.result(results[l*32+:32]),.flags(add_flags));
        assign flags[l*5+:5]=add_flags | flag_pipe[ADD_LATENCY-1];
    end endgenerate
    wire unused_init_group=|init_group[5:GROUP_AW];
endmodule
`default_nettype wire
