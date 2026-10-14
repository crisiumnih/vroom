`default_nettype none
// Buffer IDs: 0/1 inner A/B; 2/3 outer A/B. Four independent ownership FSMs.
// A single 128-bit DMA stream can fill one tile while any other is read.
// Tags must identify the complete transfer (including epoch/job/tile identity).
module vroom_weight_tiles #(
    parameter integer LANES = 12
) (
    input wire clk, reset,
    input wire load_valid,
    output wire load_ready,
    input wire [1:0] load_buffer,
    input wire [31:0] load_tag,
    input wire [9:0] load_words,
    input wire dma_valid,
    output wire dma_ready,
    input wire [1:0] dma_buffer,
    input wire [31:0] dma_tag,
    input wire [127:0] dma_data,
    input wire dma_last,
    input wire status_valid,
    output wire status_ready,
    input wire [1:0] status_buffer,
    input wire [31:0] status_tag,
    input wire status_ok,
    input wire [11:0] status_bytes,
    input wire acquire_valid,
    output wire acquire_ready,
    input wire [1:0] acquire_buffer,
    input wire [31:0] acquire_tag,
    input wire release_valid,
    output wire release_ready,
    input wire [1:0] release_buffer,
    input wire [31:0] release_tag,
    input wire read_valid,
    output wire read_ready,
    input wire [1:0] read_buffer,
    input wire [8:0] read_base,
    input wire [LANES-1:0] read_mask,
    output wire response_valid,
    output wire [LANES-1:0] response_mask,
    output wire [LANES*32-1:0] response_data,
    output wire [7:0] states,
    output wire fault,
    output wire [3:0] fault_buffers,
    output wire [15:0] fault_codes
);
    wire [3:0] load_r, dma_r, status_r, acquire_r, release_r, read_r, response_v;
    wire [4*LANES-1:0] response_m;
    wire [4*LANES*32-1:0] response_d;
    reg [1:0] response_buffer;
    assign fault = |fault_buffers;
    assign load_ready = load_r[load_buffer] && !fault;
    assign dma_ready = dma_r[dma_buffer] && !fault;
    assign status_ready = status_r[status_buffer] && !fault;
    assign acquire_ready = acquire_r[acquire_buffer] && !fault;
    assign release_ready = release_r[release_buffer] && !fault;
    assign read_ready = read_r[read_buffer] && !fault;
    assign response_valid = (|response_v) && !fault && !reset;
    assign response_mask = response_m[response_buffer*LANES +: LANES];
    assign response_data = response_d[response_buffer*LANES*32 +: LANES*32];
    always @(posedge clk) begin
        if (reset) response_buffer <= 0;
        else if (read_valid && read_ready) response_buffer <= read_buffer;
    end
    genvar b;
    generate for (b=0; b<4; b=b+1) begin : buffers
        vroom_tile_buffer #(.LANES(LANES)) tile (
            .clk(clk), .reset(reset),
            .load_valid(load_valid && load_buffer == b && !fault), .load_ready(load_r[b]),
            .load_tag(load_tag), .load_words(load_words),
            .dma_valid(dma_valid && dma_buffer == b && !fault), .dma_ready(dma_r[b]),
            .dma_tag(dma_tag), .dma_data(dma_data), .dma_last(dma_last),
            .status_valid(status_valid && status_buffer == b && !fault), .status_ready(status_r[b]),
            .status_tag(status_tag), .status_ok(status_ok), .status_bytes(status_bytes),
            .acquire_valid(acquire_valid && acquire_buffer == b && !fault), .acquire_ready(acquire_r[b]),
            .acquire_tag(acquire_tag),
            .release_valid(release_valid && release_buffer == b && !fault), .release_ready(release_r[b]),
            .release_tag(release_tag),
            .read_valid(read_valid && read_buffer == b && !fault), .read_ready(read_r[b]),
            .read_base(read_base), .read_mask(read_mask),
            .response_valid(response_v[b]), .response_mask(response_m[b*LANES +: LANES]),
            .response_data(response_d[b*LANES*32 +: LANES*32]),
            .state(states[b*2 +: 2]), .fault(fault_buffers[b]), .fault_code(fault_codes[b*4 +: 4])
        );
    end endgenerate
endmodule
`default_nettype wire
