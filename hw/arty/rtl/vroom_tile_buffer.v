`default_nettype none
// One 2 KiB, actor-owned weight tile. Payload is opaque IEEE-754 binary32.
// DMA order is column-major within a tile: consecutive output rows first.
// Sixteen single-read banks allow up to 16 consecutive words per core cycle.
// All interfaces are synchronous to clk; DMA CDC belongs outside this block.
module vroom_tile_buffer #(
    parameter integer LANES = 12
) (
    input wire clk,
    input wire reset,
    input wire load_valid,
    output wire load_ready,
    input wire [31:0] load_tag,
    input wire [9:0] load_words,
    input wire dma_valid,
    output wire dma_ready,
    input wire [31:0] dma_tag,
    input wire [127:0] dma_data,
    input wire dma_last,
    input wire status_valid,
    output wire status_ready,
    input wire [31:0] status_tag,
    input wire status_ok,
    input wire [11:0] status_bytes,
    input wire acquire_valid,
    output wire acquire_ready,
    input wire [31:0] acquire_tag,
    input wire release_valid,
    output wire release_ready,
    input wire [31:0] release_tag,
    input wire read_valid,
    output wire read_ready,
    input wire [8:0] read_base,
    input wire [LANES-1:0] read_mask,
    output reg response_valid,
    output reg [LANES-1:0] response_mask,
    output reg [LANES*32-1:0] response_data,
    output reg [1:0] state,
    output reg fault,
    output reg [3:0] fault_code
);
    localparam [1:0] EMPTY=0, FILLING=1, READY=2, READING=3;
    reg [31:0] tag;
    reg [9:0] words, written;
    reg data_done, status_done;
    wire load_fire = load_valid && load_ready;
    wire dma_fire = dma_valid && dma_ready;
    wire status_fire = status_valid && status_ready;
    wire read_fire = read_valid && read_ready;
    wire final_beat = (written + 10'd4 == words);
    wire good_beat = dma_fire && dma_tag == tag && dma_last == final_beat;
    wire [LANES-1:0] lane_out_of_bounds;
    wire read_out_of_bounds = |lane_out_of_bounds;
    wire [511:0] bank_data;
    wire [LANES*32-1:0] selected_data;
    // One shared subtract replaces LANES independent add/compare paths.
    // Bit 10 detects underflow; retain it so an out-of-range base cannot wrap.
    wire [10:0] remaining = {1'b0,words} - {2'b0,read_base};

    assign load_ready = state == EMPTY && !fault && !reset;
    assign dma_ready = state == FILLING && !data_done && !fault && !reset;
    assign status_ready = !fault && !reset;
    assign acquire_ready = state == READY && !fault && !reset;
    assign release_ready = state == READING && !fault && !reset;
    assign read_ready = state == READING && !fault && !reset;

    genvar b, lane;
    generate for (b=0; b<16; b=b+1) begin : banks
        localparam integer WRITE_GROUP = b/4;
        localparam [3:0] BANK = b;
        // For a consecutive vector, banks below the starting bank wrap to
        // the next word address. Unselected banks may read arbitrary payload.
        wire wrap;
        if (b == 15) begin : last_bank
            assign wrap = 1'b0;
        end else begin : wrapping_bank
            assign wrap = BANK < read_base[3:0];
        end
        wire [4:0] address = read_base[8:4] + {4'b0, wrap};
        // One async read and one synchronous write per bank; 16 x 32 x 32 bits.
        // No reset on payload RAM: validity/ownership is reset instead.
        (* ram_style = "distributed" *) reg [31:0] mem [0:31];
        assign bank_data[b*32 +: 32] = mem[address];
        always @(posedge clk) begin
            if (good_beat && written[3:2] == WRITE_GROUP[1:0])
                mem[written[8:4]] <= dma_data[(b%4)*32 +: 32];
        end
    end
    for (lane=0; lane<LANES; lane=lane+1) begin : read_lanes
        localparam [10:0] OFFSET = lane;
        assign lane_out_of_bounds[lane] = read_mask[lane] && (remaining[10] || remaining <= OFFSET);
    end endgenerate
    vroom_bank_rotate #(.WORDS(LANES)) read_rotation (
        .data_in(bank_data), .offset(read_base[3:0]), .data_out(selected_data));

    integer output_lane;
    always @(posedge clk) begin
        response_valid <= 1'b0;
        if (reset) begin
            state <= EMPTY;
            fault <= 1'b0;
            fault_code <= 0;
            tag <= 0;
            words <= 0;
            written <= 0;
            data_done <= 1'b0;
            status_done <= 1'b0;
            response_mask <= 0;
        end else if (!fault) begin
            if (load_fire) begin
                if (load_words == 0 || load_words > 512 || load_words[1:0] != 0) begin
                    fault <= 1'b1;
                    fault_code <= 1;
                end else begin
                    state <= FILLING;
                    tag <= load_tag;
                    words <= load_words;
                    written <= 0;
                    data_done <= 1'b0;
                    status_done <= 1'b0;
                end
            end
            if (dma_fire) begin
                if (dma_tag != tag) begin
                    fault <= 1'b1;
                    fault_code <= 2;
                end else if (dma_last != final_beat) begin
                    fault <= 1'b1;
                    fault_code <= 3;
                end else begin
                    written <= written + 10'd4;
                    data_done <= final_beat;
                end
            end
            // Status may cross the CDC before or after the final data beat.
            if (status_fire) begin
                if (state != FILLING || status_done || status_tag != tag ||
                    !status_ok || status_bytes != {words, 2'b00}) begin
                    fault <= 1'b1;
                    fault_code <= 4;
                end else status_done <= 1'b1;
            end
            if (state == FILLING && data_done && status_done)
                state <= READY;
            if (acquire_valid && acquire_ready) begin
                if (acquire_tag != tag) begin
                    fault <= 1'b1;
                    fault_code <= 5;
                end else state <= READING;
            end
            if (release_valid && release_ready) begin
                if (release_tag != tag) begin
                    fault <= 1'b1;
                    fault_code <= 6;
                end else state <= EMPTY;
            end
            if (read_fire) begin
                if (read_out_of_bounds) begin
                    fault <= 1'b1;
                    fault_code <= 7;
                end else begin
                    response_valid <= 1'b1;
                    response_mask <= read_mask;
                    for (output_lane=0; output_lane<LANES; output_lane=output_lane+1) begin
                        response_data[output_lane*32 +: 32] <=
                            read_mask[output_lane] ?
                            selected_data[output_lane*32 +: 32] : 32'b0;
                    end
                end
            end
        end
    end
`ifndef SYNTHESIS
    initial if (LANES < 1 || LANES > 16) $fatal(1, "LANES must be 1..16");
`endif
endmodule
`default_nettype wire
