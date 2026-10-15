`default_nettype none
// Resident inner W1 rows 0..175, all 256 columns: 45,056 opaque FP32 words.
// Sequential boot stream maps index -> bank=index%11, address=index/11.
// After a complete load and seal, each bank supplies two synchronous reads.
// No reset on RAM payload; reset invalidates the whole model and requires reload.
module vroom_pinned_cache (
    input wire clk, reset,
    input wire boot_valid,
    output wire boot_ready,
    input wire [31:0] boot_data,
    input wire seal,
    output reg sealed,
    output reg fault,
    output reg [15:0] loaded_words,
    input wire [10:0] read_a,
    input wire [131:0] address_a,
    input wire [10:0] read_b,
    input wire [131:0] address_b,
    output reg [10:0] valid_a, valid_b,
    output wire [351:0] data_a, data_b
);
    reg [3:0] boot_bank;
    reg [11:0] boot_address;
    assign boot_ready = !sealed && !fault && !reset && loaded_words < 16'd45056;
    wire boot_fire = boot_valid && boot_ready;
    genvar bank;
    generate for (bank=0; bank<11; bank=bank+1) begin : banks
        (* ram_style = "block" *) reg [31:0] mem [0:4095];
        reg [31:0] qa, qb;
        // Boot and inference are exclusive phases. Explicitly share physical
        // port A's address between boot writes and inference reads, leaving
        // port B for the second read instead of inferring a replicated RAM.
        wire [11:0] port_a_address = sealed ? address_a[bank*12 +: 12] : boot_address;
        assign data_a[bank*32 +: 32] = qa;
        assign data_b[bank*32 +: 32] = qb;
        always @(posedge clk) begin
            if (boot_fire && boot_bank == bank) mem[port_a_address] <= boot_data;
            if (sealed && !fault && !reset && read_a[bank]) qa <= mem[port_a_address];
        end
        always @(posedge clk) begin
            if (sealed && !fault && !reset && read_b[bank]) qb <= mem[address_b[bank*12 +: 12]];
        end
    end endgenerate
    always @(posedge clk) begin
        valid_a <= 0;
        valid_b <= 0;
        if (reset) begin
            boot_bank <= 0;
            boot_address <= 0;
            loaded_words <= 0;
            sealed <= 1'b0;
            fault <= 1'b0;
        end else if (!fault) begin
            if (boot_fire) begin
                loaded_words <= loaded_words + 1'b1;
                if (boot_bank == 10) begin
                    boot_bank <= 0;
                    boot_address <= boot_address + 1'b1;
                end else boot_bank <= boot_bank + 1'b1;
            end
            // Seal is a separate transaction, after the final boot handshake.
            if (seal && !sealed) begin
                if (loaded_words != 16'd45056) fault <= 1'b1;
                else sealed <= 1'b1;
            end
            if (sealed) begin
                valid_a <= read_a;
                valid_b <= read_b;
            end
        end
    end
endmodule
`default_nettype wire
