`default_nettype none
// Independent release/deadline instrumentation; this is NOT a dispatcher.
// At 100 MHz: 10,000 cycles = 100 us, 100,000 cycles = 1 ms.
// Both actors release on start. Done at the exact next release edge is on time.
// Job IDs distinguish completion of the preceding job from the new release.
// No actor command is delayed/committed here; controller timing policy is external.
module vroom_rate_monitor #(
    parameter integer INNER_CYCLES = 10000,
    parameter integer OUTER_CYCLES = 100000
) (
    input wire clk, reset, start,
    input wire inner_done,
    input wire [31:0] inner_done_job,
    input wire outer_done,
    input wire [31:0] outer_done_job,
    output reg running,
    output reg inner_release, outer_release,
    output reg [31:0] inner_job, outer_job,
    output reg inner_pending, outer_pending,
    output reg [63:0] elapsed_cycles,
    output reg fault,
    output reg fault_actor,
    output reg [1:0] fault_code,
    output reg [31:0] fault_job,
    output reg [63:0] fault_cycle
);
    reg [31:0] inner_phase, outer_phase;
    wire inner_boundary = inner_phase == INNER_CYCLES-1;
    wire outer_boundary = outer_phase == OUTER_CYCLES-1;
    wire bad_inner_done = inner_done && (!inner_pending || inner_done_job != inner_job);
    wire bad_outer_done = outer_done && (!outer_pending || outer_done_job != outer_job);
    wire inner_missed = inner_boundary && inner_pending && !inner_done;
    wire outer_missed = outer_boundary && outer_pending && !outer_done;
    always @(posedge clk) begin
        inner_release <= 1'b0;
        outer_release <= 1'b0;
        if (reset) begin
            running <= 1'b0;
            fault <= 1'b0;
            fault_actor <= 1'b0;
            fault_code <= 0;
            fault_job <= 0;
            fault_cycle <= 0;
            inner_phase <= 0;
            outer_phase <= 0;
            elapsed_cycles <= 0;
            inner_job <= 0;
            outer_job <= 0;
            inner_pending <= 1'b0;
            outer_pending <= 1'b0;
        end else if (!fault) begin
            if (!running && start) begin
                running <= 1'b1;
                inner_release <= 1'b1;
                outer_release <= 1'b1;
                inner_pending <= 1'b1;
                outer_pending <= 1'b1;
            end else if (running) begin
                elapsed_cycles <= elapsed_cycles + 1'b1;
                if (bad_inner_done || bad_outer_done || inner_missed || outer_missed) begin
                    fault <= 1'b1;
                    running <= 1'b0;
                    // Protocol errors first, then earliest-period actor if both miss.
                    fault_actor <= bad_inner_done ? 1'b0 : bad_outer_done ? 1'b1 : !inner_missed;
                    fault_code <= (bad_inner_done || bad_outer_done) ? 2'd2 : 2'd1;
                    fault_job <= (bad_inner_done || (!bad_outer_done && inner_missed)) ? inner_job : outer_job;
                    fault_cycle <= elapsed_cycles + 1'b1;
                end else begin
                    inner_phase <= inner_boundary ? 32'b0 : inner_phase + 1'b1;
                    outer_phase <= outer_boundary ? 32'b0 : outer_phase + 1'b1;
                    if (inner_done) inner_pending <= 1'b0;
                    if (outer_done) outer_pending <= 1'b0;
                    if (inner_boundary) begin
                        inner_job <= inner_job + 1'b1;
                        inner_pending <= 1'b1;
                        inner_release <= 1'b1;
                    end
                    if (outer_boundary) begin
                        outer_job <= outer_job + 1'b1;
                        outer_pending <= 1'b1;
                        outer_release <= 1'b1;
                    end
                end
            end
        end
    end
`ifndef SYNTHESIS
    initial if (INNER_CYCLES < 2 || OUTER_CYCLES < INNER_CYCLES || OUTER_CYCLES % INNER_CYCLES != 0)
        $fatal(1, "Periods must be >=2 cycles and outer an integer multiple of inner");
`endif
endmodule
`default_nettype wire
