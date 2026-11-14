`default_nettype none
// Binary32 tanh via iterative hyperbolic CORDIC and restoring division.
// Internal signed coordinates/angle: 58 bits, 52 fractional bits. No weights
// or stored activations are converted to a reduced-precision representation.
// Approximation accuracy is separately tested; see ../README.md for limits.
// One request at a time; output/tag/invalid held until response_ready. Reset
// synchronously cancels pending work. Only the invalid exception is reported.
module vroom_fp32_tanh #(parameter integer RESET_GATES_HANDSHAKES=1) (
    input wire clk, reset,
    input wire valid,
    output wire ready,
    input wire [15:0] tag,
    input wire [31:0] a,
    output wire response_valid,
    input wire response_ready,
    output reg [15:0] response_tag,
    output reg [31:0] result,
    output reg invalid
);
    localparam [3:0] IDLE=0,CONVERT=1,REDUCE=2,SHIFT=3,ROTATE=4,
                     DIFFERENCE=5,SCALE=6,FRACTION=7,DIVIDE=8,NORMALIZE=9,PACK=10,HOLD=11,ROUND=12;
    localparam [57:0] ONE=58'h010000000000000;
    // Generated 100-decimal-digit RNE constants; check with sim/constants.py.
    // CONSTANTS_BEGIN
    localparam signed [57:0] INITIAL=58'h01351e87200eec2;
    localparam signed [57:0] LN2=58'h00b17217f7d1cf8;
    function [5:0] shift_index;
        input [5:0] step;
        begin case(step)
            6'd0: shift_index=6'd1;
            6'd1: shift_index=6'd2;
            6'd2: shift_index=6'd3;
            6'd3: shift_index=6'd4;
            6'd4: shift_index=6'd4;
            6'd5: shift_index=6'd5;
            6'd6: shift_index=6'd6;
            6'd7: shift_index=6'd7;
            6'd8: shift_index=6'd8;
            6'd9: shift_index=6'd9;
            6'd10: shift_index=6'd10;
            6'd11: shift_index=6'd11;
            6'd12: shift_index=6'd12;
            6'd13: shift_index=6'd13;
            6'd14: shift_index=6'd13;
            6'd15: shift_index=6'd14;
            6'd16: shift_index=6'd15;
            6'd17: shift_index=6'd16;
            6'd18: shift_index=6'd17;
            6'd19: shift_index=6'd18;
            6'd20: shift_index=6'd19;
            6'd21: shift_index=6'd20;
            6'd22: shift_index=6'd21;
            6'd23: shift_index=6'd22;
            6'd24: shift_index=6'd23;
            6'd25: shift_index=6'd24;
            6'd26: shift_index=6'd25;
            6'd27: shift_index=6'd26;
            6'd28: shift_index=6'd27;
            6'd29: shift_index=6'd28;
            6'd30: shift_index=6'd29;
            6'd31: shift_index=6'd30;
            6'd32: shift_index=6'd31;
            6'd33: shift_index=6'd32;
            6'd34: shift_index=6'd33;
            6'd35: shift_index=6'd34;
            6'd36: shift_index=6'd35;
            6'd37: shift_index=6'd36;
            6'd38: shift_index=6'd37;
            6'd39: shift_index=6'd38;
            6'd40: shift_index=6'd39;
            6'd41: shift_index=6'd40;
            6'd42: shift_index=6'd40;
            6'd43: shift_index=6'd41;
            6'd44: shift_index=6'd42;
            6'd45: shift_index=6'd43;
            6'd46: shift_index=6'd44;
            6'd47: shift_index=6'd45;
            6'd48: shift_index=6'd46;
            6'd49: shift_index=6'd47;
            6'd50: shift_index=6'd48;
            6'd51: shift_index=6'd49;
            6'd52: shift_index=6'd50;
            6'd53: shift_index=6'd51;
            6'd54: shift_index=6'd52;
            default: shift_index=6'd52;
        endcase end
    endfunction
    function signed [57:0] angle;
        input [5:0] step;
        begin case(step)
            6'd0: angle=58'h008c9f53d568185;
            6'd1: angle=58'h004162bbea04514;
            6'd2: angle=58'h00202b12393d5df;
            6'd3: angle=58'h001005588ad375b;
            6'd4: angle=58'h001005588ad375b;
            6'd5: angle=58'h000800aac448d77;
            6'd6: angle=58'h0004001556222b4;
            6'd7: angle=58'h00020002aab1112;
            6'd8: angle=58'h000100005555889;
            6'd9: angle=58'h000080000aaaac4;
            6'd10: angle=58'h000040000155556;
            6'd11: angle=58'h00002000002aaab;
            6'd12: angle=58'h000010000005555;
            6'd13: angle=58'h000008000000aab;
            6'd14: angle=58'h000008000000aab;
            6'd15: angle=58'h000004000000155;
            6'd16: angle=58'h00000200000002b;
            6'd17: angle=58'h000001000000005;
            6'd18: angle=58'h000000800000001;
            6'd19: angle=58'h000000400000000;
            6'd20: angle=58'h000000200000000;
            6'd21: angle=58'h000000100000000;
            6'd22: angle=58'h000000080000000;
            6'd23: angle=58'h000000040000000;
            6'd24: angle=58'h000000020000000;
            6'd25: angle=58'h000000010000000;
            6'd26: angle=58'h000000008000000;
            6'd27: angle=58'h000000004000000;
            6'd28: angle=58'h000000002000000;
            6'd29: angle=58'h000000001000000;
            6'd30: angle=58'h000000000800000;
            6'd31: angle=58'h000000000400000;
            6'd32: angle=58'h000000000200000;
            6'd33: angle=58'h000000000100000;
            6'd34: angle=58'h000000000080000;
            6'd35: angle=58'h000000000040000;
            6'd36: angle=58'h000000000020000;
            6'd37: angle=58'h000000000010000;
            6'd38: angle=58'h000000000008000;
            6'd39: angle=58'h000000000004000;
            6'd40: angle=58'h000000000002000;
            6'd41: angle=58'h000000000001000;
            6'd42: angle=58'h000000000001000;
            6'd43: angle=58'h000000000000800;
            6'd44: angle=58'h000000000000400;
            6'd45: angle=58'h000000000000200;
            6'd46: angle=58'h000000000000100;
            6'd47: angle=58'h000000000000080;
            6'd48: angle=58'h000000000000040;
            6'd49: angle=58'h000000000000020;
            6'd50: angle=58'h000000000000010;
            6'd51: angle=58'h000000000000008;
            6'd52: angle=58'h000000000000004;
            6'd53: angle=58'h000000000000002;
            6'd54: angle=58'h000000000000001;
            default: angle=58'd1;
        endcase end
    endfunction
    // CONSTANTS_END
    reg [3:0] state;
    reg sign_q;
    reg [4:0] convert_shift;
    reg [23:0] input_sig;
    reg signed [57:0] x,y,z,x_shift,y_shift,angle_q;
    reg [5:0] iteration;
    reg [4:0] power;
    reg [57:0] exponential,difference_q,denominator,remainder_q;
    reg [51:0] quotient;
    reg [5:0] digit;
    reg [7:0] output_exp;
    reg [24:0] rounded_significand_q;
    wire [30:0] magnitude = a[30:0];
    // Regular inputs have exponent 114..130 (2^-13 < |a| < 10).
    // 2|a| in Q52 is significand << (exponent-97). Factor out 16
    // constant zero bits and register the remaining 1..17 shift on admission,
    // keeping exponent subtraction out of the 58-bit conversion barrel.
    wire [7:0] conversion_amount = a[30:23] - 8'd113;
    // At each division step remainder < denominator; bit 58 is the borrow.
    wire [58:0] division_difference = {remainder_q,1'b0} - {1'b0,denominator};
    wire round_up = quotient[27] && ((|quotient[26:0]) || quotient[28]);
    wire [24:0] rounded_significand = {1'b0,quotient[51:28]} + {24'b0,round_up};
    wire [7:0] rounded_exponent = output_exp + {7'b0,rounded_significand_q[24]};
    // A synchronous parent can gate its external handshakes and reset its own
    // ownership, allowing internal status to omit the reset combinational path.
    // Standalone interfaces retain immediate reset suppression by default.
    assign ready = state==IDLE && (RESET_GATES_HANDSHAKES==0 || !reset);
    assign response_valid = state==HOLD && (RESET_GATES_HANDSHAKES==0 || !reset);
    always @(posedge clk) begin
        // Payload registers need no reset-enable mux. A payload update on a
        // reset edge is discarded by the final ownership reset below. Keeping
        // reset out of the ROM/coordinate enables shortens integrated timing.
        case(state)
            IDLE: if(valid) begin
                response_tag<=tag; sign_q<=a[31]; invalid<=0;
                if(magnitude>31'h7f800000) begin
                    result<=32'h7fc00000; invalid<=!a[22]; state<=HOLD;
                end else if(magnitude>=31'h41200000) begin
                    result<={a[31],31'h3f800000};state<=HOLD;
                end else if(magnitude<=31'h39000000) begin
                    result<=a;state<=HOLD;
                end else begin
                    convert_shift<=conversion_amount[4:0];input_sig<={1'b1,a[22:0]};state<=CONVERT;
                end
            end
            CONVERT: begin
                z <= $signed({18'b0,input_sig,16'b0} << convert_shift);
                x <= INITIAL; y <= 0; power<=0;iteration<=0;state<=REDUCE;
            end
            REDUCE: begin
                // 2|a| = power*ln(2) + z, 0 <= z < ln(2); power <= 28.
                if(z>=LN2) begin z<=z-LN2;power<=power+1'b1;end
                else state<=SHIFT;
            end
            SHIFT: begin
                x_shift<=x >>> shift_index(iteration);
                y_shift<=y >>> shift_index(iteration);
                angle_q<=angle(iteration);state<=ROTATE;
            end
            ROTATE: begin
                if(!z[57])begin x<=x+y_shift;y<=y+x_shift;z<=z-angle_q;end
                else begin x<=x-y_shift;y<=y-x_shift;z<=z+angle_q;end
                if(iteration==54)state<=DIFFERENCE;
                else begin iteration<=iteration+1'b1;state<=SHIFT;end
            end
            DIFFERENCE: begin difference_q<=x-y;state<=SCALE;end
            SCALE: begin exponential<=difference_q >> power;state<=FRACTION;end
            FRACTION: begin
                // tanh(a) = sign(a)*(1-exp(-2|a|))/(1+exp(-2|a|)).
                remainder_q<=ONE-exponential;denominator<=ONE+exponential;
                quotient<=0;digit<=0;output_exp<=126;state<=DIVIDE;
            end
            DIVIDE: begin
                if(!division_difference[58])begin
                    remainder_q<=division_difference[57:0];quotient<={quotient[50:0],1'b1};
                end else begin remainder_q<=remainder_q<<1;quotient<={quotient[50:0],1'b0};end
                if(digit==51)state<=NORMALIZE;else digit<=digit+1'b1;
            end
            NORMALIZE: begin
                if(quotient[51])state<=ROUND;
                else begin quotient<=quotient<<1;output_exp<=output_exp-1'b1;end
            end
            // Separate nearest/even significand rounding from exponent carry
            // adjustment. One additional clock, with exactly the same bits.
            ROUND: begin rounded_significand_q<=rounded_significand;state<=PACK;end
            PACK: begin
                result<={sign_q,rounded_exponent,rounded_significand_q[22:0]};state<=HOLD;
            end
            HOLD: if(response_ready)state<=IDLE;
            default: state<=IDLE;
        endcase
        if (reset) begin state<=IDLE; invalid<=0; end
    end
    wire unused_round_hidden = rounded_significand_q[23];
    wire unused_conversion_bits = |conversion_amount[7:5];
endmodule
`default_nettype wire
