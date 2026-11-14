`default_nettype none
// Binary32 gradual-underflow mask, exactly lowMaskLoHi(9,105,130).
// Bit i is set iff exponent < 130-i. Parallel constant comparisons avoid
// the generic primitive's 513-bit signed barrel shift and reversed output.
module vroom_fp32_round_mask(input wire [8:0] exponent,output wire [24:0] mask);
    genvar i;
    generate for(i=0;i<25;i=i+1)begin: bits
        localparam [8:0] THRESHOLD=130-i;
        assign mask[i]=exponent<THRESHOLD;
    end endgenerate
endmodule
`default_nettype wire
