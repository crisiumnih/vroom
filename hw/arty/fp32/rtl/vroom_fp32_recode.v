`default_nettype none
// Exact binary32 -> HardFloat recFN encoding. Six parallel nibble encoders
// replace the reverse/add/isolate-bit leading-zero circuit. A 6-bit group
// priority chooses a nibble; its local count forms the low two count bits.
// Zero has normalization distance 23, matching fNToRecFN(8,24) exactly.
module vroom_fp32_recode(input wire [31:0] in,output wire [32:0] out);
    wire [23:0] fraction={in[22:0],1'b0};
    wire [5:0] nonzero;
    wire [11:0] local_count;
    genvar g;
    generate for(g=0;g<6;g=g+1)begin: groups
        wire [3:0] nibble=fraction[23-g*4-:4];
        assign nonzero[5-g]=|nibble;
        assign local_count[g*2+:2]=nibble[3] ? 2'd0 : nibble[2] ? 2'd1 : nibble[1] ? 2'd2 : 2'd3;
    end endgenerate
    reg [4:0] distance;
    always @*begin
        casez(nonzero)
            6'b1?????:distance={3'd0,local_count[1:0]};
            6'b01????:distance={3'd1,local_count[3:2]};
            6'b001???:distance={3'd2,local_count[5:4]};
            6'b0001??:distance={3'd3,local_count[7:6]};
            6'b00001?:distance={3'd4,local_count[9:8]};
            6'b000001:distance={3'd5,local_count[11:10]};
            default:distance=23;
        endcase
    end
    wire zero_exp=in[30:23]==0;
    wire zero_fraction=nonzero==0;
    wire [8:0] adjusted_exp=zero_exp ? 9'd129-{4'b0,distance} : {1'b0,in[30:23]}+9'd129;
    wire special=in[30:23]==8'hff;
    wire [8:0] exponent={special ? {2'b11,!zero_fraction} :
                         zero_exp && zero_fraction ? 3'b000 : adjusted_exp[8:6],adjusted_exp[5:0]};
    wire [22:0] normalized=(in[22:0] << distance) << 1;
    assign out={in[31],exponent,zero_exp ? normalized : in[22:0]};
endmodule
`default_nettype wire
