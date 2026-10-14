/* Simulation-only IEEE binary32 multiply, then add; NOT synthesizable RTL.
 * Two distinct rounding steps, round-to-nearest/even, gradual underflow.
 * Compile with contraction disabled. memcpy preserves all operand bits. */
#include <vpi_user.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <fenv.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "FP32 simulation requires IEEE binary32 float");

static PLI_INT32 result_size(PLI_BYTE8 *data) {
    (void)data;
    return 32;
}

static PLI_INT32 mac_call(PLI_BYTE8 *data) {
    (void)data;
    vpiHandle call = vpi_handle(vpiSysTfCall, 0);
    vpiHandle args = vpi_iterate(vpiArgument, call);
    float operands[3];
    for (int i = 0; i < 3; ++i) {
        vpiHandle arg = vpi_scan(args);
        s_vpi_value value = {.format = vpiVectorVal};
        if (!arg) {
            vpi_printf("ERROR: $vroom_fp32_mac needs three operands\n");
            vpi_control(vpiFinish, 1);
            return 0;
        }
        vpi_get_value(arg, &value);
        if (value.value.vector[0].bval) {
            vpi_printf("ERROR: unknown FP32 operand\n");
            vpi_control(vpiFinish, 1);
            return 0;
        }
        uint32_t bits = (uint32_t)value.value.vector[0].aval;
        memcpy(&operands[i], &bits, sizeof bits);
    }
    vpi_free_object(args);
    if (fesetround(FE_TONEAREST)) {
        vpi_printf("ERROR: cannot set round-to-nearest\n");
        vpi_control(vpiFinish, 1);
        return 0;
    }
    volatile float product = operands[0] * operands[1];
    volatile float result = product + operands[2];
    float rounded = result;
    uint32_t bits;
    memcpy(&bits, &rounded, sizeof bits);
    s_vpi_vecval vector = {.aval = (PLI_INT32)bits, .bval = 0};
    s_vpi_value output = {.format = vpiVectorVal};
    output.value.vector = &vector;
    vpi_put_value(call, &output, 0, vpiNoDelay);
    return 0;
}

static void register_mac(void) {
    s_vpi_systf_data tf = {0};
    tf.type = vpiSysFunc;
    tf.sysfunctype = vpiSizedFunc;
    tf.tfname = "$vroom_fp32_mac";
    tf.calltf = mac_call;
    tf.sizetf = result_size;
    vpi_register_systf(&tf);
}

void (*vlog_startup_routines[])(void) = {register_mac, 0};
