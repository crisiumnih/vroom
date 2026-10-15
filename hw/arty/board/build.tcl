# Run in a fresh ignored evidence directory. Target the physical 35T, not 50T.
if {[catch {
    set board [file dirname [file normalize [info script]]]
    set arty [file dirname $board]
    set repo [file dirname [file dirname $arty]]
    set vendor [file join $repo third_party hardfloat source]
    if {[file exists build]} { error "build/ exists; select a fresh result directory" }
    file mkdir build
    set_param general.maxThreads 2
    create_project -in_memory -part xc7a35ticsg324-1L
    set_property include_dirs [list $vendor [file join $vendor RISCV]] [current_fileset]
    read_verilog [glob [file join $vendor *.v]]
    read_verilog [file join $vendor RISCV HardFloat_specialize.v]
    foreach folder {stream fp32 tanh} {read_verilog [glob [file join $arty $folder rtl *.v]]}
    read_verilog -sv [glob [file join $board *.sv]]
    read_xdc [file join $board arty_a7.xdc]
    synth_design -top vroom_board_top -part xc7a35ticsg324-1L
    foreach port [get_ports] {
        if {[get_property IOSTANDARD $port] ne "LVCMOS33" || [get_property PACKAGE_PIN $port] eq ""} {
            error "Missing physical IO constraint: $port"
        }
    }
    report_utilization -file build/synthesis_utilization.rpt
    opt_design
    place_design
    phys_opt_design
    route_design
    report_timing_summary -check_timing_verbose -file build/timing.rpt
    report_utilization -file build/utilization.rpt
    report_io -file build/io.rpt
    report_drc -file build/drc.rpt
    set setup [get_timing_paths -delay_type max -max_paths 1]
    set hold [get_timing_paths -delay_type min -max_paths 1]
    if {[llength $setup]==0 || [llength $hold]==0} {error "Missing internal timing paths"}
    if {[get_property SLACK $setup]<0 || [get_property SLACK $hold]<0} {error "Timing failed: build/timing.rpt"}
    write_checkpoint build/routed.dcp
    write_bitstream build/vroom_inference.bit
    puts "VROOM_BUILD_OK part=xc7a35ticsg324-1L clock_mhz=100"
} message options]} {
    puts stderr "VROOM_BUILD_FAILED: $message"
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
