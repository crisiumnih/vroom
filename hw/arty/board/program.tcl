# Volatile JTAG load; preserve the existing SPI flash contents.
if {[catch {
    if {$argc!=2} {error "Usage: -tclargs CABLE_SERIAL BITFILE"}
    lassign $argv serial bitfile
    if {![regexp {^[A-Za-z0-9]+$} $serial] || ![file isfile $bitfile]} {error "Invalid serial or bitfile"}
    open_hw_manager
    connect_hw_server -allow_non_jtag
    set targets [get_hw_targets -quiet *${serial}*]
    if {[llength $targets]!=1} {error "Expected one cable: $targets"}
    current_hw_target [lindex $targets 0]
    open_hw_target
    set devices [get_hw_devices]
    if {[llength $devices]!=1} {error "Expected one FPGA: $devices"}
    set device [lindex $devices 0]
    if {[get_property PART $device] ne "xc7a35t"} {error "Unsupported FPGA: [get_property PART $device]"}
    puts "VROOM_DEVICE target=[current_hw_target] part=[get_property PART $device] idcode=[get_property IDCODE_HEX $device]"
    set_property PROGRAM.FILE [file normalize $bitfile] $device
    program_hw_devices $device
    refresh_hw_device $device
    set signature [get_property REGISTER.USR_ACCESS $device]
    if {![string equal -nocase $signature 56524d32]} {error "Unexpected bitstream signature: $signature"}
    if {![get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $device]} {error "DONE is low"}
    if {[get_property REGISTER.CONFIG_STATUS.BIT00_CRC_ERROR $device]} {error "Configuration CRC error"}
    puts "VROOM_PROGRAM_OK signature=$signature done=1 crc_error=0 storage=volatile"
    close_hw_target
    disconnect_hw_server
    close_hw_manager
} message options]} {
    puts stderr "VROOM_PROGRAM_FAILED: $message"
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
