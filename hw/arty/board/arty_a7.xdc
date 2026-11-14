# Pin facts: Digilent Arty-A7-35-Master.xdc, Rev D/E and original Arty 35T.
# https://github.com/Digilent/digilent-xdc/blob/master/Arty-A7-35-Master.xdc
set_property -dict {PACKAGE_PIN E3 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -name sys_clk -period 10.000 [get_ports clk]
set_property -dict {PACKAGE_PIN A9 IOSTANDARD LVCMOS33} [get_ports uart_rx]
set_property -dict {PACKAGE_PIN D10 IOSTANDARD LVCMOS33 DRIVE 8 SLEW SLOW} [get_ports uart_tx]
# Asynchronous external UART input ends at its two-stage synchronizer.
set_false_path -from [get_ports uart_rx]
set_false_path -to [get_ports uart_tx]
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.USR_ACCESS 0x56524D32 [current_design]
set_property BITSTREAM.CONFIG.UNUSEDPIN Pullnone [current_design]
