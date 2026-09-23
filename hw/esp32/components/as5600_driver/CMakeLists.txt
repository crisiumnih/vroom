# ESP-IDF wrapper for the pinned hardware-independent AS5600 HAL.
# The driver source remains in third_party/as5600_driver; this component only
# exposes it to ESP-IDF projects under hw/esp32.

set(AS5600_DRIVER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/as5600_driver")

idf_component_register(
    SRCS "${AS5600_DRIVER_ROOT}/src/as5600.cpp"
    INCLUDE_DIRS "${AS5600_DRIVER_ROOT}/inc"
)
