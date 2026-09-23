// ACS714 calibration utility for ESP-IDF 5.5.x.
//
// Purpose:
//   1. Read the ESP32 ADC through the ESP-IDF calibration driver.
//   2. Capture the zero-current offset.
//   3. Capture voltage/current calibration points for a host-side linear fit.
//
// Signal chain assumed by this utility:
//   ACS714 OUT -> 10 kOhm -> ADC node -> 10 kOhm -> GND
//
// The utility does not drive a motor, control a power stage, or persist
// calibration coefficients. It prints machine-readable ZERO and POINT lines;
// tools/hw_bench/fit_acs714.py creates the JSON calibration artifact.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_console.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {

constexpr char kTag[] = "acs714_cal";
constexpr adc_unit_t kAdcUnit = ADC_UNIT_1;
constexpr adc_atten_t kAdcAttenuation = ADC_ATTEN_DB_12;

adc_oneshot_unit_handle_t g_adc_handle = nullptr;
adc_cali_handle_t g_cali_handle = nullptr;
adc_channel_t g_adc_channel = static_cast<adc_channel_t>(CONFIG_ACS714_ADC_CHANNEL);
TaskHandle_t g_monitor_task = nullptr;
volatile bool g_monitor_running = false;

struct Statistics {
    double mean_mv = 0.0;
    double stddev_mv = 0.0;
    int min_mv = 0;
    int max_mv = 0;
    int samples = 0;
};

void update_statistics(Statistics* stats, double value_mv, int sample_index) {
    const double delta = value_mv - stats->mean_mv;
    stats->mean_mv += delta / static_cast<double>(sample_index + 1);
    const double delta2 = value_mv - stats->mean_mv;
    // Store the sum of squares temporarily in stddev_mv.
    stats->stddev_mv += delta * delta2;
    const int rounded_mv = static_cast<int>(std::lround(value_mv));
    if (sample_index == 0) {
        stats->min_mv = rounded_mv;
        stats->max_mv = rounded_mv;
    } else {
        stats->min_mv = (rounded_mv < stats->min_mv) ? rounded_mv : stats->min_mv;
        stats->max_mv = (rounded_mv > stats->max_mv) ? rounded_mv : stats->max_mv;
    }
    stats->samples = sample_index + 1;
}

esp_err_t read_adc_sample(int* raw, int* voltage_mv) {
    esp_err_t err = adc_oneshot_read(g_adc_handle, g_adc_channel, raw);
    if (err != ESP_OK) {
        return err;
    }
    return adc_cali_raw_to_voltage(g_cali_handle, *raw, voltage_mv);
}

Statistics sample_adc() {
    Statistics stats;

    for (int i = 0; i < CONFIG_ACS714_SAMPLE_COUNT; ++i) {
        int raw = 0;
        int voltage_mv = 0;
        ESP_ERROR_CHECK(read_adc_sample(&raw, &voltage_mv));
        update_statistics(&stats, static_cast<double>(voltage_mv), i);

        if (CONFIG_ACS714_SAMPLE_PERIOD_US >= 1000) {
            vTaskDelay(pdMS_TO_TICKS(CONFIG_ACS714_SAMPLE_PERIOD_US / 1000));
        } else if (CONFIG_ACS714_SAMPLE_PERIOD_US > 0) {
            esp_rom_delay_us(CONFIG_ACS714_SAMPLE_PERIOD_US);
        }
    }

    if (stats.samples > 1) {
        stats.stddev_mv = std::sqrt(stats.stddev_mv / static_cast<double>(stats.samples - 1));
    } else {
        stats.stddev_mv = 0.0;
    }
    return stats;
}

void adc_monitor_task(void* /*arg*/) {
    while (g_monitor_running) {
        int raw = 0;
        int voltage_mv = 0;
        const esp_err_t err = read_adc_sample(&raw, &voltage_mv);
        if (err == ESP_OK) {
            std::printf("ADC,%lld,%d,%d\n",
                        static_cast<long long>(esp_timer_get_time()),
                        raw,
                        voltage_mv);
        } else {
            std::printf("ADC_ERROR,%s\n", esp_err_to_name(err));
        }
        std::fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(CONFIG_ACS714_MONITOR_PERIOD_MS));
    }
    g_monitor_task = nullptr;
    vTaskDelete(nullptr);
}

void print_statistics(const char* record_type, double current_a, const Statistics& stats) {
    if (std::strcmp(record_type, "ZERO") == 0) {
        std::printf("ZERO,%.3f,%.3f,%d,%d,%d\n",
                    stats.mean_mv,
                    stats.stddev_mv,
                    stats.min_mv,
                    stats.max_mv,
                    stats.samples);
    } else {
        std::printf("POINT,%.6f,%.3f,%.3f,%d,%d,%d\n",
                    current_a,
                    stats.mean_mv,
                    stats.stddev_mv,
                    stats.min_mv,
                    stats.max_mv,
                    stats.samples);
    }
    std::fflush(stdout);
}

void print_configuration() {
    std::printf("CONFIG,adc_unit=1,adc_channel=%d,attenuation_db=12,samples=%d,period_us=%d,monitor_period_ms=%d\n",
                CONFIG_ACS714_ADC_CHANNEL,
                CONFIG_ACS714_SAMPLE_COUNT,
                CONFIG_ACS714_SAMPLE_PERIOD_US,
                CONFIG_ACS714_MONITOR_PERIOD_MS);
    std::fflush(stdout);
}

int command_zero(int /*argc*/, char** /*argv*/) {
    std::printf("MEASURING,zero\n");
    const Statistics stats = sample_adc();
    print_statistics("ZERO", 0.0, stats);
    return 0;
}

int command_point(int argc, char** argv) {
    if (argc != 2) {
        std::printf("ERROR,usage=p <current_a>\n");
        return 1;
    }

    char* end = nullptr;
    const float current_a = std::strtof(argv[1], &end);
    if (end == argv[1] || *end != '\0' || !std::isfinite(current_a)) {
        std::printf("ERROR,current_a_must_be_finite_number\n");
        return 1;
    }

    std::printf("MEASURING,point,current_a=%.6f\n", current_a);
    const Statistics stats = sample_adc();
    print_statistics("POINT", static_cast<double>(current_a), stats);
    return 0;
}

int command_configuration(int /*argc*/, char** /*argv*/) {
    print_configuration();
    return 0;
}

int command_monitor_start(int /*argc*/, char** /*argv*/) {
    if (g_monitor_task != nullptr) {
        std::printf("ERROR,adc_monitor_already_running\n");
        return 1;
    }
    g_monitor_running = true;
    const BaseType_t created = xTaskCreate(
        adc_monitor_task,
        "adc_monitor",
        3072,
        nullptr,
        3,
        &g_monitor_task);
    if (created != pdPASS) {
        g_monitor_running = false;
        g_monitor_task = nullptr;
        std::printf("ERROR,adc_monitor_task_create_failed\n");
        return 1;
    }
    std::printf("ADC_MONITOR,started\n");
    return 0;
}

int command_monitor_stop(int /*argc*/, char** /*argv*/) {
    g_monitor_running = false;
    std::printf("ADC_MONITOR,stopping\n");
    return 0;
}

void init_adc() {
    adc_oneshot_unit_init_cfg_t unit_config{};
    unit_config.unit_id = kAdcUnit;
    unit_config.ulp_mode = ADC_ULP_MODE_DISABLE;
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_config, &g_adc_handle));

    adc_oneshot_chan_cfg_t channel_config{};
    channel_config.bitwidth = ADC_BITWIDTH_DEFAULT;
    channel_config.atten = kAdcAttenuation;
    ESP_ERROR_CHECK(adc_oneshot_config_channel(g_adc_handle, g_adc_channel, &channel_config));

    adc_cali_line_fitting_config_t cali_config{};
    cali_config.unit_id = kAdcUnit;
    cali_config.atten = kAdcAttenuation;
    cali_config.bitwidth = ADC_BITWIDTH_DEFAULT;
    ESP_ERROR_CHECK(adc_cali_create_scheme_line_fitting(&cali_config, &g_cali_handle));
}

}  // namespace

extern "C" void app_main(void) {
    init_adc();

    ESP_LOGI(kTag, "ACS714 calibration utility ready");
    std::printf("READY,acs714_calibration\n");
    print_configuration();

    ESP_ERROR_CHECK(esp_console_register_help_command());

    esp_console_cmd_t zero_command{};
    zero_command.command = "z";
    zero_command.help = "capture zero-current voltage";
    zero_command.func = &command_zero;

    esp_console_cmd_t point_command{};
    point_command.command = "p";
    point_command.help = "capture a point: p <current_a>";
    point_command.hint = "<current_a>";
    point_command.func = &command_point;

    esp_console_cmd_t configuration_command{};
    configuration_command.command = "v";
    configuration_command.help = "print ADC configuration";
    configuration_command.func = &command_configuration;

    esp_console_cmd_t monitor_start_command{};
    monitor_start_command.command = "m";
    monitor_start_command.help = "start continuous ADC monitor";
    monitor_start_command.func = &command_monitor_start;

    esp_console_cmd_t monitor_stop_command{};
    monitor_stop_command.command = "x";
    monitor_stop_command.help = "stop continuous ADC monitor";
    monitor_stop_command.func = &command_monitor_stop;

    ESP_ERROR_CHECK(esp_console_cmd_register(&zero_command));
    ESP_ERROR_CHECK(esp_console_cmd_register(&point_command));
    ESP_ERROR_CHECK(esp_console_cmd_register(&configuration_command));
    ESP_ERROR_CHECK(esp_console_cmd_register(&monitor_start_command));
    ESP_ERROR_CHECK(esp_console_cmd_register(&monitor_stop_command));

    esp_console_repl_t* repl = nullptr;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = "acs714> ";
    repl_config.max_cmdline_length = 64;

#if defined(CONFIG_ESP_CONSOLE_UART_DEFAULT) || defined(CONFIG_ESP_CONSOLE_UART_CUSTOM)
    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_config, &repl_config, &repl));
#elif defined(CONFIG_ESP_CONSOLE_USB_CDC)
    esp_console_dev_usb_cdc_config_t usb_config = ESP_CONSOLE_DEV_CDC_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_cdc(&usb_config, &repl_config, &repl));
#elif defined(CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG)
    esp_console_dev_usb_serial_jtag_config_t usb_config = ESP_CONSOLE_DEV_USB_SERIAL_JTAG_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_usb_serial_jtag(&usb_config, &repl_config, &repl));
#else
#error Unsupported ESP-IDF console configuration
#endif

    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
