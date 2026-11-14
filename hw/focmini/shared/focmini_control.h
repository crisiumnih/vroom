// Small host-testable command parser and motion interlock shared by the Arduino
// and ESP-IDF adapters. Commands commit only after a complete valid line.
// Numeric targets are mechanical rad/s; hardware adapters enforce their
// configured range and own all GPIO/power-stage operations.
#pragma once

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace focmini {

enum class CommandType { kNone, kArm, kStop, kTarget, kInvalid };

struct Command {
  CommandType type;
  float target_rad_per_sec;

  Command() : type(CommandType::kNone), target_rad_per_sec(0.0f) {}
  Command(CommandType command_type, float target)
      : type(command_type), target_rad_per_sec(target) {}
};

// Bounded, nonblocking line collection. CR, LF, and CRLF are accepted. Once a
// line overflows, its remaining bytes are discarded through its terminator.
class CommandParser {
 public:
  static constexpr unsigned int kMaxLineBytes = 32;

  Command feed(char ch) {
    if (skip_lf_) {
      skip_lf_ = false;
      if (ch == '\n') return Command{};
    }
    if (ch == '\r' || ch == '\n') {
      const Command command = parse_line();
      length_ = 0;
      overflow_ = false;
      invalid_ = false;
      line_[0] = '\0';
      skip_lf_ = ch == '\r';
      return command;
    }
    if (overflow_ || invalid_) return Command{};
    const unsigned char byte = static_cast<unsigned char>(ch);
    if ((byte < 0x20 && ch != '\t') || byte > 0x7e) {
      invalid_ = true;
      return Command{};
    }
    if (length_ >= kMaxLineBytes) {
      overflow_ = true;
      return Command{};
    }
    line_[length_++] = ch;
    line_[length_] = '\0';
    return Command{};
  }

 private:
  static bool space(char ch) { return ch == ' ' || ch == '\t'; }

  static char upper_ascii(char ch) {
    return ch >= 'a' && ch <= 'z' ? static_cast<char>(ch - 'a' + 'A') : ch;
  }

  static bool equals_ascii(const char *left, const char *right) {
    while (*left && *right && upper_ascii(*left) == upper_ascii(*right)) {
      ++left;
      ++right;
    }
    return *left == '\0' && *right == '\0';
  }

  Command parse_line() {
    if (overflow_ || invalid_) return {CommandType::kInvalid, 0.0f};
    char *begin = line_;
    while (space(*begin)) ++begin;
    char *end = begin + std::strlen(begin);
    while (end > begin && space(end[-1])) --end;
    *end = '\0';
    if (*begin == '\0') return {CommandType::kInvalid, 0.0f};

    if (equals_ascii(begin, "ARM")) return {CommandType::kArm, 0.0f};
    if (equals_ascii(begin, "STOP")) return {CommandType::kStop, 0.0f};
    if (upper_ascii(*begin) != 'T') return {CommandType::kInvalid, 0.0f};

    char *number = begin + 1;
    while (space(*number)) ++number;
    if (*number == '\0') return {CommandType::kInvalid, 0.0f};
    char *parsed_end = nullptr;
    errno = 0;
    const float value = std::strtof(number, &parsed_end);
    if (parsed_end == number || errno == ERANGE || !std::isfinite(value)) {
      return {CommandType::kInvalid, 0.0f};
    }
    while (space(*parsed_end)) ++parsed_end;
    if (*parsed_end != '\0') return {CommandType::kInvalid, 0.0f};
    return {CommandType::kTarget, value};
  }

  char line_[kMaxLineBytes + 1] = {};
  unsigned int length_ = 0;
  bool overflow_ = false;
  bool invalid_ = false;
  bool skip_lf_ = false;
};

// Pure software state for startup, command gating, stop, and latched failures.
// The adapters separately drive external EN/SLEEP and trip those outputs in
// their IRAM-safe FAULT handlers. A failed setup/calibration uses the same
// reboot-only latch as a hardware fault.
class MotionInterlock {
 public:
  bool mark_ready() {
    if (fault_latched_) return false;
    ready_ = true;
    return true;
  }

  bool arm() {
    if (!ready_ || fault_latched_) return false;
    armed_ = true;
    target_rad_per_sec_ = 0.0f;
    return true;
  }

  bool set_target(float target_rad_per_sec, float limit_rad_per_sec) {
    if (!ready_ || !armed_ || fault_latched_ ||
        !std::isfinite(limit_rad_per_sec) || limit_rad_per_sec <= 0.0f ||
        !std::isfinite(target_rad_per_sec) ||
        std::fabs(target_rad_per_sec) > limit_rad_per_sec) {
      return false;
    }
    if (target_rad_per_sec == 0.0f) {
      stop();
      return true;
    }
    target_rad_per_sec_ = target_rad_per_sec;
    return true;
  }

  void stop() {
    target_rad_per_sec_ = 0.0f;
    armed_ = false;
  }

  void latch_fault() {
    target_rad_per_sec_ = 0.0f;
    armed_ = false;
    ready_ = false;
    fault_latched_ = true;
  }

  bool ready() const { return ready_ && !fault_latched_; }
  bool armed() const { return armed_ && ready(); }
  bool fault_latched() const { return fault_latched_; }
  float target_rad_per_sec() const { return target_rad_per_sec_; }
  bool active() const {
    return armed() && target_rad_per_sec_ != 0.0f;
  }

 private:
  bool ready_ = false;
  bool armed_ = false;
  bool fault_latched_ = false;
  float target_rad_per_sec_ = 0.0f;
};

}  // namespace focmini
