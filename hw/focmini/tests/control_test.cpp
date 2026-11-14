// Focused host regression tests for the shared line parser and startup interlock.
// Run with ./hw/focmini/tests/run_control_tests.sh from the repository root.
#include "../shared/focmini_control.h"
#include "../shared/motor_test_command.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const char *description) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", description);
    ++failures;
  }
}

std::vector<focmini::Command> feed_bytes(focmini::CommandParser &parser,
                                         const char *bytes, size_t count) {
  std::vector<focmini::Command> commands;
  for (size_t i = 0; i < count; ++i) {
    const focmini::Command command = parser.feed(bytes[i]);
    if (command.type != focmini::CommandType::kNone) commands.push_back(command);
  }
  return commands;
}

std::vector<focmini::Command> feed_text(focmini::CommandParser &parser,
                                        const std::string &text) {
  return feed_bytes(parser, text.data(), text.size());
}

void test_complete_commands_and_line_endings() {
  focmini::CommandParser parser;
  check(feed_text(parser, " ARM \r").size() == 1, "ARM completes on CR");
  check(feed_text(parser, "\nT1.25\r\nSTOP\n").size() == 2,
        "CRLF is one line break and subsequent LF is accepted");

  focmini::CommandParser numeric_parser;
  const auto target = feed_text(numeric_parser, "t -1.25 \n");
  check(target.size() == 1 && target[0].type == focmini::CommandType::kTarget &&
            std::fabs(target[0].target_rad_per_sec + 1.25f) < 1e-6f,
        "signed numeric target and surrounding spaces parse");
}

void test_commands_need_complete_valid_lines() {
  focmini::CommandParser parser;
  check(parser.feed('T').type == focmini::CommandType::kNone &&
            parser.feed('1').type == focmini::CommandType::kNone,
        "incomplete target has no command result");
  const auto invalid = feed_text(parser, "x\n");
  check(invalid.size() == 1 && invalid[0].type == focmini::CommandType::kInvalid,
        "invalid completed line is rejected");

  const char *bad_lines[] = {"T1junk\n", "Tnan\n", "Tinf\n", "T1e999\n",
                             "T1e-999\n", "ARMjunk\n", "T\n"};
  for (const char *line : bad_lines) {
    focmini::CommandParser fresh;
    const auto commands = feed_text(fresh, line);
    check(commands.size() == 1 &&
              commands[0].type == focmini::CommandType::kInvalid,
          "malformed, nonfinite, overflow, and underflow lines are rejected");
  }

  focmini::CommandParser overflow_parser;
  std::string too_long(focmini::CommandParser::kMaxLineBytes + 1, 'A');
  too_long.push_back('\n');
  const auto overflow = feed_text(overflow_parser, too_long);
  check(overflow.size() == 1 &&
            overflow[0].type == focmini::CommandType::kInvalid,
        "overlong line is rejected through its terminator");

  focmini::CommandParser nul_parser;
  const char nul_arm[] = {'A', 'R', 'M', '\0', 'x', '\n'};
  const auto nul_arm_result = feed_bytes(nul_parser, nul_arm, sizeof(nul_arm));
  check(nul_arm_result.size() == 1 &&
            nul_arm_result[0].type == focmini::CommandType::kInvalid,
        "embedded NUL cannot truncate ARM into a valid command");

  focmini::CommandParser nul_target_parser;
  const char nul_target[] = {'T', '1', '\0', 'x', '\n'};
  const auto nul_target_result =
      feed_bytes(nul_target_parser, nul_target, sizeof(nul_target));
  check(nul_target_result.size() == 1 &&
            nul_target_result[0].type == focmini::CommandType::kInvalid,
        "embedded NUL cannot truncate a numeric command");

  focmini::CommandParser control_parser;
  const char control_line[] = {'A', 'R', 'M', '\b', '\n'};
  const auto control_result =
      feed_bytes(control_parser, control_line, sizeof(control_line));
  check(control_result.size() == 1 &&
            control_result[0].type == focmini::CommandType::kInvalid,
        "embedded control byte invalidates the line");
}

void test_motor_test_parser() {
  using C = focmini::MotorTestCommand;
  const std::string bad_lines[] = {
      std::string(16, 'x') + "RUN10", std::string(64, 'x') + "RUN10",
      std::string("RUN10\0junk", 10), "RUN10\b", "RUN10junk",
      std::string("RUN10") + static_cast<char>(0xff)};
  for (const auto &line : bad_lines) {
    focmini::MotorTestParser parser;
    for (char ch : line) {
      check(parser.feed(ch) == C::kNone, "diagnostic waits for line terminator");
    }
    check(parser.feed('\r') == C::kInvalid, "reject whole malformed diagnostic line");
    check(parser.feed('\n') == C::kNone, "diagnostic CRLF emits only one result");
    for (char ch : std::string("stop")) parser.feed(ch);
    check(parser.feed('\n') == C::kStop, "valid STOP works after malformed input");
    for (char ch : std::string("run10")) parser.feed(ch);
    check(parser.feed('\n') == C::kRun10, "valid lowercase RUN10 still works");
  }
  focmini::MotorTestParser parser;
  check(parser.feed('\n') == C::kNone, "empty diagnostic line is ignored");
}

void test_interlock_gating_and_latched_failures() {
  focmini::MotionInterlock motion;
  check(!motion.arm(), "boot state rejects ARM before setup succeeds");
  check(!motion.set_target(1.0f, 5.0f),
        "boot state rejects movement before setup succeeds");
  check(motion.mark_ready(), "successful setup transitions to ready");
  check(motion.arm() && motion.armed() && !motion.active(),
        "ARM unlocks commands but zero target remains inactive");
  check(motion.set_target(1.0f, 5.0f) && motion.active(),
        "armed valid nonzero target becomes active");

  check(!motion.set_target(5.01f, 5.0f) &&
            std::fabs(motion.target_rad_per_sec() - 1.0f) < 1e-6f,
        "out-of-range target preserves the current target");
  check(!motion.set_target(2.0f, 0.0f) &&
            std::fabs(motion.target_rad_per_sec() - 1.0f) < 1e-6f,
        "nonpositive limit rejects without changing target");
  check(!motion.set_target(2.0f,
                           std::numeric_limits<float>::quiet_NaN()) &&
            std::fabs(motion.target_rad_per_sec() - 1.0f) < 1e-6f,
        "nonfinite limit rejects without changing target");

  focmini::CommandParser invalid_target_parser;
  const auto invalid_target = feed_text(invalid_target_parser, "T1junk\n");
  if (invalid_target.size() == 1 &&
      invalid_target[0].type == focmini::CommandType::kTarget) {
    motion.set_target(invalid_target[0].target_rad_per_sec, 5.0f);
  }
  check(std::fabs(motion.target_rad_per_sec() - 1.0f) < 1e-6f,
        "invalid parser result cannot overwrite current target");

  check(motion.set_target(0.0f, 5.0f) && !motion.armed() && !motion.active(),
        "T0 semantics clear target and disarm");
  check(motion.arm() && motion.set_target(-2.0f, 5.0f),
        "explicit re-arm permits a bounded reverse target");
  motion.stop();
  check(!motion.active() && motion.target_rad_per_sec() == 0.0f,
        "STOP clears target and disarms");

  motion.latch_fault();
  check(motion.fault_latched() && !motion.ready() && !motion.arm() &&
            !motion.mark_ready() && !motion.set_target(1.0f, 5.0f),
        "latched hardware or initialization failure requires reboot");
}
}  // namespace

int main() {
  test_motor_test_parser();
  test_complete_commands_and_line_endings();
  test_commands_need_complete_valid_lines();
  test_interlock_gating_and_latched_failures();
  if (failures != 0) {
    std::fprintf(stderr, "%d control regression check(s) failed\n", failures);
    return 1;
  }
  std::puts("focmini shared control regressions passed");
  return 0;
}
