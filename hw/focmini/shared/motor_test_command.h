// Line parser for the bounded diagnostic. Invalid/overlong input is discarded
// through CR/LF, so a suffix can never become a motion command. No GPIO access.
#pragma once

#include <cstring>

namespace focmini {
enum class MotorTestCommand { kNone, kRun10, kStop, kInvalid };

class MotorTestParser {
 public:
  MotorTestCommand feed(char ch) {
    if (skip_lf_) {
      skip_lf_ = false;
      if (ch == '\n') return MotorTestCommand::kNone;
    }
    if (ch == '\r' || ch == '\n') {
      MotorTestCommand result = MotorTestCommand::kInvalid;
      if (!invalid_) {
        line_[length_] = '\0';
        if (length_ == 0) result = MotorTestCommand::kNone;
        else if (std::strcmp(line_, "RUN10") == 0) result = MotorTestCommand::kRun10;
        else if (std::strcmp(line_, "STOP") == 0) result = MotorTestCommand::kStop;
      }
      length_ = 0;
      invalid_ = false;
      skip_lf_ = ch == '\r';
      return result;
    }
    if (invalid_) return MotorTestCommand::kNone;
    const unsigned char byte = static_cast<unsigned char>(ch);
    if (byte < 0x20 || byte > 0x7e || length_ >= sizeof(line_) - 1) {
      invalid_ = true;
      return MotorTestCommand::kNone;
    }
    line_[length_++] = ch >= 'a' && ch <= 'z' ? static_cast<char>(ch - 'a' + 'A') : ch;
    return MotorTestCommand::kNone;
  }

 private:
  char line_[16] = {};
  unsigned int length_ = 0;
  bool invalid_ = false;
  bool skip_lf_ = false;
};
}  // namespace focmini
