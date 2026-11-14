#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
test_tmp=$(mktemp -d)
trap 'rm -rf "$test_tmp"' EXIT HUP INT TERM

${CXX:-c++} -std=c++11 -Wall -Wextra -Werror -pedantic \
  "$repo_root/hw/focmini/tests/control_test.cpp" \
  -o "$test_tmp/focmini-control-test"
"$test_tmp/focmini-control-test"
