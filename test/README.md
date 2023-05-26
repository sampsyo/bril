# tests

- `test/check`: Tests for statically checkable Bril errors across all extensions
- `test/interp/core`: Tests for core Bril
- `test/interp/float`: Tests for the floating point extension
- `test/interp/char`: Tests for the char extension
- `test/interp/mem`: Tests for the memory extension
- `test/interp/mixed`: Tests for programs that mix multiple extensions
- `test/interp/spec`: Tests for the speculation extension
- `test/interp/ssa`: Tests for the ssa extension
- `test/interp-error/core-error`: Tests for errors raised by core Bril
- `test/interp-error/char-error`: Tests for errors raised by the char extension
- `test/interp-error/mem-error`: Tests for errors raised by the memory extension
- `test/interp-error/spec-error`: Tests for errors raised by the speculation extension
- `test/interp-error/ssa-error`: Tests for errors raised by the ssa extension
- `test/linking`: Tests for the import extension
- `test/parse`: Tests for converting Bril text to Bril JSON
- `test/print`: Tests for converting Bril JSON to Bril text
- `test/ts`: Tests for converting Typescript to Bril text
- `test/ts-error`: Tests for errors raised by running Typescript programs as Bril programs
