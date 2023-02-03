# riscvsim.py

A python Risc-V cpu simulator.

Supports RV32I, RV32C, RV32M, RV32S

No support for atomics, floating point, vector. rv128, rv64, crypto

# Dependencies

For the simulator you will need to clone these two repositories.

    git clone https://github.com/riscv/riscv-opcodes
    git clone https://github.com/riscv/riscv-crypto

Known to work with commit hashes 902fa8a1b66e9cb6 and 854648324a4dfbc4 respectively.


# test suite

The scripts in the `NF5-tests` directory can be used to run a risc-v instruction level test suite.

 * `rvtst.py`  know what NF5 tests exist, and can determine from the riscvsim output if the test succeeded
 * `dtests.sh`  disassemble all tests
 * `rvtests.sh`  run all tests
 * `dis.sh`    disassemble the test
 * `tst.sh`      run a single test


# files

 * `riscvsim.py`              the riscv simulator.
 * `riscv_decode_unknown.py`  disasm the titan-m2 custom instructions.
 * `check-dauntless-sigs.py`  decode the titan-m, titan-m2 binaries.
