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

 * `rvtst.py`    know what NF5 tests exist, and can determine from the riscvsim output if the test succeeded
 * `dtests.sh`   disassemble all tests
 * `rvtests.sh`  run all tests
 * `dis.sh`      disassemble the test
 * `runtst.sh`   run a single test

For more info, see [NF5-tests/README.md](NF5-tests/README.md)

Most, but not all tests succeed.


# files

 * `riscvsim.py`              the riscv simulator.
 * `riscv_decode_unknown.py`  disasm the titan-m2 custom instructions.
 * `check-dauntless-sigs.py`  decode the titan-m, titan-m2 binaries.

# simulate

Example: calculate the sha1 of 10 zero bytes:

    riscvsim --simulate -l evt.ec.bin@0x80000 -r sp=0x10800 -r gp=0x12000 --loadnames evt.names \
        -m 0x2000=0x14000 -m 0x2004=0xA000 -m 0x2008=1 \
        -m 0x3000=0x2800 -m 0x3004=0x2c00 \
        -f "A_hash_digest(a0=0x1000, a1=0x2000, a2=0x3000)"

| option  | description
| -------- | -------
| `-h,` `--help`           | show this help message and exit
| `--verbose`             
| `--debugdecoder`         | output details of insn decoding

Disassembly

| option  | description
| -------- | -------
| `--disasm,` `-d`         | disassemble.
| `--startaddr OFS`        | disassemble from.
| `--endaddr OFS`          | disassemble until.

Simulation

| option  | description
| -------- | -------
| `--simulate,` `-s`       | run simulator with the given instructions.
| `--trace`                | trace simulation execution, prints all changed memory and register values
| `--breakonscall`         | break when encountering an scall insn.
| `--breakonunimplemented` | break when encountering an unimplemented insn.
| `--breakonstore ADDR`    | break when writing to address.
| `--breakonaddress ADDR`  | break when reaching this address.
| `--breakoncode BYTES`    | break when encountering code bytes.
| `--call fn(arg...)`, `-f fn(arg...)` | call function with args.


Preparing the memory / cpu state

| option  | description
| -------- | -------
| `--hex,` `-x`            | args are hex numbers
| `--bytes,` `-b`          | args are byte lists, implies hex
| `--loadimage filename@OFS`, `-l filename@OFS` | loadimage at offset.
| `--heximage filename`    | load hex(verilogtxt) image.
| `--setreg regname=value`, `-r regname=value` | set register values
| `--setcsreg regname=value`, `-c regname=value` | set csregister values
| `--setmem addr=value`, `-m addr=value` | set memory. either a symbol, single value, range..range of values, or value\*count
| `--sethalf addr=value`   | set memory halfwords. either a symbol, single value, range..range of values, or value\*count
| `--setpriv SETPRIV`      | set initial privilege level
| `--randomize`            | use random value instead of 0 for uninitialized memory.
| `--loadnames LOADNAMES`  | load symbols for the current image

# Author:

Willem Hengeveld
email: itsme@xs4all.nl
github: https://github.com/nlitsme/python-riscv-sim

