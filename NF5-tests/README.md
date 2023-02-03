Run risc-v test suite from https://github.com/NUDT-NF5/NF5

These tests depend on this repository:

    git clone https://github.com/NUDT-NF5/NF5

Probably the current master will be good.  I tested this with commit hash: d38d19de8cd7c27

List all available tests:

    python3 rvtst.py -l

Running the tests:

    bash rvtests.sh  > out.txt

Validate the results using:

    python3 rvtst.py out.txt


# NF5

For each test there are 5 different files:
 * a disassembly in objdump format.
 * the expected output in `signature` files.
 * a hexdump of the binary in `verilogtxt` directories.
 * verilog code
 * a `.nm` symbol table.

There are two kinds of tests:
 * function test
   * these succeed with `gp==1`,
   * failure codes are encoded in values > 1 or == 0
 * compliance test
   * these match when the memory at the `begin_signature` symbol matches the contents of the `signature` file.

