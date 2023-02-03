Run risc-v test suite from https://github.com/NUDT-NF5/NF5

These tests depend on this repository:

    git clone https://github.com/NUDT-NF5/NF5

Probably the current master will be good.  I tested this with commit hash: d38d19de8cd7c27

Running the tests:

    bash rvtests.sh  > out.txt

Validate the results using:

    python3 rvtst.py out.txt

