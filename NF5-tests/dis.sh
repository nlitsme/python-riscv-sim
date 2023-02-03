# disassemble a single NF5 test"
f=$1
shift
a=0
if [[ $f == *test_0x8000* ]]; then
    a=0x80000000
fi
if [[ -z $2 ]]; then
    e=$[$a+0x400]
else
    e=$[$a+$2]
fi
python3 ../riscvsim.py --disasm --breakonscall --heximage $f --startaddr $a --endaddr $e
e=$?
if [[ $e != 0 ]]; then
   echo "ERROR: result = $e"
fi

