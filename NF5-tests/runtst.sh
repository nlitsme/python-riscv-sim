# run a single NF5 test
f=$1
shift
a=0
if [[ $f == *test_0x8000* ]]; then
    a=0x80000000
fi
echo == $f

#  These code sequences all indicate the test program is looping
# 
# +0: 00001f17: auipc      t5, pc, #1___
# +4: fc3f2023: sw         gp, [t5, #-40]
# +8: ff9ff06f: j          0
# 
# +0: 00002f17: auipc      t5, pc, #2___
# +4: fc3f2023: sw         gp, [t5, #-40]
# +8: ff9ff06f: j          0
# 
# +0: 00003f17: auipc      t5, pc, #3___
# +4: fc3f2023: sw         gp, [t5, #-40]
# +8: ff9ff06f: j          0
# 
# +0: 00001f17: auipc      t5, pc, #1___
# +4: fc3f2523: sw         gp, [t5, #-36]
# +8:     bfe5: c.j        00000000
# 
# +0: 00003f17: auipc      t5, pc, #3___
# +4: fc3f2523: sw         gp, [t5, #-36]
# +8:     bfe5: c.j        00000000
# 
# +0: 0000006f: j          0


arg+=(--breakoncode  171f000023203ffc6ff09fff)
arg+=(--breakoncode  172f000023203ffc6ff09fff)
arg+=(--breakoncode  173f000023203ffc6ff09fff)
arg+=(--breakoncode  171f000023253ffce5bf    )
arg+=(--breakoncode  173f000023253ffce5bf    )
arg+=(--breakoncode  6f000000)

arg+=(--breakonunimplemented)

d=$(dirname $f)
d=$(dirname $d)
v=$(echo $d/dump/$(basename $f).*dump)
tohost=$(perl -ne 'if (/#\s+(\w+) <tohost>/) { print $1; exit(0); }' $v)
if [[ $f == *-p-* ]]; then
    arg+=(--breakonstore 0x$tohost)
fi

timeout 10 python3 ../riscvsim.py --simulate "${arg[@]}" --heximage $f -f "$a()"  "$@"
e=$?
if [[ $e != 0 ]]; then
   echo "ERROR: result = $e"
fi

