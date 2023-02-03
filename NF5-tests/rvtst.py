from collections import defaultdict
import os.path
import re
import struct
from binascii import a2b_hex

# tool for listing NF5 tests, and reading the output of riscvsim to determine if the test succeeded.

class Test:
    """
     * dump    -> objdump
     * output  -> expected output
     * hexdump -> used as binary image
     * verilog 
     * nm      -> names from objdump

    """
    def __init__(self):
        self.name = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.files = defaultdict(set)
    def add(self, tag, path):
        self.files[tag].add(path)
    def get(self, tag):
        s = self.files[tag]
        if s:
            return list(s)[0]
    def __repr__(self):
        return f"{self.name}:{self.files}"


#  NF5/isa-test/isa-functional-test/TVM_name.txt

#  NF5/isa-test/isa-functional-test/test_0x0000/..<tvmname>../testcase_list.txt
#      -> ok when 'gp==1',  bad when 'gp==0x11'
#  NF5/isa-test/isa-compliance-test/test_0x0000/..<tvmname>../..<testcase>..
#      -> ok when <begin_signature> matches

# filenames match these patterns:
# 
#   /dump/(S+)\.(?:obj)?dump
#   /signature/(S+)\.signature\.output
#   /verilogtxt/(S+)
#   /TB_w*/(S+?)(?:-(iverilog|vivado|nc))?\.sv
#   /(S+)\.elf\.nm


def findtests():
    t1 = ""
    t2 = ""
    t3 = ""
    tests = defaultdict(Test)
    for basepath, d, f in os.walk("NF5/isa-test"):
        match os.path.basename(basepath):
            case 'dump':
                for fn in f:
                    if m := re.match(r'(\S+)\.(?:obj)?dump$', fn):
                        tests[(t1, t2, t3, m[1])].add('objdump', os.path.join(basepath, fn))
            case 'signature':
                for fn in f:
                    if m := re.match(r'(\S+)\.signature\.output$', fn):
                        tests[(t1, t2, t3, m[1])].add('output', os.path.join(basepath, fn))
            case 'verilogtxt':
                for fn in f:
                    tests[(t1, t2, t3, fn)].add('hexdump', os.path.join(basepath, fn))
            case other:
                if other.startswith("TB_"):
                    for fn in f:
                        if m := re.match(r'(\S+?)(?:-(iverilog|vivado|nc))?\.sv$', fn):
                            tests[(t1, t2, t3, m[1])].add('verilog', os.path.join(basepath, fn))
                elif other.startswith("isa-"):
                    t1 = other
                elif other.startswith("test_0x"):
                    t2 = other
                elif other.startswith("rv"):
                    t3 = other
                else:
                    for fn in f:
                        if m := re.match(r'^(\S+)\.elf\.nm$', fn):
                            tests[(t1, t2, t3, m[1])].add('nm', os.path.join(basepath, fn))

    for (t1, t2, t3, name), tst in tests.items():
        tst.t1 = t1
        tst.t2 = t2
        tst.t3 = t3
        tst.name = name
    return tests

def readresults(fn):
    dumpfile = None
    memory = []
    gp = None
    import sys
    fh = sys.stdin if fn=='-' else open(fn)
    for line in fh:
        if m := re.match(r'^== (\S+)', line):
            # line with the test name
            if dumpfile:
                yield dumpfile, memory, gp

            dumpfile = m[1]
            memory = []
            gp = None
        elif m := re.match(r'^(\w{8}): ((?: \w\w)+)', line):
            # line with memory dump
            ofs = int(m[1], 16)
            data = a2b_hex(m[2].replace(' ', ''))
            if memory:
                prevofs, prevdata = memory[-1]
                if prevofs+len(prevdata) == ofs:
                    # extend prev entry
                    memory[-1] = (prevofs, prevdata + data)
                    ofs = data = None
            if ofs or data:
                memory.append((ofs, data))

        elif m := re.match(r'^x00:(?:\s+\S+){3}\s+(\S+)', line):
            # lne with register values
            if memory:
                # only capture 2nd gp
                if m[1].startswith('?'):
                    # gp is uninitialized.
                    gp = None
                else:
                    gp = int(m[1], 16)

    if dumpfile:
        yield dumpfile, memory, gp

def readmem(mem, ofs):
    # linear search
    for o, d, in mem:
        if o <= ofs < o+len(d):
            return struct.unpack_from("<L", d, ofs-o)[0]

def findsigofs(fn):
    if not fn:
        return
    for line in open(fn):
        if m := re.match(r'^(\w+) <begin_signature>', line):
            return int(m[1], 16)

def readoutput(fn):
    if not fn:
        return
    for line in open(fn):
        yield int(line.rstrip(), 16)

def verifyresults(test, memory, gp):
    msgs = []
    sig = findsigofs(test.get('objdump'))

    out = list(readoutput(test.get('output')))
    if not out:
        if gp is None:
            msgs.append("gp=None")
        elif gp!=1:
            msgs.append("gp=0x%x" % gp)
        return msgs
    for i, val in enumerate(out):
        m = readmem(memory, sig+i*4)
        if m != val:
            msgs.append("mismatch: %04x = %s, expected %08x" % (sig+i*4, "<none>" if m is None else "%08x" % m, val))
    return msgs

def main():
    import argparse
    parser = argparse.ArgumentParser(description='test result decoder')
    parser.add_argument('--listtests', '-l', action='store_true')
    parser.add_argument('files', type=str, nargs='*')
    args = parser.parse_args()

    tests = findtests()

    if args.listtests:
        for k, t in tests.items():
            print(t.get('hexdump'))
        return

    for fn in args.files:
        for dumpfile, memory, gp in readresults(fn):
            if m := re.search(r'isa-test/(isa-\S*?)/(test_\w+)/(.*?)/verilogtxt/(\S+)', dumpfile):
                t1, t2, t3, tcase = m.groups()
                if t := tests.get((t1, t2, t3, tcase)):
                    msgs = verifyresults(t, memory, gp)
                    print("%-5s - %-10s %s %s %s - %s" % (not msgs, tcase, t1, t2, t3, t.get('objdump')))
                    for m in msgs:
                        print("   ", m)
                else:
                    print("test not found:  - %-10s %s %s %s" % (tcase, t1, t2, t3))
            else:
                print("? unmatched result file")

if __name__=='__main__':
    main()

