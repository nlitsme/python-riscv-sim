#!/usr/bin/python3
# a risc-v disassembler ans simulator. using the definitions from the riscv-opcodes repository.
import re
import os
import os.path
from binascii import a2b_hex
import struct
import random

# TODO:
#    -DONE add support for multiple fuction calls
#    -DONE add support for function calls to include args
#      specified as: (a0, a1, ...)

#    - DONE improve indent algorithm - correlate with stackpointer.
#    - add i/o handlers.

# TODO: make misa.S configurable
#      - NF5 tests need S mode
#      - Titan-M2 runs without S mode

# TODO: support irq calls  -> note TODO in trap-handler.
# TODO: extract code for devices from mem.store, mem.load
# TODO: add fast (in-python) implementations for certains calls, like memset, memcpu, sha256, modexp
# TODO: implement pmp memory protection
# TODO: merge disassemble and simulate functions
# todo: add special handlers for certain memory addresses
# todo: add special handlers for certain subroutines.


# register names:
#  compressed:
#      0     1     2     3     4     5     6     7
#     x8    x9   x10   x11   x12   x13   x14   x15
#  (f)s0 (f)s1 (f)a0 (f)a1 (f)a2 (f)a3 (f)a4 (f)a5

# conventions:
#     x0  = zero
#     x1  = ra    : return address
#     x2  = sp    : stack pointer
#     x3  = gp    : global pointer
#     x4  = tp    : thread pointer
#     x5  =       : alternate link register
#  x5-x7  = t0-t2 : temporary registers: not preserved
#  x8-x9  = s0-s1 : callee saved regs
# x10-x17 = a0-a7 : argument regs
# x18-x27 = s2-s11 : callee saved regs
# x28-x31 = t3-t6  : temporrary regs: not preserved

#  or more compact:
#       0  1  2  3  4  5  6  7
#  00: 00 ra sp gp tp t0 t1 t2
#  08: s0 s1 a0 a1 a2 a3 a4 a5
#  10: a6 a7 s2 s3 s4 s5 s6 s7
#  18: s8 s9 sA sB t3 t4 t5 t6

# floating point regs
#  f0-f7  = ft0-ft7
#  f8-f9  = fs0-fs1
# f10-f17 = fa0-fa7
# f18-f27 = fs2-fs11
# f28-f31 = ft8-ft11

# note that 0x7c0-0x7ff  is listed as 'Machine-Level CSR, Custom R/W'
# note that 0x800-0x8ff  is listed as 'Unprivileged, User-Level CSR, Custom R/W'

#  0300 mstatus - machine status register
#      1 - SIE    S-mode interrupt enable
#      3 - MIE    M-mode interrupt enable
#      5 - SPIE   S-mode privileged interrupt enable
#      6 - UBE    U-mode big-endian memory reads
#      7 - MPIE   M-mode privileged interrupt enable
#      8 - SPP    S-mode previous privilege mode
#      10,9  - VS   Vector-extension unit status (0 = off, 1 = initial, 2 = clean, 3 = dirty)
#      12,11 - MPP  M-mode previous privilege mode
#      14,13 - FS   Floating-Point unit status (0 = off, 1 = initial, 2 = clean, 3 = dirty)
#      16,15 - XS   Usermode-extension unit status (0 = off, 1 = some on, 2 = some clean, 3 = some dirty)
#      17 - MPRV   Modify Privilege
#      18 - SUM    permit supervisor user memory access
#      19 - MXR    Make executable readable
#      20 - TVM    Trap Virtual Memory
#      21 - TW     Timeout Wait -> WFI has timeout in lower privilege levels
#      22 - TSR    Trap SRET ( supervisor return )
#      31 - SD     any of VS|FS|XS dirty
#  0304 mie     - machine interrupt enable register
#  0305 mtvec   - machine trap handler base address
#  0340 mscratch- scratch reg or machine trap handlers
#  0341 mepc    - machine exception program counter
#  0342 mcause  - machine trap cause
#  0343 mtval   - machine bad address or instruction
#  0344 mip     - machine interrupt pending
#  0b00 mcycle  - machine cycle counter

# exceptions
#  *  E_Fetch_Addr_Align()   => 0,  # "misaligned-fetch",
#  *  E_Fetch_Access_Fault() => 1,  # "fetch-access-fault",
#  *  E_Illegal_Instr()      => 2,  # "illegal-instruction",
#  *  E_Breakpoint()         => 3,  # "breakpoint",
#  *  E_Load_Addr_Align()    => 4,  # "misaligned-load",
#  *  E_Load_Access_Fault()  => 5,  # "load-access-fault",
#  *  E_SAMO_Addr_Align()    => 6,  # "misaliged-store/amo",
#  *  E_SAMO_Access_Fault()  => 7,  # "store/amo-access-fault",
#  *  E_U_EnvCall()          => 8,  # "u-call",
#  *  E_S_EnvCall()          => 9,  # "s-call",
#  *  E_Reserved_10()        => 10, # "reserved-0",
#  *  E_M_EnvCall()          => 11, # "m-call",
#  *  E_Fetch_Page_Fault()   => 12, # "fetch-page-fault",
#  *  E_Load_Page_Fault()    => 13, # "load-page-fault",
#  *  E_Reserved_14()        => 14, # "reserved-1",
#  *  E_SAMO_Page_Fault()    => 15, # "store/amo-page-fault",

# bits are in medeleg register

# interrupt type:
#  * I_U_Software => 0x00,  # USI
#  * I_S_Software => 0x01,  # SSI
#  * I_M_Software => 0x03,  # MSI
#  * I_U_Timer    => 0x04,  # UTI
#  * I_S_Timer    => 0x05,  # STI
#  * I_M_Timer    => 0x07,  # MTI
#  * I_U_External => 0x08,  # UEI
#  * I_S_External => 0x09,  # SEI
#  * I_M_External => 0x0b   # MEI

#  bits are in  mip, mie, mideleg registers



# pmp
# cfgbits:  L 00 AA X W R
#   L = locked: readonly until reset
#   X = execute perm    -> E_Fetch_Access_Fault
#   W = write perm      -> E_SAMO_Access_Fault
#   R = read perm       -> E_Load_Access_Fault
#  AA = matchtype:
#   00  -> none
#   01  -> TOR    : pmpaddr[-1] <= addr < pmpaddr
#   10  -> NA4    : pmpaddr <= addr < pmpaddr+4
#   11  -> NAPOT  : pmpaddr <= addr < pmpaddr+size
# pmpaddr are stored >>2
# pmpaddr ends in 01...1  -> this is te NAPOT-size >>3 -1
#   
# arch:  01=rv32, 10=rv64, 11=rv128
# mode:  00=user, 01=super, 11=machine


# instruction patterns:
#                ..aa  -> 16-bit, aa!=11
#            ...bbb11  -> 32-bit, bb!=111
#        ......011111  -> 48-bit
#    .........0111111  -> 64-bit
#  **nnn......1111111  -> 80+16*nnn -bit, nnn!=111
# ***nnn......1111111  -> >192-bit

#base instruction types:
#  1098765|43210  |98765  |432    |10987   |6543210
# |31   25|24   20|19   15|14   12|11     7|6      0|
# +-------+-------+-------+-------+--------+--------+
# | func7 |  rs2  |  rs1  | func3 | rd     | opcode |  R-type
# | ----imm12---- |  rs1  | func3 | rd     | opcode |  I-type
# | immh12|  rs2  |  rs1  | func3 | imml12 | opcode |  S-type + B-type
# | -------------imm20----------- | rd     | opcode |  U-type + J-type

# major opcode groups:
#
#bit65\ 0      1         2        3          4       5          6          7   <-bits4-2
#  0    LOAD   LOAD-FP  custom-0  MISC-MEM  OP-IMM  AUIPC     OP-IMM-32      48b
#  1    STORE  STORE-FP custom-1  AMO       OP      LUI       OP-32          64b
#  2    MADD   MSUB     NMSUB     NMADD     OP-FP   OP-V      cust2/ru128    48b
#  3    BRANCH JALR     reserved  JAL       SYSTEM  reserved  cust3/ru128    >80b


def undef(x):
    return "?"*8 if x is None else f"{x:08x}"


def maskbits(n):
    """ return a value with the lowest 'n' bits set to one. """
    return (1<<n)-1


def reorderbits(value, *bitspec):
    """
    bitspec specifies where a bit from that position should be copied to, in big-endian order

    This function is used to decode values using the bit-order as specified in the riscv-spec.
    """
    # in imm20 order:
    #    20 10-1 11 19-12 -> ofs bits
    #    19 18-9 8   7-0  -> imm20 bits
    # in ofs order:
    #    20 19-12 11 10-1 -> ofs bits
    #    19  7-0  8  18-9 -> imm20 bits

    result = 0
    for b in bitspec[::-1]:
        if type(b) == tuple:
            h, l = b
            bits = value & ((1<<(h-l+1))-1)
            value >>= h-l+1
            result |= bits<<l
        else:
            bit = value&1
            value >>= 1
            result |= bit<<b

    return result


def signed(val, bits):
    """
    Sign-extend the value from 'bits'

    example:
       0  1  2  3  4  5  6  7  val
       0  1  2  3 -4 -3 -2 -1  signed(val, 3)
    """
    val &= maskbits(bits)
    if val >= 1<<(bits-1):
        return val - (1<<bits)
    return val


def unsigned(val, bits=32):
    """
    converts 'val' to a 32-bit unsigned value
    """
    return val & maskbits(bits)


class NamedBitFields:
    """
    Create named bitfields.
    The specification is passed using keyword arguments, where an integer
    value indicates a single bit, and a tuple indicates an inclusive range of bits.

    example spec:
        mstatus = NamedBitFields(SPP=8, MPP=(12,11), FS=(14,13), XS=(16,15))
        mstatus.MPP = 1

    access to all 32 bits using the `bits()` method.
    """
    def __init__(self, **spec):
        self._spec = spec
        self._value = 0

    def bits(self): return self._value
    def setbits(self, value): self._value = value

    def __getattr__(self, name):
        match bits := self._spec.get(name):
            case (hi, lo):
                return self._value>>lo & maskbits(hi-lo+1)
            case int(bit):
                return self._value>>bit & 1
            case None:
                raise AttributeError()

    def __setattr__(self, name, value):
        if name.startswith('_'):
            return super().__setattr__(name, value)
        match bits := self._spec.get(name):
            case (hi, lo):
                m = maskbits(hi-lo+1)
                self._value &= ~(m<<lo)
                self._value |= (value&m)<<lo
            case int(bit):
                self._value &= ~(1<<bit)
                self._value |= (value&1)<<bit
            case None:
                return super().__setattr__(name, value)

    def __repr__(self):
        txt = ""
        for k in self._spec.keys():
            txt += " %s=%d" % (k, getattr(self, k))
        return txt


class Instruction:
    """
    Represent a decoded instruction.
    This is the object returned by InstructionPattern.decode()
    """
    def __init__(self, opc, mnemonic, args):
        self.opc = opc
        self.mnemonic = mnemonic
        self.args = dict(args)
    def __str__(self):
        argstr = ", ".join("%s=0x%x" % (k, v) for k, v in self.args.items())
        return f"{self.mnemonic}\t{argstr}"
    def __getattr__(self, name):
        if name in self.args:
            return self.args[name]
        print("known attr: ", self.args)
        raise AttributeError()


class InstructionPattern:
    """
    Represents a instruction pattern from the riscv-opcodes repository
    """
    def __init__(self, line, decoder):
        self.decoder = decoder

        self.insnname = None
        self.comment = None
        self.ignoremask = 0
        self.valuemask = 0
        self.value = 0
        self.namedmask = 0
        self.named = []

        self.parse(line)

    def parse(self, line):
        if m := re.match(r'''^(?:\$pseudo_op\s+\S+\s+)?(\S+)\s+(\S.*?\S)\s*(?:#\s*(.*))?$''', line):
            self.insnname = m.group(1)
            fieldspec = m.group(2)
            self.comment = m.group(3)
            for spec in re.split(r'\s+', fieldspec):
                if m := re.match(r'^(\d+)(?:\.\.(\d+))?=(\d\w*|ignore)$', spec):
                    #  hi..lo=value
                    b1 = int(m.group(1), 0)
                    b0 = int(m.group(2), 0) if m.group(2) else b1
                    val = int(m.group(3), 0) if m.group(3) != 'ignore' else None

                    #print(spec, "->", b1, b0, val)

                    newmask = maskbits(b1-b0+1)<<b0
                    if self.allmasks() & newmask:
                        print("WARN: bits already masked: %s" % spec)
                    if val is not None:
                        self.valuemask |= newmask
                        self.value |= val<<b0
                    else:
                        self.ignoremask |= newmask

                elif spec in self.decoder.namedbitfields:
                    # names like: rs1, rd, etc.
                    self.named.append(spec)
                    b1, b0 = self.decoder.namedbitfields[spec]
                    #print(spec, "->", b1, b0)
                    newmask = maskbits(b1-b0+1)<<b0

                    if self.allmasks() & newmask:
                        print("WARN: bits already maked: %s" % spec)
                    self.namedmask |= newmask
                    # print("spec is named: %s(%d,%d) -> %x" % (spec, b1, b0, newmask))
                elif self.decoder.args.verbose:
                    """
                    note: currently one unhandled case:
                    riscv-crypto/tools/opcodes-crypto-scalar-rv64
                         aes64ks1i  rd rs1 rcon rcon:<=10 31..30=0 29..25=0b11000 24=1     14..12=0b001 6..0=0x13
                    """
                    print("WARN: unknown spec: %s" % spec)
        else:
            print("WARN: unrecognised opcode line: %s" % line)

        if self.namedmask & self.valuemask:
            print("WARN: mask overlap")
        elif self.namedmask & self.ignoremask:
            print("WARN: mask overlap")
        elif self.ignoremask & self.valuemask:
            print("WARN: mask overlap")

        allbits = self.allmasks()
        if allbits > 0x10000:
            allbits ^= maskbits(32)
        else:
            allbits ^= maskbits(16)
        if allbits:
            print("WARN: unused bits: %8x - %s" % (allbits, line))

    def allmasks(self):
        return self.namedmask | self.valuemask | self.ignoremask

    def matches(self, opc):
        if opc & ~ self.allmasks():
            return False
        return (opc & self.valuemask) == self.value

    def decode(self, opc):
        info = []
        for f in self.named:
            b1, b0 = self.decoder.namedbitfields[f]
            mask = maskbits(b1-b0+1)<<b0
            info.append((f, (opc&mask)>>b0))

        return Instruction(opc, self.insnname, info)


class InstructionDecoder:
    """
    Reads instruction definitions from riscv-opcodes, and uses that information to decode
    opcodes.
    """
    def __init__(self, args):
        self.args = args
        self.instructions = []
        self.namedbitfields = dict()
        self.csr_regs = dict()

        self.load()

    def load(self):
        basepath = os.path.dirname(os.path.realpath(__file__))
        self.loadconstants(os.path.join(basepath, "riscv-opcodes/constants.py"))
        self.loadconstants(os.path.join(basepath, "riscv-crypto/bin/parse_opcodes.py"))
        for opcfilename, sectionname in self.enumopcodefiles():
            with open(opcfilename, "r") as fh:
                for op in self.readopcodefile(fh):
                    op.srcfile = sectionname
                    self.instructions.append(op)

        for spec in self.google_custom_insn():
            self.instructions.append(InstructionPattern(spec, self))

    def google_custom_insn(self):
        #                         opcode      f3       imm12
        yield "gbswap32    rd rs1 6..0=0x0b 14..12=0 31..20=0x018"
        yield "grbitscan   rd rs1 6..0=0x0b 14..12=2 31..20=0x400"
        yield "gbitscan    rd rs1 6..0=0x0b 14..12=2 31..20=0x000"
        yield "gclrbit     rd rs1 rs2  6..0=0x2b 14..12=1 31..25=0x00"
        yield "gsetbit     rd rs1 rs2  6..0=0x2b 14..12=1 31..25=0x20"
        yield "gclrbiti    rd rs1 shamt 6..0=0x0b 14..12=1 31..27=0x00"
        yield "gsetbiti    rd rs1 shamt 6..0=0x0b 14..12=1 31..27=0x08"

        yield "illegal     15..0=0"

    def loadconstants(self, fn):
        # loads constants used in the instruction definitions.
        for line in open(fn):
            if m := re.match(r"^\s*\(\s*(0x\w\w\w)\s*,\s*'(\S+)'\),", line):
                regnum = int(m[1], 0)
                regname = m[2]
                if regnum in self.csr_regs:
                    # note: this happens for: 
                    #  csr07b2: dscratch0 / dscratch
                    #  csr0f15: mconfigptr / mentropy
                    if self.csr_regs[regnum] != regname:
                        if self.args.verbose:
                            print("WARNING: multiple names for csr%04x: %s / %s" % (regnum, self.csr_regs[regnum], regname))
                else:
                    self.csr_regs[regnum] = regname

            elif m := re.match(r"arg_?lut\['(\S+)'\s*\] *= *\(\s*(\d+),\s*(\d+)\s*\)", line):
                fieldname = m[1]
                highbit = int(m[2])
                lowbit = int(m[3])
                if fieldname in self.namedbitfields:
                    if self.namedbitfields[fieldname] != (highbit, lowbit):
                        if self.args.verbose:
                            # note: inconsistency in defs for main/cryptp: shamt -> main: (26, 20) / crypto: (25, 20)
                            print("WARNING: incompatible field def: %s -> %s / %s" % (fieldname, self.namedbitfields[fieldname], (highbit, lowbit)))
                else:
                    self.namedbitfields[fieldname] = (highbit, lowbit)

    def readopcodefile(self, fh):
        while True:
            line = fh.readline()
            if not line:
                # eof
                break
            line = line.rstrip("\n")
            if m := re.match(r'^\s*#', line):
                # skip comments
                pass
            elif m := re.match(r'''^\$import \S+''', line):
                pass 
            elif not line:
                # skip empty
                pass
            elif line.startswith("c.addiw "):
                #  one exception: c.addiw is rv64+rv128, while c.jal is rv32 for the same opcode.
                pass
            elif line.startswith("$"):
                pass
            else:
                yield InstructionPattern(line, self)

    def enumopcodefiles(self):
        basepath = os.path.dirname(os.path.realpath(__file__))
        for opcdir in ("riscv-opcodes", "riscv-crypto/tools"):
            for path, dirs, files in os.walk(os.path.join(basepath, opcdir)):
                if ".git" in dirs:
                    dirs.remove(".git")
                for fn in files:
                    if fn.find(".") >= 0: continue
                    if fn in ("Makefile", "LICENSE" ,"Gemfile"): continue

                    if m := re.match(r'opcodes-(\S+)', fn):
                        yield os.path.join(path, fn), m.group(1)
                    elif m := re.match(r'rv(\S+)', fn):
                        yield os.path.join(path, fn), m.group(1)
                    else:
                        print("WARN: unmatched filename: ", fn)

    def decodeinsn(self, opc):
        """
        searches the instruction table for a matching opcode.

        returns an Instruction object when found.
        returns None when the instruction is unknown.
        """
        matches = []
        for insn in self.instructions:
            if insn.matches(opc):

                # make sure the match with most fixed bits is found
                # for example:
                # riscv-opcodes/rv_c
                #   c.addi16sp c_nzimm10hi c_nzimm10lo       1..0=1 15..13=3 11..7=2
                #   c.lui rd_n2 c_nzimm18hi c_nzimm18lo      1..0=1 15..13=3

                # n=0000107c v=0000ef83
                # n=00001ffc v=0000e003

                # other problem: cpu specific insn
                # c.jal c_imm12                           1..0=1 15..13=1   riscv-opcodes/rv32_c
                # c.addiw rd_rs1 c_imm6lo c_imm6hi        1..0=1 15..13=1   riscv-opcodes/rv64_c + riscv-opcodes/unratified/rv128_c

                matches.append((insn.namedmask, insn.valuemask, insn.ignoremask, insn))

        matches = sorted(matches, key=lambda m:m[1])
        if matches:
            return matches[-1][-1].decode(opc)

    def analyzeopcodes(self, opcodelist):
        """
        print decoding information for the given opcodes.
        """
        for opcbytes in opcodelist:
            opc = int.from_bytes(opcbytes, 'little')
            matches = []
            for insn in self.instructions:
                if insn.matches(opc):

                    # make sure the match with most fixed bits is found
                    # for example:
                    # riscv-opcodes/rv_c
                    #   c.addi16sp c_nzimm10hi c_nzimm10lo       1..0=1 15..13=3 11..7=2
                    #   c.lui rd_n2 c_nzimm18hi c_nzimm18lo      1..0=1 15..13=3

                    # n=0000107c v=0000ef83
                    # n=00001ffc v=0000e003

                    # other problem: cpu specific insn
                    # c.jal c_imm12                           1..0=1 15..13=1   riscv-opcodes/rv32_c
                    # c.addiw rd_rs1 c_imm6lo c_imm6hi        1..0=1 15..13=1   riscv-opcodes/rv64_c + riscv-opcodes/unratified/rv128_c

                    matches.append((insn.namedmask, insn.valuemask, insn.ignoremask, insn))

            matches = sorted(matches, key=lambda m:m[1])
            if not matches:
                print("%08x: WARN: no matches for opcode" % (opc))
            #lif args.debugdecoder:
            #   for a,b,c, m in matches:
            #       print("%08x: [%08x:%08x:%08x] -> %s" % (opc, a,b,c, m.decode(opc)))
            else:
                insn = matches[-1][-1].decode(opc)
                print("%08x: %s" % (opc, insn))


class PRIV:
    """
    Named constants for the privilege levels.
    """
    USER   = 0
    SUPER  = 1
    MACHINE= 3
    @staticmethod
    def name(x):
        match x:
            case PRIV.USER:
                return "user"
            case PRIV.SUPER:
                return "super"
            case PRIV.MACHINE:
                return "machine"

    @staticmethod
    def decode(x):
        match x:
            case 'user': return PRIV.USER
            case 'super': return PRIV.SUPER
            case 'machine': return PRIV.MACHINE


class CPU:
    """
    Keep state for the risc-v cpu.
    """

    # class dicts.
    _csrnames = { }
    _csrnums = { }
    _regnames = [ "zero", "ra", "sp", "gp", "tp" ]
    _rnums = { }

    def __init__(self, args):
        self.args = args
        if not self._csrnums:
            # init class dict on first use.
            for k, v in self._csrnames.items():
                self._csrnums[v] = k
        if not self._rnums:
            # init class dict on first use.
            for rnum in range(32):
                self._rnums[self.regname(rnum)] = rnum

        self.triggers = dict()
        self.pc = 0
        self.nextpc = 0
        self._regs = [None] * 32
        self._regs[0] = 0   # zero reg.
        self._csregs = [None] * 4096
        self._curpriv = PRIV.MACHINE

        self.mstatus = NamedBitFields(UIE=0, SIE=1, MIE=3, UPIE=4, SPIE=5, MPIE=7,
                                      SPP=8, MPP=(12,11), FS=(14,13), XS=(16,15),
                                      MPRV=17, SUM=18, MXR=19, TVM=20, TW=21, TSR=22)

        bitnames = { chr(65+i):i for i in range(26) }
        self.misa = NamedBitFields(**bitnames, MXL=(31,30))
        ARCH_32 = 1
        ARCH_64 = 2
        ARCH_128 = 3
        self.misa.MXL = ARCH_32
        self.misa.C = 1 # C-extension
        self.misa.E = 1 # RV32E base ISA
        self.misa.I = 1 # RV32I base ISA
        self.misa.M = 1 # have mul/div
        self.misa.S = 1 # have supervisor mode
        self.misa.U = 1 # have user mode
        # misa.A - atomics
        # misa.F - float
        # misa.D - double

        # reset values
        self.mstatus.MIE = 0
        self.mstatus.MPRV = 0
        self.mcause = 0
        self.mcycle = 0
        self.medeleg = 0   # initially 0, optionally support delegate to priv=super mode.
        self.mhartid = 0   # always 0, since we are emulating a single core cpu.
        self.csr7c4 = 0   # Titan-M2 specific
        self.pc = 0

    def __getattr__(self, name):
        """
        Convenience accessors, so I can access registers and CSRs by name.
        """
        if name in self._csrnums:
            return self.getcsreg(self._csrnums[name])
        elif name in self._rnums:
            return self.reg(self._rnums[name])
        else:
            print("unknown attr: %s" % name)
            raise AttributeError()

    def __setattr__(self, name, value):
        # convenience setters
        if name in self._csrnums:
            self.setcsreg(self._csrnums[name], value)
        elif name in self._rnums:
            self.setreg(self._rnums[name], value)
        else:
            super().__setattr__(name, value)

    def reg(self, num):
        # access reg by number
        if self._regs[num] is None:
            print("read from uninitialized reg %s" % self.regname(num))
        return self._regs[num] or 0

    def setreg(self, num, value):
        # set reg by number
        if num==0:
            print("WARN: write to zero register")
            return
        if self.args.trace:
            print("change reg %s from %s -> %08x" % (self.regname(num), undef(self._regs[num]), value & 0xFFFFFFFF))
        self._regs[num] = value & 0xFFFFFFFF
        if t := self.triggers.get(num):
            t(value & 0xFFFFFFFF)

    def getcsreg(self, num):
        # get csreg object by register number
        if self._csregs[num] is None:
            print("read from uninitialized csreg %s" % self.csregname(num))
        return self._csregs[num] or 0

    def csreg(self, num):
        # get csreg numerical value by register number
        r = self.getcsreg(num)
        if isinstance(r,NamedBitFields):
            return r.bits()
        return r

    def setcsreg(self, num, value):
        # set csreg by number
        r = self._csregs[num]
        if isinstance(r, NamedBitFields):
            if self.args.trace:
                print("change csreg %s from %08x -> %08x" % (self.csregname(num), r.bits(), value))

            r.setbits(value)
        elif isinstance(value, NamedBitFields):
            if self.args.trace:
                print("change csreg %s to a bitfield" % (self.csregname(num)))
            # this is where the register is converted to NamedBitFields
            self._csregs[num] = value
        else:
            if self.args.trace:
                print("change csreg %s from %s -> %08x" % (self.csregname(num), undef(r), value))
            self._csregs[num] = value & 0xFFFFFFFF

    def priv(self):
        return self._curpriv

    def setpriv(self, p):
        print("change priv from %s to %s" % (PRIV.name(self._curpriv), PRIV.name(p)))
        self._curpriv = p

    def dump(self):
        # dump current state
        print("pc = %08x" % self.pc)
        for i, r in enumerate(self._regs):
            if (i%8)==0:
                print("x%02d:" % i, end="")

            print(" %s" % undef(r), end="")

            if (i%8)==7:
                match i//8:
                    case 0: print(" - zero, ra, sp, gp, tp, t0-t2")
                    case 1: print(" - s0, s1, a0-a5")
                    case 2: print(" - a6, a7, s2-s7")
                    case 3: print(" - s8-s11, t3-t6")

    def translate_regname(self, name):
        """ name to register number """
        if m := re.match(r'^x(\d+)', name):
            return int(m[1])
        elif m := re.match(r'^t(\d+)', name):
            t = int(m[1])
            if t<3: return t+5
            else:   return t+25
        elif m := re.match(r'^s(\d+)', name):
            s = int(m[1])
            if s<2: return s+8
            else:   return s+16
        elif m := re.match(r'^a(\d+)', name):
            a = int(m[1])
            return a + 10
        if name in self._rnums:
            return self._rnums[name]
        raise Exception("invalid regname")

    def regname(self, num):
        """ regnum to name """
        if num < 0:
            raise Exception("invalid regnum")
        if num < len(self._regnames):
            return self._regnames[num]
        if num < 8: return f"t{num-5}"
        if num < 10: return f"s{num-8}"
        if num < 18: return f"a{num-10}"
        if num < 28: return f"s{num-16}"
        if num < 32: return f"t{num-25}"
        raise Exception("invalid regnum")

    def csregname(self, num):
        """ regnum to name """
        if num in self._csrnames:
            return self._csrnames[num]
        return f"csr{num:03x}"

    def translate_csregname(self, name):
        """ name to register number """
        if name in self._csrnums:
            return self._csrnums[name]
        if m := re.match(r'csr([0-9a-f]+)', name):
            return int(m[1], 16)
        print(name)
        raise Exception("unknown csregname")


class BreakException(Exception):
    pass


class IllegalInstruction(Exception):
    pass


# TODO
# a device plugin has:
# methods called by Memory:
#    load(addr, size, value) -> value
#    store(addr, size, value)
# methods called by CPU:
#    tick(cpu)

class Memory:
    """
    keep track of memory contents.
    """
    def __init__(self, args):
        # memory contains bytes
        self.enablebreak = False
        self.memory = dict()
        self.args = args
        self.excluderanges = []
        self.pmp = None
        self.f_0008 = None
        self.n_7b0044 = 0

    def loadvalue(self, addr, size, recursed=False):
        """ load a 'size'-bit value from memory address 'addr' """
        if not recursed and self.pmp:
            if not self.pmp.mayread(addr, size):
                raise InvalidMemoryAccess()
        if size==8:
            # default value is 0 for memory
            res = self.memory.get(addr & 0xFFFFFFFF, None)
            if res is None:
                print("read uninitialized memory %08x" % addr)
                res = random.randint(0,255) if self.args.randomize else 0
                self.storevalue(addr, size, res, True)
        else:
            res = self.loadvalue(addr, size//2, True) | (self.loadvalue(addr+size//16, size//2, True)<<(size//2))
        if not recursed and self.args.trace:
            print("load  %08x:%02d -> %0*x" % (addr, size, size//4, res))

        return self.check_plugins_load(addr, size, res)

    def storevalue(self, addr, size, value, recursed=False):
        """ store a 'size'-bit value at memory address 'addr' """
        if not recursed and self.args.trace:
            print("store %08x:%02d := %0*x" % (addr, size, size//4, value))
        if self.enablebreak and addr in self.args.breakstores:
            print("break because store at %x" % addr)
            raise BreakException()
        if size==8:
            self.memory[addr & 0xFFFFFFFF] = value&0xFF
        else:
            self.storevalue(addr, size//2, value, True)
            self.storevalue(addr+size//16, size//2, value>>(size//2), True)

        self.check_plugins_store(addr, size, value)

    def hexdump(self, addr, data):
        for i, b in enumerate(data):
            if i%64 == 0:
                print("%08x: " % (addr + i), end="")
            print(" %02x" % b, end="")
            if (i+1)%64 == 0:
                print()
        print()

    def dump(self):
        bufaddr = None
        buf = []
        for a, v in sorted(self.memory.items()):
            if self.exclude(a):
                continue
            if bufaddr is None or bufaddr+len(buf) < a:
                if buf:
                    self.hexdump(bufaddr, buf)
                buf = []
                bufaddr = a

            buf.append(v)
        if buf:
            self.hexdump(bufaddr, buf)

    def loadprogram(self, addr, pgm):
        # pgm is the commandline specified byte sequence.
        # encoded as a sequence of byte strings
        o = addr
        for opcbytes in pgm:
            for i, b in enumerate(opcbytes):
                self.storevalue(o+i, 8, b, True)
            o += len(opcbytes)

    def loadimage(self, addr, data):
        """
        loads a raw binary image.
        """
        for i, b in enumerate(data):
            self.storevalue(addr+i, 8, b, True)

    def addexclude(self, a, b):
        self.excluderanges.append( (a, b) )

    def exclude(self, a):
        for r in self.excluderanges:
            if r[0] <= a < r[1]:
                return True

    def check_plugins_load(self, addr, size, res):
        if addr == 0x40630014 and size==32:
            # increment timer
            res += 15
            self.storevalue(addr, size, res, True)
        if addr in (0x40110004, 0x40110008) and res != 0xe89d48b7:
            if self.f_0008 is None:
                self.f_0008 = 2
            elif  self.f_0008:
                self.f_0008 -= 1
            else:
                self.f_0008 = None
                res = 0
                self.storevalue(addr, size, res, True)
        return res

    def check_plugins_store(self, addr, size, value):
        if addr == 0x40110010 and size==32 and value==0x00fb0043:
            # 'flash' erase
            self.storevalue(0x0017b000, 32, 0xFFFFFFFF, True)
            self.storevalue(0x0017b004, 32, 0xFFFFFFFF, True)
            self.n_7b0044 = 0
        elif addr == 0x40110010 and size==32 and value==0x007b0044:
            # 'flash' ?
            if self.n_7b0044 == 0:
                self.storevalue(0x0017b000, 32, 0x73614aff, True)
                self.n_7b0044 = 1
            else:
                self.storevalue(0x0017b000, 32, 0x73614af0, True)

            self.storevalue(0x0017b004, 32, 0x00000001, True)

        elif addr == 0x40110010 and size==32 and value==0x00f98003:
            # 'flash' erase
            self.storevalue(0x000f9800, 32, 0xFFFFFFFF, True)
            self.storevalue(0x000f9804, 32, 0xFFFFFFFF, True)
            self.n_7b0044 = 0
        elif addr == 0x40110010 and size==32 and value==0x00798004:
            # 'flash' ?
            if self.n_7b0044 == 0:
                self.storevalue(0x000f9800, 32, 0x73614aff, True)
                self.n_7b0044 = 1
            else:
                self.storevalue(0x000f9800, 32, 0x73614af0, True)

            self.storevalue(0x000f9804, 32, 0x00000001, True)
        elif addr == 0x40620010 and size==32 and (value & 9)==9:
            # satisfy FUN_000a90f4
            self.storevalue(addr, 32, 0, True)
        elif addr == 0x40620000 and size==32 and value!=0:
            # FUN_000a9202 wahts 0x40620004 to be != 0
            self.storevalue(0x40620004, 32, 1, True)
        elif addr == 0x40620000 and size==32 and value==0:
            # FUN_0008254e, FUN_0008268c wahts 0x40620004 to be == 0
            self.storevalue(0x40620004, 32, 0, True)


class NameResolver:
    """
    resolve offsets to name[+delta], or name to an offset.
    """
    def __init__(self):
        # names is a list of tuples(ofs, name), and kept sorted by offset.
        self.names = []
        self.byname = dict()

    def load(self, fn):
        """
        load names from the given file.
        The file has a simple format, where
        each line starts with a name and address.

        I use the output of the ghidra 'ctrl-t' symbol table, and copy-paste that to a file.
        """
        x = dict()
        for line in open(fn, "r"):
            if m := re.match(r'^(\S+)\t(\w+)', line):
                addr = int(m[2], 16)
                name = m[1]
                x[addr] = name
        self.names.extend(x.items())
        self.names = sorted(self.names)
        self.byname = { n: o for o, n in self.names }

    def getname(self, ofs):
        if not self.names:
            # names list is empty -> anything is unknown
            return "unknown"
        a = 0
        b = len(self.names)
        if ofs < self.names[a][0]:
            # ofs is before first item -> unknown
            return "unknown"
        elif ofs > self.names[b-1][0]:
            # ofs is after last item -> last + delta
            return "%s+%x" % (self.names[b-1][1], ofs-self.names[b-1][0])

        # otherwise do a binary search
        while a<b:
            m = (a+b)//2
            if ofs < self.names[m][0]:
                b = m
            elif self.names[m][0] < ofs:
                a = m + 1
            else:
                return self.names[m][1]

        o, n = self.names[a]
        if o < ofs:
            # return symbol + delta
            return "%s+%x" % (n, ofs-o)
        else:
            # exact match, return symbol itself.
            return n

    def resolve(self, name):
        if name is None:
            return
        try:
            return int(name, 0)
        except ValueError:
            if m := re.match(r'^(\w+)([+-]\d\w*)', name):
                n = self.byname.get(m[1])
                if n is None:
                    return
                o = int(m[2], 0)
                return n + o
            return self.byname.get(name)


def decodehex(fh):
    """
    Decode 'verilogtxt' files as supplied by `NF5`.
    These files consist of lines with a single address, followed by lines with hex bytes.

    This function yields tuples of (offset, data)
    """
    ofs = None
    data = None
    for line in fh:
        line = line.rstrip("\r\n")
        if m := re.match(r'^@(\w+)', line):
            if data:
                yield ofs, data
            ofs = int(m[1], 16)
            data = b""
        else:
            data += a2b_hex(line.replace(' ', ''))
    if data:
        yield ofs, data


def decodecall(cpu, names, txt):
    """
    decodes a call spec:
        memcpy(a0=1234, a1=4567, a2=9)
    """
    class Call:
        pass
    if m := re.match(r'(\S+)\((.*)\)', txt):
        c = Call()
        c.startaddr = names.resolve(m[1])
        c.regs = []
        argsspec = m[2]
        for argspec in re.split(r',\s*', argsspec):
            if not argspec:
                pass
            elif m := re.match(r'(\w+)=(\S+)', argspec):
                c.regs.append((cpu.translate_regname(m[1]), names.resolve(m[2])))
            else:
                raise Exception("unsupported argspec")
        return c


def get_trap_vector(vector, cause):
    """
    return the trap address for the cause.
    """
    def is_interrupt(c):
        # TODO
        return False
    if vector&3==1 and is_interrupt(cause):
        print("tv: v=%08x, cause=%d, int -> %08x" % (vector, cause, (vector&~3) + 4 * cause))
        return (vector&~3) + 4 * cause
    else:
        print("tv: v=%08x, cause=%d" % (vector, cause))
        return vector & ~3


def do_trap(cpu, cause):
    """
    Handle various kinds of traps.
    """
    if cpu.medeleg & (1<<cause) and cpu.priv()!=PRIV.MACHINE:
        del_priv = PRIV.SUPER
    else:
        del_priv = PRIV.MACHINE

    match del_priv:
        case PRIV.MACHINE:
            cpu.mcause = cause
            cpu.mstatus.MPIE = cpu.mstatus.MIE
            cpu.mstatus.MIE = 0
            cpu.mstatus.MPP = cpu.priv()
            # todo: mtval
            cpu.mepc = cpu.pc
            cpu.setpriv(del_priv)

            print("trap to priv=machine, mstatus = %08x(%s)" % (cpu.mstatus.bits(), cpu.mstatus))
            return get_trap_vector(cpu.mtvec, cause)

        case PRIV.SUPER:
            cpu.scause = cause
            cpu.mstatus.SPIE = cpu.mstatus.SIE
            cpu.mstatus.SIE = 0
            if cpu.priv()==PRIV.MACHINE:
                raise Exception("can't trap from machine to super")
            cpu.mstatus.SPP = cpu.priv()
            # todo: stval
            cpu.sepc = cpu.pc
            cpu.setpriv(del_priv)

            print("trap to priv=super, mstatus = %08x(%s)" % (cpu.mstatus.bits(), cpu.mstatus))
            return get_trap_vector(cpu.stvec, cause)

        case PRIV.USER:
            cpu.ucause = cause
            cpu.mstatus.UPIE = cpu.mstatus.UIE
            cpu.mstatus.UIE = 0
            # todo: utval
            cpu.uepc = cpu.pc
            cpu.setpriv(del_priv)

            print("trap to priv=user, mstatus = %08x(%s)" % (cpu.mstatus.bits(), cpu.mstatus))
            return get_trap_vector(cpu.utvec, cause)


def do_xret(cpu, retname):
    """
    handle return-from-trap
    """
    match retname:
        case 'mret':
            cpu.mstatus.MIE = cpu.mstatus.MPIE
            cpu.mstatus.MPIE = 1
            cpu.setpriv(cpu.mstatus.MPP)
            cpu.mstatus.MPP = PRIV.USER
            if cpu.priv() != PRIV.MACHINE:
                cpu.mstatus.MPRV = 0
            print("machine return to %08x, mstatus = %08x(%s)" % (cpu.mepc, cpu.mstatus.bits(), cpu.mstatus))
            return cpu.mepc
        case 'sret':
            cpu.mstatus.SIE = cpu.mstatus.SPIE
            cpu.mstatus.SPIE = 1
            cpu.setpriv(cpu.mstatus.SPP)
            cpu.mstatus.SPP = PRIV.USER if cpu.misa.U else PRIV.MACHINE

            # TODO: in sail, in 'exception_handler', first curpriv is set to super or user, then tested, == machine ??
            #     cur_privilege   = if mstatus.SPP() == 0b1 then Supervisor else User;
            #     mstatus->SPP()  = 0b0;
            #     if   cur_privilege != Machine   << this would always be true, right??
            #     then mstatus->MPRV() = 0b0;

            if cpu.priv() != PRIV.MACHINE:
                cpu.mstatus.MPRV = 0
            print("super return to %08x, mstatus = %08x(%s)" % (cpu.mepc, cpu.mstatus.bits(), cpu.mstatus))
            return cpu.sepc
        case 'uret':
            cpu.mstatus.UIE = cpu.mstatus.UPIE
            cpu.mstatus.UPIE = 1
            cpu.setpriv(PRIV.USER)

            print("user return to %08x, mstatus = %08x(%s)" % (cpu.mepc, cpu.mstatus.bits(), cpu.mstatus))
            return cpu.uepc


def evaluate_instruction(args, cpu, mem, logger, insn):
    """
    Evaluate a single decoded instruction.
    """

    # source: riscv-spec.
    #   paragraph 18.8, pages 124 - 125  have the detailed summary of 16-bit instructions.
    #   chapter 27, pages 150 - 156 have the detailed summary of 32-bit instructions

    # note: regnames inding in '_p'  are compressed regnums.
    # compressed regnum to full regnum:  r = c_r + 8 
    newpc = None
    match insn.mnemonic:
        # moves
        case 'c.mv': # rd, c_rs2_n0
            cpu.setreg(insn.rd, cpu.reg(insn.c_rs2_n0))
        case 'c.nop': # c_nzimm6hi, c_nzimm6lo
            pass

        # loads
        case 'c.li': # rd, c_imm6lo, c_imm6hi
            cpu.setreg(insn.rd, signed((insn.c_imm6hi<<5) | insn.c_imm6lo, 6))
        case 'lui':  # rd, imm20
            cpu.setreg(insn.rd, unsigned(insn.imm20<<12))
        case 'c.lui': # rd_n2, c_nzimm18hi, c_nzimm18lo
            cpu.setreg(insn.rd_n2, signed((insn.c_nzimm18hi<<5) | insn.c_nzimm18lo, 6) << 12)
        case 'auipc': # rd, imm20
            cpu.setreg(insn.rd, cpu.pc + unsigned(insn.imm20<<12))

        # add, sub
        case 'add': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) + cpu.reg(insn.rs2))
        case 'addi': # rd, rs1, imm12
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) + signed(insn.imm12, 12))
        case 'c.addi':  # rd_rs1_n0, c_nzimm6lo, c_nzimm6hi
            cpu.setreg(insn.rd_rs1_n0, cpu.reg(insn.rd_rs1_n0) + signed((insn.c_nzimm6hi<<5) | insn.c_nzimm6lo, 6))
        case 'c.add': # rd_rs1, c_rs2_n0
            cpu.setreg(insn.rd_rs1, cpu.reg(insn.rd_rs1) + cpu.reg(insn.c_rs2_n0))
        case 'c.addi4spn':  # rd_p, c_nzuimm10
            ofs = unsigned(reorderbits(insn.c_nzuimm10, (5,4), (9,6), 2, 3))
            cpu.setreg(insn.rd_p+8, cpu.sp + ofs)
        case 'c.addi16sp': # c_nzimm10hi, c_nzimm10lo
            ofs = signed(reorderbits((insn.c_nzimm10hi<<5) | insn.c_nzimm10lo, 9, 4, 6, 8, 7, 5), 10)
            cpu.sp += ofs
        case 'sub': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) - cpu.reg(insn.rs2))
        case 'c.sub': # rd_rs1_p, rs2_p
            cpu.setreg(insn.rd_rs1_p+8, cpu.reg(insn.rd_rs1_p+8) - cpu.reg(insn.rs2_p+8))

        # shifts
        case 'c.slli': # rd_rs1_n0, c_nzuimm6lo c_nzuimm6hi
            # rv32: shift no more than 32 bits
            if not insn.c_nzuimm6hi:
                cpu.setreg(insn.rd_rs1_n0, cpu.reg(insn.rd_rs1_n0) << insn.c_nzuimm6lo)
        case 'c.srli': # rd_rs1_p, c_nzuimm6lo c_nzuimm6hi
            if not insn.c_nzuimm6hi:
                cpu.setreg(insn.rd_rs1_p+8, cpu.reg(insn.rd_rs1_p+8,) >> insn.c_nzuimm6lo)
        case 'c.srai': # rd_rs1_p, c_nzuimm6lo c_nzuimm6hi
            if not insn.c_nzuimm6hi:
                cpu.setreg(insn.rd_rs1_p+8, signed(cpu.reg(insn.rd_rs1_p+8), 32) >> insn.c_nzuimm6lo)
        case 'slli': # rd, rs1, shamt
            # rv32: shift no more than 32 bits
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) << (insn.shamt&31))
        case 'srli': # rd, rs1, shamt
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) >> (insn.shamt&31))
        case 'srai': # rd, rs1, shamt
            cpu.setreg(insn.rd, signed(cpu.reg(insn.rs1), 32) >> (insn.shamt&31))
        case 'sll': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) << (cpu.reg(insn.rs2)&31))
        case 'srl': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) >> (cpu.reg(insn.rs2)&31))
        case 'sra': # rd, rs1, rs2
            cpu.setreg(insn.rd, signed(cpu.reg(insn.rs1), 32) >> (cpu.reg(insn.rs2)&31))

        # or / and / xor
        case 'c.or': # rd_rs1_p, rs2_p
            cpu.setreg(insn.rd_rs1_p+8, cpu.reg(insn.rd_rs1_p+8) | cpu.reg(insn.rs2_p+8))
        case 'or': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) | cpu.reg(insn.rs2))
        case 'ori': # rd, rs1, imm12
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) | signed(insn.imm12, 12))

        case 'c.and': # rd_rs1_p, rs2_p
            cpu.setreg(insn.rd_rs1_p+8, cpu.reg(insn.rd_rs1_p+8) & cpu.reg(insn.rs2_p+8))
        case 'and': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) & cpu.reg(insn.rs2))
        case 'andi': # rd, rs1, imm12
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) & signed(insn.imm12, 12))
        case 'c.andi': # rd_rs1_p, c_imm6hi, c_imm6lo
            cpu.setreg(insn.rd_rs1_p+8, cpu.reg(insn.rd_rs1_p+8) & signed((insn.c_imm6hi<<5) | insn.c_imm6lo, 6))

        case 'c.xor': # rd_rs1_p, rs2_p
            cpu.setreg(insn.rd_rs1_p+8, cpu.reg(insn.rd_rs1_p+8) ^ cpu.reg(insn.rs2_p+8))
        case 'xori': # rd, rs1, imm12
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) ^ signed(insn.imm12, 12))
        case 'xor': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) ^ cpu.reg(insn.rs2))

        # mul / div
        case 'mul': # rd, rs1, rs2
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) * cpu.reg(insn.rs2))
        case 'mulhu': # rd, rs1, rs2
            cpu.setreg(insn.rd, (cpu.reg(insn.rs1) * cpu.reg(insn.rs2)) >> 32)
        case 'mulh': # rd, rs1, rs2
            cpu.setreg(insn.rd, (signed(cpu.reg(insn.rs1), 32) * signed(cpu.reg(insn.rs2), 32)) >> 32)
        case 'mulhsu': # rd, rs1, rs2
            cpu.setreg(insn.rd, (signed(cpu.reg(insn.rs1), 32) * cpu.reg(insn.rs2)) >> 32)

        # dividend = divisor * quotient + remainder
        # REM:  sign(remainder) = sign(dividend)
        # DIV, DIVU: round towards 0

        # REMU(x/0) = x
        case 'remu': # rd, rs1, rs2
            a = cpu.reg(insn.rs1)
            b = cpu.reg(insn.rs2)
            r = a % b if b else a
            cpu.setreg(insn.rd, r)

        # REM(x/0) = x
        # REM(-2^31/-1) = 0
        case 'rem': # rd, rs1, rs2
            a = signed(cpu.reg(insn.rs1), 32)
            b = signed(cpu.reg(insn.rs2), 32)
            sa = -1 if a<0 else 1
            sb = -1 if b<0 else 1
            if a==-2**31 and b==-1:
                r = 0
            elif b==0:
                r = a
            else:
                r = abs(a) % abs(b) if b else a
                r *= sa

            cpu.setreg(insn.rd, r)

        # DIVU(x/0) = 2^32-1
        case 'divu': # rd, rs1, rs2
            a = cpu.reg(insn.rs1)
            b = cpu.reg(insn.rs2)
            q = a // b if b else 0xFFFFFFFF
            cpu.setreg(insn.rd, q)

        # DIV(x/0) = -1
        # DIV(-2^31/-1) = -2^31
        case 'div': # rd, rs1, rs2
            a = signed(cpu.reg(insn.rs1), 32)
            b = signed(cpu.reg(insn.rs2), 32)
            sa = -1 if a<0 else 1
            sb = -1 if b<0 else 1
            if a==-2**31 and b==-1:
                q = a
            elif b==0:
                q = 0xFFFFFFFF
            else:
                q = abs(a) // abs(b)
                q *= sa * sb

            cpu.setreg(insn.rd, q)

        # assign conditional 
        case 'sltiu': # rd, rs1, imm12
            cpu.setreg(insn.rd, cpu.reg(insn.rs1) < unsigned(signed(insn.imm12, 12)))
        case 'slti': # rd, rs1, imm12
            cpu.setreg(insn.rd, signed(cpu.reg(insn.rs1), 32) < signed(insn.imm12, 12))
        case 'sltu': # rd, rs1, rs2
            if insn.rs1==0:
                cpu.setreg(insn.rd, cpu.reg(insn.rs2)!=0)
            else:
                cpu.setreg(insn.rd, cpu.reg(insn.rs1) < cpu.reg(insn.rs2))
        case 'slt': # rd, rs1, rs2
            cpu.setreg(insn.rd, signed(cpu.reg(insn.rs1), 32) < signed(cpu.reg(insn.rs2), 32))

        # load from stack / memory
        case 'c.lw':  # rd_p, rs1_p, c_uimm7lo, c_uimm7hi
            ofs = unsigned(reorderbits(insn.c_uimm7hi<<2 | insn.c_uimm7lo, (5,3), 2, 6), 7)
            cpu.setreg(insn.rd_p+8, mem.loadvalue(cpu.reg(insn.rs1_p+8) + ofs, 32))
        case 'c.lwsp': # rd_n0, c_uimm8sphi, c_uimm8splo
            ofs = unsigned(reorderbits(insn.c_uimm8sphi<<5 | insn.c_uimm8splo, 5, (4,2), 7,6))
            addr = cpu.sp + ofs
            cpu.setreg(insn.rd_n0, mem.loadvalue(addr, 32))
        case 'lb': # rd, rs1, imm12
            ofs = signed(insn.imm12, 12)
            cpu.setreg(insn.rd, signed(mem.loadvalue(cpu.reg(insn.rs1) + ofs, 8), 8))
        case 'lbu': # rd, rs1, imm12
            ofs = signed(insn.imm12, 12)
            cpu.setreg(insn.rd, mem.loadvalue(cpu.reg(insn.rs1) + ofs, 8))
        case 'lh': # rd, rs1, imm12
            ofs = signed(insn.imm12, 12)
            cpu.setreg(insn.rd, signed(mem.loadvalue(cpu.reg(insn.rs1) + ofs, 16), 16))
        case 'lhu': # rd, rs1, imm12
            ofs = signed(insn.imm12, 12)
            cpu.setreg(insn.rd, mem.loadvalue(cpu.reg(insn.rs1) + ofs, 16))
        case 'lw':  # rd, rs1, imm12
            ofs = cpu.reg(insn.rs1) + signed(insn.imm12, 12)
            cpu.setreg(insn.rd, mem.loadvalue(ofs, 32))

        # store to stack / memory
        case 'c.sw': # rs1_p, rs2_p, c_uimm7lo, c_uimm7hi
            ofs = unsigned(reorderbits(insn.c_uimm7hi<<2 | insn.c_uimm7lo, (5,3), 2, 6), 7)
            mem.storevalue(cpu.reg(insn.rs1_p+8) + ofs, 32, cpu.reg(insn.rs2_p+8))
        case 'c.sb': # rs1_p, rs2_p, c_uimm2
            # note: insn is not in the isa manual !!
            ofs = unsigned(insn.c_uimm2)
            mem.storevalue(cpu.reg(insn.rs1_p+8) + ofs, 32, cpu.reg(insn.rs2_p+8))
        case 'c.swsp': # c_rs2, c_uimm8sp_s
            ofs = unsigned(reorderbits(insn.c_uimm8sp_s, (5,2), 7, 6))
            addr = cpu.sp + ofs
            mem.storevalue(addr, 32, cpu.reg(insn.c_rs2))
        case 'sb': # imm12hi, rs1, rs2, imm12lo
            ofs = signed(insn.imm12hi<<5 | insn.imm12lo, 12)
            mem.storevalue(cpu.reg(insn.rs1) + ofs, 8, cpu.reg(insn.rs2) & 0xFF)
        case 'sh': # imm12hi, rs1, rs2, imm12lo
            ofs = signed(insn.imm12hi<<5 | insn.imm12lo, 12)
            mem.storevalue(cpu.reg(insn.rs1) + ofs, 16, cpu.reg(insn.rs2) & 0xFFFF)
        case 'sw':  # imm12hi, rs1, rs2, imm12lo
            ofs = signed(insn.imm12hi<<5 | insn.imm12lo, 12)
            mem.storevalue(cpu.reg(insn.rs1) + ofs, 32, cpu.reg(insn.rs2))

        # control flow
        case 'c.j':  # c_imm12
            ofs = signed(reorderbits(insn.c_imm12, 11, 4, (9,8), 10, 6, 7, (3,1), 5), 12)
            newpc = unsigned(cpu.pc + ofs)
        case 'c.jr': # rs1_n0
            newpc = cpu.reg(insn.rs1_n0)
            if insn.rs1_n0 == 1:
                logger.leave()
        case 'c.jal': # c_imm12
            ofs = signed(reorderbits(insn.c_imm12, 11, 4, 9, 8, 10, 6, 7, (3,1), 5), 12)
            cpu.ra = cpu.nextpc
            newpc = unsigned(cpu.pc + ofs)

            logger.enter(newpc, cpu)
        case 'jal': # rd, jimm20
            ofs = signed(reorderbits(insn.jimm20, 20, (10,1), 11, (19,12)), 21)
            if insn.rd:
                cpu.setreg(insn.rd, cpu.nextpc)
            newpc = unsigned(cpu.pc + ofs)
            logger.enter(newpc, cpu)
        case 'c.jalr': # c_rs1_n0
            dst = cpu.reg(insn.c_rs1_n0) & ~1
            cpu.ra = cpu.nextpc
            newpc = dst
            logger.enter(newpc, cpu)
        case 'jalr': # rd, rs1, imm12
            dst = unsigned((cpu.reg(insn.rs1) + signed(insn.imm12, 12)) & ~1)
            if insn.rd:
                cpu.setreg(insn.rd, cpu.nextpc)
            newpc = dst
            if insn.rd==1 and insn.rs1==0 and insn.imm12==0:
                logger.leave()
            else:
                logger.enter(newpc, cpu)

        # conditional control flow
        case 'c.bnez': # rs1_p, c_bimm9lo, c_bimm9hi
            if cpu.reg(insn.rs1_p+8):
                ofs = signed(reorderbits((insn.c_bimm9hi<<5) | insn.c_bimm9lo, 8, (4,3), (7,6), (2,1), 5), 9)
                newpc = unsigned(cpu.pc + ofs)
        case 'c.beqz': # rs1_p, c_bimm9lo, c_bimm9hi
            if not cpu.reg(insn.rs1_p+8):
                ofs = signed(reorderbits((insn.c_bimm9hi<<5) | insn.c_bimm9lo, 8, (4,3), (7,6), (2,1), 5), 9)
                newpc = unsigned(cpu.pc + ofs)
        case 'bne': # bimm12hi, rs1, rs2, bimm12lo
            if cpu.reg(insn.rs1) != cpu.reg(insn.rs2):
                ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
                newpc = unsigned(cpu.pc + ofs)
        case 'beq': # bimm12hi, rs1, rs2, bimm12lo
            if cpu.reg(insn.rs1) == cpu.reg(insn.rs2):
                ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
                newpc = unsigned(cpu.pc + ofs)
        case 'bltu': # bimm12hi, rs1, rs2, bimm12lo
            # note: inconsistency in the riscv manual: it says offset[11|4:1]  iso offset[4:1|11]
            if cpu.reg(insn.rs1) < cpu.reg(insn.rs2):
                ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
                newpc = unsigned(cpu.pc + ofs)
        case 'bgeu': # bimm12hi, rs1, rs2, bimm12lo
            if cpu.reg(insn.rs1) >= cpu.reg(insn.rs2):
                ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
                newpc = unsigned(cpu.pc + ofs)
        case 'blt': # bimm12hi, rs1, rs2, bimm12lo
            if signed(cpu.reg(insn.rs1), 32) < signed(cpu.reg(insn.rs2), 32):
                ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
                newpc = unsigned(cpu.pc + ofs)
        case 'bge': # bimm12hi, rs1, rs2, bimm12lo
            if signed(cpu.reg(insn.rs1), 32) >= signed(cpu.reg(insn.rs2), 32):
                ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
                newpc = unsigned(cpu.pc + ofs)

        # system call/ret
        case ('scall' | 'ecall' | 'ebreak' | 'illegal') as trapname: # 
            match trapname:
                case 'scall' | 'ecall':
                    match cpu.priv():
                        case PRIV.USER:
                            cause = 8
                        case PRIV.SUPER:
                            cause = 9
                        case PRIV.MACHINE:
                            cause = 11
                    cpu.mtval = 0
                case 'ebreak':
                    cause = 3
                    cpu.mtval = cpu.pc
                case 'illegal':
                    cause = 2
                    cpu.mtval = insn.opc

            newpc = do_trap(cpu, cause)

            if args.breakonscall:
                print("break on scall")
                raise BreakException()

        case ('mret' | 'sret' | 'uret') as retname:
            newpc = do_xret(cpu, retname)
        # system
        case 'csrrw': # rd, rs1, csr
            v = cpu.reg(insn.rs1)
            if insn.rd:
                # only read when rd!=zero
                cpu.setreg(insn.rd, cpu.csreg(insn.csr))
            cpu.setcsreg(insn.csr, v)
        case 'csrrs': # rd, rs1, csr
            mask = cpu.reg(insn.rs1)
            v = cpu.csreg(insn.csr)
            cpu.setreg(insn.rd, v)
            if insn.rs1:
                # only write when rs1!=zero
                cpu.setcsreg(insn.csr, v | mask)
        case 'csrrc': # rd, rs1, csr
            mask = cpu.reg(insn.rs1)
            v = cpu.csreg(insn.csr)
            cpu.setreg(insn.rd, v)
            if insn.rs1:
                # only write when rs1!=zero
                cpu.setcsreg(insn.csr, cpu.csreg(insn.csr) & ~mask)
        case 'csrrwi': # rd, csr, zimm
            if insn.rd:
                # only read when rd!=zero
                cpu.setreg(insn.rd, cpu.csreg(insn.csr))
            cpu.setcsreg(insn.csr, unsigned(insn.zimm, 5))
        case 'csrrci': # rd, csr, zimm
            v = cpu.csreg(insn.csr)
            if insn.rd:
                cpu.setreg(insn.rd, v)
            if insn.zimm:
                # only write when imm!=0
                cpu.setcsreg(insn.csr, v & ~insn.zimm)
        case 'csrrsi': # rd, csr, zimm
            v = cpu.csreg(insn.csr)
            if insn.rd:
                cpu.setreg(insn.rd, v)
            if insn.zimm:
                # only write when imm!=0
                cpu.setcsreg(insn.csr, v | insn.zimm)
        case 'wfi': #  wait for interrupt
            pass
        case 'fence':
            pass
        case 'fence.i':
            pass
        case 'sfence.vma':
            pass

        # custom Titan-M2 instructions.
        case 'grbitscan':
            # count number of zeroes starting from bit 31
            x = cpu.reg(insn.rs1)
            i = 31
            while i>0:
                if x&(1<<i):
                    break
                i -= 1

            cpu.setreg(insn.rd, 32-i)

        case 'gbitscan':
            # count number of zeroes starting from bit 0
            x = cpu.reg(insn.rs1)
            i = 0
            while i<32:
                if x&(1<<i):
                    break
                i += 1

            cpu.setreg(insn.rd, i)

        case 'gbswap32':
            # byte-order swap
            x = cpu.reg(insn.rs1)
            x = (x>>24) | ((x>>8)&0xff00) | ((x<<8)&0xff0000) | ((x&0xFF) <<24)
            cpu.setreg(insn.rd, x)
        case 'gclrbit':
            # clear bit by bitnumber from register
            x = cpu.reg(insn.rs1)
            bit = cpu.reg(insn.rs2)
            cpu.setreg(insn.rd, x & ~(1<<bit))
        case 'gsetbit':
            # set bit by bitnumber from register
            x = cpu.reg(insn.rs1)
            bit = cpu.reg(insn.rs2)
            cpu.setreg(insn.rd, x | (1<<bit))
        case 'gclrbiti':
            # clear bit by bitnumber from immediate
            x = cpu.reg(insn.rs1)
            bit = insn.shamt
            cpu.setreg(insn.rd, x & ~(1<<bit))
        case 'gsetbiti':
            # set bit by bitnumber from immediate
            x = cpu.reg(insn.rs1)
            bit = insn.shamt
            cpu.setreg(insn.rd, x | (1<<bit))
        case _:
            print("WARN: unimplemented insn: %s" % insn.mnemonic)
            if args.breakonunimplemented:
                print("pc=%08x, break on unimplemented" % cpu.pc)
                raise BreakException()

    return newpc


def disassemble_instruction(cpu, insn):
    """
    disassemble a single decoded instruction.

    Note that I don't follow the standard way of disassembling Risc-V instructions.
    For loads and stores, I add [] around the registers specifying the offset.
    """
    operands = None
    mnemonic = insn.mnemonic
    match mnemonic:
        # moves
        case 'c.mv': # rd, c_rs2_n0
            operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.c_rs2_n0))
        case 'c.nop': # c_nzimm6hi, c_nzimm6lo
            operands = ""

        # loads
        case 'c.li': # rd, c_imm6lo, c_imm6hi
            operands = "%s, #%+x" % (cpu.regname(insn.rd), signed((insn.c_imm6hi<<5) | insn.c_imm6lo, 6))
        case 'lui':  # rd, imm20
            operands = "%s, %x___" % (cpu.regname(insn.rd), insn.imm20)
        case 'c.lui': # rd_n2, c_nzimm18hi, c_nzimm18lo
            operands = "%s, #%x___" % (cpu.regname(insn.rd_n2), unsigned(signed((insn.c_nzimm18hi<<5) | insn.c_nzimm18lo, 6)))
        case 'auipc': # rd, imm20
            operands = "%s, pc, #%x___" % (cpu.regname(insn.rd), insn.imm20)

        # add, sub
        case 'add': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'addi': # rd, rs1, imm12
            if insn.imm12:
                if insn.rs1:
                    operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
                else:
                    mnemonic = "li"
                    operands = "%s, #%+x" % (cpu.regname(insn.rd), signed(insn.imm12, 12))
            else:
                if insn.rd:
                    mnemonic = "mv"
                    operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1))
                else:
                    mnemonic = "nop"
                    operands = ""

        case 'c.addi':  # rd_rs1_n0, c_nzimm6lo, c_nzimm6hi
            operands = "%s, #%+x" % (cpu.regname(insn.rd_rs1_n0), signed((insn.c_nzimm6hi<<5) | insn.c_nzimm6lo, 6))
        case 'c.addi4spn':  # rd_p, c_nzuimm10
            ofs = unsigned(reorderbits(insn.c_nzuimm10, (5,4), (9,6), 2, 3))
            operands = "%s, %s, #%+x" % (cpu.regname(insn.rd_p+8), "sp", ofs)
        case 'c.add': # rd_rs1, c_rs2_n0
            operands = "%s, %s, %s" % (cpu.regname(insn.rd_rs1), cpu.regname(insn.rd_rs1), cpu.regname(insn.c_rs2_n0))
        case 'c.addi16sp': # c_nzimm10hi, c_nzimm10lo
            ofs = signed(reorderbits((insn.c_nzimm10hi<<5) | insn.c_nzimm10lo, 9, 4, 6, 8, 7, 5), 10)
            operands = "%s, %s, #%x" % (cpu.regname(2), "sp", ofs)

        case 'sub': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'c.sub': # rd_rs1_p, rs2_p
            operands = "%s, %s, %s" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rs2_p+8))

        # shifts
        case 'c.slli': # rd_rs1_n0, c_nzuimm6lo c_nzuimm6hi
            operands = "%s, %s, #%d" % (cpu.regname(insn.rd_rs1_n0), cpu.regname(insn.rd_rs1_n0), insn.c_nzuimm6lo)
        case 'c.srli': # rd_rs1_p, c_nzuimm6lo c_nzuimm6hi
            operands = "%s, %s, #%d" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), insn.c_nzuimm6lo)
        case 'c.srai': # rd_rs1_p, c_nzuimm6lo c_nzuimm6hi
            operands = "%s, %s, #%d" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), insn.c_nzuimm6lo)
        case 'slli': # rd, rs1, shamt
            operands = "%s, %s, #%d" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), insn.shamt)
        case 'srli': # rd, rs1, shamt
            operands = "%s, %s, #%d" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), insn.shamt)
        case 'srai': # rd, rs1, shamt
            operands = "%s, %s, #%d" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), insn.shamt)
        case 'sll': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'srl': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'sra': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))

        # or / and / xor
        case 'c.or': # rd_rs1_p, rs2_p
            operands = "%s, %s, %s" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rs2_p+8))
        case 'or': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'ori': # rd, rs1, imm12
            operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))

        case 'c.and': # rd_rs1_p, rs2_p
            operands = "%s, %s, %s" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rs2_p+8))
        case 'and': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'andi': # rd, rs1, imm12
            operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'c.andi': # rd_rs1_p, c_imm6hi, c_imm6lo
            operands = "%s, %s, #%x" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), signed((insn.c_imm6hi<<5) | insn.c_imm6lo, 6))

        case 'c.xor': # rd_rs1_p, rs2_p
            operands = "%s, %s, %s" % (cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rd_rs1_p+8), cpu.regname(insn.rs2_p+8))
        case 'xori': # rd, rs1, imm12
            if insn.imm12==0xFFF:
                mnemonic = 'not'
                operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1))
            else:
                operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'xor': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))

        # mul / div
        case 'mul': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'mulhu': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'mulh': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'mulhsu': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'remu': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'rem': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'divu': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'div': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))

        # assign conditional 
        case 'sltiu': # rd, rs1, imm12
            if insn.imm12==1:
                mnemonic = 'seqz'
                operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1))
            else:
                operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), unsigned(signed(insn.imm12, 12)))
        case 'slti': # rd, rs1, imm12
            operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'sltu': # rd, rs1, rs2
            if insn.rs1==0:
                mnemonic = "snez"
                operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs2))
            else:
                operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'slt': # rd, rs1, rs2
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))

        # load from stack / memory
        case 'c.lw':  # rd_p, rs1_p, c_uimm7lo, c_uimm7hi
            ofs = unsigned(reorderbits(insn.c_uimm7hi<<2 | insn.c_uimm7lo, (5,3), 2, 6), 7)
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd_p+8), cpu.regname(insn.rs1_p+8), ofs)
        case 'c.lwsp': # rd_n0, c_uimm8sphi, c_uimm8splo
            ofs = unsigned(reorderbits(insn.c_uimm8sphi<<5 | insn.c_uimm8splo, 5, (4,2), 7,6))
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd_n0), "sp", ofs)
        case 'lb': # rd, rs1, imm12
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'lbu': # rd, rs1, imm12
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'lh': # rd, rs1, imm12
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'lhu': # rd, rs1, imm12
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))
        case 'lw':  # rd, rs1, imm12
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))

        # store to stack / memory
        case 'c.sw': # rs1_p, rs2_p, c_uimm7lo, c_uimm7hi
            ofs = unsigned(reorderbits(insn.c_uimm7hi<<2 | insn.c_uimm7lo, (5,3), 2, 6), 7)
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rs2_p+8), cpu.regname(insn.rs1_p+8), ofs)
        case 'c.sb': # rs1_p, rs2_p, c_uimm2
            # note: insn is not in the isa manual !!
            ofs = unsigned(insn.c_uimm2)
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rs2_p+8), cpu.regname(insn.rs1_p+8), ofs)
        case 'c.swsp': # c_rs2, c_uimm8sp_s
            ofs = unsigned(reorderbits(insn.c_uimm8sp_s, (5,2), 7, 6))
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.c_rs2), "sp", ofs)
        case 'sb': # imm12hi, rs1, rs2, imm12lo
            ofs = signed(insn.imm12hi<<5 | insn.imm12lo, 12)
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rs2), cpu.regname(insn.rs1), ofs)
        case 'sh': # imm12hi, rs1, rs2, imm12lo
            ofs = signed(insn.imm12hi<<5 | insn.imm12lo, 12)
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rs2), cpu.regname(insn.rs1), ofs)
        case 'sw':  # imm12hi, rs1, rs2, imm12lo
            ofs = signed(insn.imm12hi<<5 | insn.imm12lo, 12)
            operands = "%s, [%s, #%+x]" % (cpu.regname(insn.rs2), cpu.regname(insn.rs1), ofs)

        # control flow
        case 'c.j':  # c_imm12
            ofs = signed(reorderbits(insn.c_imm12, 11, 4, (9,8), 10, 6, 7, (3,1), 5), 12)
            operands = "%08x" % (cpu.pc + ofs)
        case 'c.jr': # rs1_n0
            if insn.rs1_n0 == 1:
                mnemonic = "c.ret"
                operands = ""
            else:
                operands = "%s" % (cpu.regname(insn.rs1_n0))
        case 'c.jal': # c_imm12
            ofs = signed(reorderbits(insn.c_imm12, 11, 4, 9, 8, 10, 6, 7, (3,1), 5), 12)
            operands = "%08x" % (cpu.pc + ofs)
        case 'jal': # rd, jimm20
            ofs = signed(reorderbits(insn.jimm20, 20, (10,1), 11, (19,12)), 21)
            if insn.rd:
                operands = "%s, %08x" % (cpu.regname(insn.rd), cpu.pc + ofs)
            else:
                mnemonic = 'j'
                operands = "%x" % (cpu.pc + ofs)
        case 'c.jalr': # c_rs1_n0
            operands = "ra, %s" % cpu.regname(insn.c_rs1_n0)
        case 'jalr': # rd, rs1, imm12
            if insn.rd==1 and insn.rs1==0 and insn.imm12==0:
                mnemonic = "ret"
                operandse = ""
            else:
                operands = "%s, %s, #%+x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), signed(insn.imm12, 12))

        # conditional control flow
        case 'c.bnez' | 'c.beqz': # rs1_p, c_bimm9lo, c_bimm9hi
            ofs = signed(reorderbits((insn.c_bimm9hi<<5) | insn.c_bimm9lo, 8, (4,3), (7,6), (2,1), 5), 9)
            operands = "%s, zero, %08x" % (cpu.regname(insn.rs1_p+8), cpu.pc+ofs)

        case 'beq'|'bltu'|'bgeu'|'blt'|'bge'|'bne': # bimm12hi, rs1, rs2, bimm12lo
            ofs = signed(reorderbits((insn.bimm12hi<<5) | insn.bimm12lo, 12, (10,5), (4,1), 11), 13)
            operands = "%s, %s, %08x" % (cpu.regname(insn.rs1), cpu.regname(insn.rs2), cpu.pc+ofs)
         

        # system call/ret
        case 'scall' | 'ecall': # 
            operands = ""
        case 'mret': #  -- return from scall
            operands = ""
        case 'sret': #  -- return from scall
            operands = ""
        case 'ebreak':
            operands = ""

        # system
        #   csrr rd, csr 	csrrs rd, csr, x0 	Read CSR
        #   csrw csr, rs 	csrrw x0, csr, rs 	Write CSR
        #   csrs csr, rs 	csrrs x0, csr, rs 	Set bits in CSR
        #   csrc csr, rs 	csrrc x0, csr, rs 	Clear bits in CSR
        #   csrwi csr, imm 	csrrwi x0, csr, imm 	Write CSR, immediate
        #   csrsi csr, imm 	csrrsi x0, csr, imm 	Set bits in CSR, immediate
        #   csrci csr, imm 	csrrci x0, csr, imm 	Clear bits in CSR, immediate
        case 'csrrw': # rd, rs1, csr
            if insn.rd:
                operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.csregname(insn.csr), cpu.regname(insn.rs1))
            else:
                mnemonic = "csrw"
                operands = "%s, %s" % (cpu.csregname(insn.csr), cpu.regname(insn.rs1))
        case 'csrrs': # rd, rs1, csr
            if insn.rs1==0:
                mnemonic = 'csrr'
                operands = "%s, %s" % (cpu.regname(insn.rd), cpu.csregname(insn.csr))
            elif insn.rd==0:
                mnemonic = 'csrs'
                operands = "%s, %s" % (cpu.csregname(insn.csr), cpu.regname(insn.rs1))
            else:
                operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.csregname(insn.csr), cpu.regname(insn.rs1))

        case 'csrrc': # rd, rs1, csr
            if insn.rs1:
                operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.csregname(insn.csr), cpu.regname(insn.rs1))
            else:
                mnemonic = "csrc"
                operands = "%s, %s" % (cpu.csregname(insn.csr), cpu.regname(insn.rd))
        case 'csrrwi': # rd, csr, zimm
            if insn.rd:
                operands = "%s, %s, #%x" % (cpu.regname(insn.rd), cpu.csregname(insn.csr), insn.zimm)
            else:
                mnemonic = "csrwi"
                operands = "%s, #%x" % (cpu.csregname(insn.csr), insn.zimm)
        case 'csrrci': # rd, csr, zimm
            if insn.rd:
                operands = "%s, %s, #%x" % (cpu.regname(insn.rd), cpu.csregname(insn.csr), insn.zimm)
            else:
                mnemonic = "csrci"
                operands = "%s, #%x" % (cpu.csregname(insn.csr), insn.zimm)
        case 'csrrsi': # rd, csr, zimm
            if insn.rd:
                operands = "%s, %s, #%x" % (cpu.regname(insn.rd), cpu.csregname(insn.csr), insn.zimm)
            else:
                mnemonic = "csrsi"
                operands = "%s, #%x" % (cpu.csregname(insn.csr), insn.zimm)
        case 'wfi': #  wait for interrupt
            operands = ""
        case 'fence': 
            operands = ""
        case 'fence.i':
            operands = ""
        case 'sfence.vma':
            operands = ""

        # custom insn
        case 'gbitscan' | 'grbitscan':
            operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1))
        case 'gbswap32':
            operands = "%s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1))
        case 'gclrbit' | 'gsetbit':
            operands = "%s, %s, %s" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), cpu.regname(insn.rs2))
        case 'gclrbiti' | 'gsetbiti':
            operands = "%s, %s, #%x" % (cpu.regname(insn.rd), cpu.regname(insn.rs1), insn.shamt)
        case _:
            operands = "unimplemented"

    return mnemonic, operands


def simulate(decoder, names, args):
    """
    Load state, then simulate the specified calls.
    """

    cpu = CPU(args)
    mem = Memory(args)

    if args.setpriv is not None:
        cpu.setpriv(args.setpriv)

    if args.insn:
        mem.loadprogram(args.startaddr, args.insn)

    for filename in args.heximages:
        baseofs = 0
        if filename.find('test_0x8000')>=0:
            baseofs = 0x80000000
        for ofs, data in decodehex(open(filename)):
            mem.loadimage(baseofs+ofs, data)

    for arg in args.images:
        filename, loadofs = arg.split('@', 1)
        loadofs = names.resolve(loadofs)
        data = open(filename, "rb").read()
        mem.loadimage(loadofs, data)
        mem.addexclude(loadofs, loadofs+len(data))

    for arg in args.registers:
        reg, value = arg.split('=', 1)
        reg = cpu.translate_regname(reg)
        value = names.resolve(value)
        cpu.setreg(reg, value)

    for arg in args.csregisters:
        reg, value = arg.split('=', 1)
        reg = cpu.translate_csregname(reg)
        value = names.resolve(value)
        cpu.setcsreg(reg, value)

    for arg in args.memory:
        addr, value = arg.split('=', 1)
        addr = names.resolve(addr)
        if m := re.match(r'^(\w+)\.\.(\w+)(?::(\w+))?$', value):
            # range
            lo = int(m[1], 0)
            hi = int(m[2], 0)
            step = int(m[3], 0) if m[3] else 1
            for value in range(lo, hi+1, step):
                mem.storevalue(addr, 32, value)
                addr += 4
        elif m := re.match(r'^(\w+)\*(\w+)$', value):
            # repeat n times
            value = int(m[1], 0)
            n = int(m[2], 0)
            for _ in range(n):
                mem.storevalue(addr, 32, value)
                addr += 4
        else:
            # single, or list of values.
            for val in value.split(","):
                val = names.resolve(val)
                if val is None:
                    print("Bad value: %s" % value)
                    raise Exception("Can't resolve value")
                mem.storevalue(addr, 32, val)
                addr += 4

    for arg in args.half:
        addr, value = arg.split('=', 1)
        addr = names.resolve(addr)
        if m := re.match(r'^(\w+)\.\.(\w+)(?::(\w+))?$', value):
            # range
            lo = int(m[1], 0)
            hi = int(m[2], 0)
            step = int(m[3], 0) if m[3] else 1
            for value in range(lo, hi+1, step):
                mem.storevalue(addr, 16, value)
                addr += 2
        elif m := re.match(r'^(\w+)\*(\w+)$', value):
            # repeat n times
            value = int(m[1], 0)
            n = int(m[2], 0)
            for _ in range(n):
                mem.storevalue(addr, 16, value)
                addr += 2
        else:
            # single, or list of values.
            for val in value.split(","):
                val = names.resolve(val)
                mem.storevalue(addr, 16, val)
                addr += 2

    mem.enablebreak = True

    print("regs:")
    cpu.dump()
    import signal
    ctrlc_pressed = False
    def handle_ctrlc(*args):
        nonlocal ctrlc_pressed 
        ctrlc_pressed = True
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGINT, handle_ctrlc)

    class Logger:
        def __init__(self, names):
            self.indent = 0
            self.names = names
            self.spstack = []
        def enter(self, pc, cpu):
            print("Entering: %s(a0=%08x, a1=%08x, a2=%08x, a3=%08x)" % (self.names.getname(pc), cpu.a0, cpu.a1, cpu.a2, cpu.a3))
        def leave(self):
            pass
        def updatesp(self, sp):
            if not self.spstack or self.spstack[-1] > sp:
                # enter
                self.spstack.append(sp)

            while self.spstack and self.spstack[-1] < sp:
                # leave
                self.spstack.pop()
            self.indent = len(self.spstack)

    logger = Logger(names)
    cpu.triggers[2] = logger.updatesp

    first = True
    recentcode = b''
    try:
        for ctxt in args.calls:
            call = decodecall(cpu, names, ctxt)
            cpu.pc = call.startaddr
            for reg, value in call.regs:
                cpu.setreg(reg, value)

            while cpu.pc in mem.memory and not ctrlc_pressed:

                cpu.mcycle += 1

                for bc in args.breakcodes:
                    if recentcode[-len(bc):] == bc:
                        print("break on recentcode: %s" % bc.hex())
                        raise BreakException()

                opc = mem.loadvalue(cpu.pc, 16, True)
                if opc&3 == 3:
                    # 32 bit insn
                    opc |= mem.loadvalue(cpu.pc+2, 16, True) << 16
                    cpu.nextpc = unsigned(cpu.pc+4)

                    recentcode += struct.pack("<L", opc)
                else:
                    # 16 bit insn
                    cpu.nextpc = unsigned(cpu.pc+2)

                    recentcode += struct.pack("<H", opc)

                recentcode = recentcode[-16:]

                insn = decoder.decodeinsn(opc)
                if not insn:
                    # TODO: raise invalid instruction exception
                    print("%s%08x: %s: **unimplemented**" % ("  "*logger.indent, cpu.pc, "%08x" % opc if opc&3 == 3 else "    %04x" % opc))
                    cpu.pc = cpu.nextpc
                    continue

                mnemonic, operands = disassemble_instruction(cpu, insn)
                print("%s%08x: %s: %-10s %s" % ("  "*logger.indent, cpu.pc, "%08x" % opc if opc&3 == 3 else "    %04x" % opc, mnemonic, operands))

                newpc = evaluate_instruction(args, cpu, mem, logger, insn)

                if args.trace:
                    if newpc is not None:
                        print("Control transfer from %08x -> %08x" % (cpu.pc, newpc))
                        if cpu.pc == newpc:
                            print("tight loop detected")
                            raise BreakException()

                cpu.pc = newpc if newpc is not None else cpu.nextpc

                # test breakaddr at the end of the loop, so we execute the addr at least once.
                if cpu.pc in args.breakaddrs:
                    print("pc=%08x, in breakaddrs list" % cpu.pc)
                    raise BreakException()

    except BreakException:
        pass
    except KeyboardInterrupt:
        print("interrupted with ctrl-c")
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    print("memory:")
    mem.dump()
    print("regs:")
    cpu.dump()


def disassemble(decoder, names, args):
    """
    Load memory and images, then disassemble the code range specified
    using --startaddr and --endaddr.
    """
    cpu = CPU(args)
    mem = Memory(args)
    if args.insn:
        mem.loadprogram(args.startaddr, args.insn)

    for filename in args.heximages:
        baseofs = 0
        if filename.find('test_0x8000')>=0:
            baseofs = 0x80000000
        for ofs, data in decodehex(open(filename)):
            mem.loadimage(baseofs+ofs, data)

    for arg in args.images:
        filename, loadofs = arg.split('@', 1)
        loadofs = names.resolve(loadofs)
        data = open(filename, "rb").read()
        mem.loadimage(loadofs, data)

    for arg in args.memory:
        addr, value = arg.split('=', 1)
        addr = names.resolve(addr)

        if m := re.match(r'^(\w+)\.\.(\w+)(?::(\w+))?$', value):
            # range
            lo = int(m[1], 0)
            hi = int(m[2], 0)
            step = int(m[3], 0) if m[3] else 1
            for value in range(lo, hi+1, step):
                mem.storevalue(addr, 32, value)
                addr += 4
        elif m := re.match(r'^(\w+)\*(\w+)$', value):
            # repeat n times
            value = int(m[1], 0)
            n = int(m[2], 0)
            for _ in range(n):
                mem.storevalue(addr, 32, value)
                addr += 4
        else:
            # single, or list of values.
            for val in value.split(","):
                val = names.resolve(val)
                mem.storevalue(addr, 32, val)
                addr += 4

    cpu.pc = args.startaddr
    while cpu.pc in mem.memory:
        if args.endaddr is not None and cpu.pc == args.endaddr:
            break
        opc = mem.loadvalue(cpu.pc, 16, True)
        if opc&3 == 3:
            # 32 bit insn
            opc |= mem.loadvalue(cpu.pc+2, 16, True) << 16
            cpu.nextpc = unsigned(cpu.pc+4)
        else:
            # 16 bit insn
            cpu.nextpc = unsigned(cpu.pc+2)

        insn = decoder.decodeinsn(opc)

        if insn:
            mnemonic, operands = disassemble_instruction(cpu, insn)
            print("%08x: %s: %-10s %s" % (cpu.pc, "%08x" % opc if opc&3 == 3 else "    %04x" % opc, mnemonic, operands))
        else:
            print("%08x: %s: **unimplemented**" % (cpu.pc, "%08x" % opc if opc&3 == 3 else "    %04x" % opc))
        cpu.pc = cpu.nextpc


def main():
    import argparse
    class ArrayArg(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super(ArrayArg, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            arr = getattr(namespace, self.dest)
            if arr is None:
                arr = []
                setattr(namespace, self.dest, arr)
            arr.append( values )
 
    parser = argparse.ArgumentParser(description='Riscv instruction decoder')
    parser.add_argument('--trace', action='store_true', help='trace simulation execution')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debugdecoder', action='store_true', help='output details of insn decoding')
    parser.add_argument('--hex', '-x', action='store_true', help='args are hex numbers')
    parser.add_argument('--bytes', '-b', action='store_true', help='args are byte lists, implies hex')
    parser.add_argument('--simulate', '-s', action='store_true', help='run simulator with the given instructions.')
    parser.add_argument('--disasm', '-d', action='store_true', help='disassemble.')

    # breaks
    parser.add_argument('--breakonscall', action='store_true', help='break when encountering an scall insn.')
    parser.add_argument('--breakonunimplemented', action='store_true', help='break when encountering an unimplemented insn.')
    parser.add_argument('--breakonstore', dest='breakstores', action=ArrayArg, type=str, help='break when writing to address.', metavar='ADDR', default=[])
    parser.add_argument('--breakonaddress', dest='breakaddrs', action=ArrayArg, type=str, help='break when reaching this address.', metavar='ADDR', default=[])
    parser.add_argument('--breakoncode', dest='breakcodes', action=ArrayArg, type=str, help='break when encountering code bytes.', metavar='BYTES', default=[])

    # disasm
    parser.add_argument('--startaddr', type=str, help='disassemble from.', metavar='OFS', default='0')
    parser.add_argument('--endaddr', type=str, help='disassemble until.', metavar='OFS', default=None)

    # sim
    parser.add_argument('--call', '-f', dest='calls', action=ArrayArg, type=str, help='call function with args.', metavar='fn(arg...)', default=[])

    # image load
    parser.add_argument('--loadimage', '-l', dest='images', action=ArrayArg, type=str, help='loadimage at offset.', metavar='filename@OFS', default=[])
    parser.add_argument('--heximage', dest='heximages', action=ArrayArg, type=str, help='load hex(verilogtxt) image.', metavar='filename', default=[])

    # custom reg/mem edits
    parser.add_argument('--setreg', '-r', dest='registers', action=ArrayArg, type=str, help='set register values', metavar='regname=value', default=[])
    parser.add_argument('--setcsreg', '-c', dest='csregisters', action=ArrayArg, type=str, help='set csregister values', metavar='regname=value', default=[])
    parser.add_argument('--setmem', '-m', dest='memory', action=ArrayArg, type=str, metavar='addr=value', default=[], help='set memory. either a symbol, single value, range..range of values, or value*count')
    parser.add_argument('--sethalf', dest='half', action=ArrayArg, type=str, metavar='addr=value', default=[], help='set memory halfwords. either a symbol, single value, range..range of values, or value*count')
    parser.add_argument('--setpriv', type=str, help='set initial privilege level')
    parser.add_argument('--randomize', action='store_true', help='use random value instead of 0 for uninitialized memory.')

    # readable output
    parser.add_argument('--loadnames', type=str, help='load symbols for the current image')

    # manually specified insn.
    parser.add_argument('insn', type=str, nargs='*')
    args = parser.parse_args()

    def convertarg(a):
        """ convert to byte sequence """
        if args.bytes:
            #  bytes as-is
            return bytes.fromhex(a)
        elif args.hex:
            #  as hexnumbers which need to be converted to little-endian
            return bytes.fromhex(a)[::-1]
        else:
            x = int(a, 0)
            if x&3 == 3:
                return x.to_bytes(4, 'little')
            else:
                return x.to_bytes(2, 'little')

    args.insn = [convertarg(a) for a in args.insn]

    decoder = InstructionDecoder(args)
    CPU._csrnames = decoder.csr_regs

    names = NameResolver()
    if args.loadnames:
        names.load(args.loadnames)

    args.breakstores = set(names.resolve(_) for _ in args.breakstores)
    args.breakaddrs = set(names.resolve(_) for _ in args.breakaddrs)
    args.breakcodes = set(bytes.fromhex(_) for _ in args.breakcodes)
    args.startaddr = names.resolve(args.startaddr)
    args.endaddr = names.resolve(args.endaddr)

    args.setpriv = PRIV.decode(args.setpriv)

    if args.simulate:
        simulate(decoder, names, args)
    elif args.disasm:
        disassemble(decoder, names, args)
    else:
        decoder.analyzeopcodes(args.insn)


if __name__=='__main__':
    main()
