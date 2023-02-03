from riscvsim import InstructionDecoder
import struct

# all the custom opcodes as used by the Titan-M2 chip.

unknownopcodes = [
#    086C   000B464A     
#           bit432  bit65                  U/J type                 I-type       14-12            R/S/B-type

#  This is probably bswap32, like 'REV.W' on arm.  I originally named this: 'google2'
0x0185050B,   #  2  0     a0,a0,0x18  -> (i20:01850 rd:0a | i12:018 rs1:0a rd:0a f3:00 |i7:00 rs2:18 rs1:0a rd:0a f3:00) opc:0b   000A1792,000A1844,000A1898,000A194A  
0x0185858B,   #  2  0     a1,a1,0x18  -> (i20:01858 rd:0b | i12:018 rs1:0b rd:0b f3:00 |i7:00 rs2:18 rs1:0b rd:0b f3:00) opc:0b   000A178E,000A1848,000A1894,000A194E  
0x0187070B,   #  2  0     a4,a4,0x18  -> (i20:01870 rd:0e | i12:018 rs1:0e rd:0e f3:00 |i7:00 rs2:18 rs1:0e rd:0e f3:00) opc:0b   000B0114,000B0132,000C4B50  
0x0187878B,   #  2  0     a5,a5,0x18  -> (i20:01878 rd:0f | i12:018 rs1:0f rd:0f f3:00 |i7:00 rs2:18 rs1:0f rd:0f f3:00) opc:0b   000B0110,000B012E,000BF6C4,000C2828,000C4B38,000C4B4C  
0x0188080B,   #  2  0     a6,a6,0x18  -> (i20:01880 rd:10 | i12:018 rs1:10 rd:10 f3:00 |i7:00 rs2:18 rs1:10 rd:10 f3:00) opc:0b   000A55CC  
0x018A078B,   #  2  0     a5,s4,0x18  -> (i20:018a0 rd:0f | i12:018 rs1:14 rd:0f f3:00 |i7:00 rs2:18 rs1:14 rd:0f f3:00) opc:0b   000B0370  
0x018A0A0B,   #  2  0     s4,s4,0x18  -> (i20:018a0 rd:14 | i12:018 rs1:14 rd:14 f3:00 |i7:00 rs2:18 rs1:14 rd:14 f3:00) opc:0b   000A1D72,000BD032  
0x018A8A8B,   #  2  0     s5,s5,0x18  -> (i20:018a8 rd:15 | i12:018 rs1:15 rd:15 f3:00 |i7:00 rs2:18 rs1:15 rd:15 f3:00) opc:0b   000A1D76  

# 'google0'  type-I   probably: clear/set bit

# gclrbit  formerly called google0a
0x0034140B,   #  2  0     s0,s0, 0x03     -> (i20:00341 rd:08 | i12:003 rs1:08 rd:08 f3:01 |i7:00 rs2:03 rs1:08 rd:08 f3:01) opc:0b   000A8602,000A8750  
0x0044140B,   #  2  0     s0,s0, 0x04     -> (i20:00441 rd:08 | i12:004 rs1:08 rd:08 f3:01 |i7:00 rs2:04 rs1:08 rd:08 f3:01) opc:0b   000A8606  
0x0054140B,   #  2  0     s0,s0, 0x05     -> (i20:00541 rd:08 | i12:005 rs1:08 rd:08 f3:01 |i7:00 rs2:05 rs1:08 rd:08 f3:01) opc:0b   000A8762  
0x0064140B,   #  2  0     s0,s0, 0x06     -> (i20:00641 rd:08 | i12:006 rs1:08 rd:08 f3:01 |i7:00 rs2:06 rs1:08 rd:08 f3:01) opc:0b   000A8798  
0x0074140B,   #  2  0     s0,s0, 0x07     -> (i20:00741 rd:08 | i12:007 rs1:08 rd:08 f3:01 |i7:00 rs2:07 rs1:08 rd:08 f3:01) opc:0b   000A8770  
0x00A5158B,   #  2  0     a1,a0, 0x0a     -> (i20:00a51 rd:0b | i12:00a rs1:0a rd:0b f3:01 |i7:00 rs2:0a rs1:0a rd:0b f3:01) opc:0b   000A0944  
0x01E4140B,   #  2  0     s0,s0, 0x1e     -> (i20:01e41 rd:08 | i12:01e rs1:08 rd:08 f3:01 |i7:00 rs2:1e rs1:08 rd:08 f3:01) opc:0b   000A865C  

# gsetbit   formerly called google0b
0x4024140B,   #  2  0     s0,s0,0x402     -> (i20:40241 rd:08 | i12:402 rs1:08 rd:08 f3:01 |i7:20 rs2:02 rs1:08 rd:08 f3:01) opc:0b   000A866C  
0x4034140B,   #  2  0     s0,s0,0x403     -> (i20:40341 rd:08 | i12:403 rs1:08 rd:08 f3:01 |i7:20 rs2:03 rs1:08 rd:08 f3:01) opc:0b   000A8742  
0x4044140B,   #  2  0     s0,s0,0x404     -> (i20:40441 rd:08 | i12:404 rs1:08 rd:08 f3:01 |i7:20 rs2:04 rs1:08 rd:08 f3:01) opc:0b   000A8754  
0x4054140B,   #  2  0     s0,s0,0x405     -> (i20:40541 rd:08 | i12:405 rs1:08 rd:08 f3:01 |i7:20 rs2:05 rs1:08 rd:08 f3:01) opc:0b   000A8612  
0x4064140B,   #  2  0     s0,s0,0x406     -> (i20:40641 rd:08 | i12:406 rs1:08 rd:08 f3:01 |i7:20 rs2:06 rs1:08 rd:08 f3:01) opc:0b   000A86A4  
0x4074140B,   #  2  0     s0,s0,0x407     -> (i20:40741 rd:08 | i12:407 rs1:08 rd:08 f3:01 |i7:20 rs2:07 rs1:08 rd:08 f3:01) opc:0b   000A861E  

# 'google1'  in the 'custom-1' group   type-R/S/B   probably: clear/set bit
# gclrbit   formerly called google1a
0x0087172B,   #  2  1     a4,a4, 0x08 ?s0 -> (i20:00871 rd:0e | i12:008 rs1:0e rd:0e f3:01 |i7:00 rs2:08 rs1:0e rd:0e f3:01) opc:2b   000A85B6  
0x00A696AB,   #  2  1     a3,a3, 0x0a ?a0 -> (i20:00a69 rd:0d | i12:00a rs1:0d rd:0d f3:01 |i7:00 rs2:0a rs1:0d rd:0d f3:01) opc:2b   000A63FC  
0x00A7172B,   #  2  1     a4,a4, 0x0a ?a0 -> (i20:00a71 rd:0e | i12:00a rs1:0e rd:0e f3:01 |i7:00 rs2:0a rs1:0e rd:0e f3:01) opc:2b   000A5A5A,000A8450  
0x00A797AB,   #  2  1     a5,a5, 0x0a ?a0 -> (i20:00a79 rd:0f | i12:00a rs1:0f rd:0f f3:01 |i7:00 rs2:0a rs1:0f rd:0f f3:01) opc:2b   000A60CC,000A63DE,000A83D8,000A846C,000AA1B8  
0x00D7172B,   #  2  1     a4,a4, 0x0d ?a3 -> (i20:00d71 rd:0e | i12:00d rs1:0e rd:0e f3:01 |i7:00 rs2:0d rs1:0e rd:0e f3:01) opc:2b   000A85CE,000A85DE  
0x00D797AB,   #  2  1     a5,a5, 0x0d ?a3 -> (i20:00d79 rd:0f | i12:00d rs1:0f rd:0f f3:01 |i7:00 rs2:0d rs1:0f rd:0f f3:01) opc:2b   000A2B4E,000A5F7E  
0x00E696AB,   #  2  1     a3,a3, 0x0e ?a4 -> (i20:00e69 rd:0d | i12:00e rs1:0d rd:0d f3:01 |i7:00 rs2:0e rs1:0d rd:0d f3:01) opc:2b   000A63BE  
0x00E797AB,   #  2  1     a5,a5, 0x0e ?a4 -> (i20:00e79 rd:0f | i12:00e rs1:0f rd:0f f3:01 |i7:00 rs2:0e rs1:0f rd:0f f3:01) opc:2b   000A5A1A,000A637E,000A639E,000A879C  
0x010797AB,   #  2  1     a5,a5, 0x10 ?a6 -> (i20:01079 rd:0f | i12:010 rs1:0f rd:0f f3:01 |i7:00 rs2:10 rs1:0f rd:0f f3:01) opc:2b   000A8D0E  
# gsetbit     formerly called google1b
0x4087172B,   #  2  1     a4,a4,0x408 ?s0 -> (i20:40871 rd:0e | i12:408 rs1:0e rd:0e f3:01 |i7:20 rs2:08 rs1:0e rd:0e f3:01) opc:2b   000A8714  
0x40A7172B,   #  2  1     a4,a4,0x40a ?a0 -> (i20:40a71 rd:0e | i12:40a rs1:0e rd:0e f3:01 |i7:20 rs2:0a rs1:0e rd:0e f3:01) opc:2b   000A8404  
0x40A797AB,   #  2  1     a5,a5,0x40a ?a0 -> (i20:40a79 rd:0f | i12:40a rs1:0f rd:0f f3:01 |i7:20 rs2:0a rs1:0f rd:0f f3:01) opc:2b   000A2C7A,000A588A,000A60B4,000A8422,000A8434  
0x40D7172B,   #  2  1     a4,a4,0x40d ?a3 -> (i20:40d71 rd:0e | i12:40d rs1:0e rd:0e f3:01 |i7:20 rs2:0d rs1:0e rd:0e f3:01) opc:2b   000A8724,000A8736  
0x40D797AB,   #  2  1     a5,a5,0x40d ?a3 -> (i20:40d79 rd:0f | i12:40d rs1:0f rd:0f f3:01 |i7:20 rs2:0d rs1:0f rd:0f f3:01) opc:2b   000A6002,000A8B18  
0x40E797AB,   #  2  1     a5,a5,0x40e ?a4 -> (i20:40e79 rd:0f | i12:40e rs1:0f rd:0f f3:01 |i7:20 rs2:0e rs1:0f rd:0f f3:01) opc:2b   000A877A  
0x4106162B,   #  2  1     a2,a2,0x410 ?a6 -> (i20:41061 rd:0c | i12:410 rs1:0c rd:0c f3:01 |i7:20 rs2:10 rs1:0c rd:0c f3:01) opc:2b   000A8D02  

# gbitscan,  I originally named this: google3   - counts lowest zero bits.
0x0005250B,   #  2  0     a0,a0,0x0   -> (i20:00052 rd:0a | i12:000 rs1:0a rd:0a f3:02 |i7:00 rs2:00 rs1:0a rd:0a f3:02) opc:0b   000A8BBE  
0x0007250B,   #  2  0     a0,a4,0x0   -> (i20:00072 rd:0a | i12:000 rs1:0e rd:0a f3:02 |i7:00 rs2:00 rs1:0e rd:0a f3:02) opc:0b   000A5A56  
0x0007A50B,   #  2  0     a0,a5,0x0   -> (i20:0007a rd:0a | i12:000 rs1:0f rd:0a f3:02 |i7:00 rs2:00 rs1:0f rd:0a f3:02) opc:0b   000A63DA,000AA1B4  
0x0007A70B,   #  2  0     a4,a5,0x0   -> (i20:0007a rd:0e | i12:000 rs1:0f rd:0e f3:02 |i7:00 rs2:00 rs1:0f rd:0e f3:02) opc:0b   000A5A16,000A637A,000A639A  
0x0007A78B,   #  2  0     a5,a5,0x0   -> (i20:0007a rd:0f | i12:000 rs1:0f rd:0f f3:02 |i7:00 rs2:00 rs1:0f rd:0f f3:02) opc:0b   000A2B56,000A9CE8  

# grbitscan  probably: get-highest-onebit  - counts highest zero bits
0x400D278B,   #  2  0     a5,s10,0x400-> (i20:400d2 rd:0f | i12:400 rs1:1a rd:0f f3:02 |i7:20 rs2:00 rs1:1a rd:0f f3:02) opc:0b   000A27B2  
]

decoder = InstructionDecoder()
decoder.analyzeopcodes([ struct.pack("<L", _) for _ in unknownopcodes ])
