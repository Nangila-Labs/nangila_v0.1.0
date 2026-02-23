* 6T SRAM Cell — Reference Circuit for Nangila SPICE
* Standard 6-transistor SRAM cell for partition boundary testing.

.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=110u)
.model PMOS PMOS (LEVEL=1 VTO=-0.7 KP=55u)

* Supply
VDD vdd 0 DC 1.8

* Wordline
VWL wl 0 PULSE(0 1.8 1n 100p 100p 5n 20n)

* Bitlines (precharged)
VBL  bl  0 DC 1.8
VBLB blb 0 DC 1.8

* Cross-coupled inverters
M1 q  qb vdd vdd PMOS W=0.5u L=0.18u
M2 q  qb 0   0   NMOS W=0.25u L=0.18u
M3 qb q  vdd vdd PMOS W=0.5u L=0.18u
M4 qb q  0   0   NMOS W=0.25u L=0.18u

* Access transistors
M5 bl  wl q  0 NMOS W=0.36u L=0.18u
M6 blb wl qb 0 NMOS W=0.36u L=0.18u

* Parasitic caps
CQ  q  0 1f
CQB qb 0 1f

.ic V(q)=1.8 V(qb)=0

.tran 10p 25n
.end
