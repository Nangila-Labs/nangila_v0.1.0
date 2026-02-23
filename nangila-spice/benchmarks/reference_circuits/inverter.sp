* CMOS Inverter — Reference Circuit for Nangila SPICE
* Basic test case for netlist parsing and single-partition simulation.

.model NMOS NMOS (LEVEL=1 VTO=0.7 KP=110u)
.model PMOS PMOS (LEVEL=1 VTO=-0.7 KP=55u)

* Supply
VDD vdd 0 DC 1.8
VSS vss 0 DC 0

* Input stimulus
VIN in 0 PULSE(0 1.8 0 100p 100p 5n 10n)

* Inverter
M1 out in vdd vdd PMOS W=2u L=0.18u
M2 out in vss vss NMOS W=1u L=0.18u

* Load
CL out 0 10f

.tran 10p 20n
.end
