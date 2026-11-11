/* QEMU microbit (nRF51822, Cortex-M0). The chip has 256K of flash and 16K of
   RAM; the sweeps image needs more of both (the conv sweep's code, the LSTM
   sweep's stack), so scripts/qemu-bench.sh gives QEMU's nRF51 1M and 256K
   (-global nrf51-soc.flash-size / sram-size). Only the instruction set is
   under test here. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 1M
  RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}
