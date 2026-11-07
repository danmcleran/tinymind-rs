//! Bare-metal board support, ported from tinymind's
//! benchmarks/framework_comparison_f411/board/{board.cpp, board_l031k6.cpp} so
//! the clocks, flash wait states and timing match its C++ measurements.
//! Register addresses are from RM0383 (F411) and RM0377 (L031).

use core::fmt;
use core::ptr::{read_volatile, write_volatile};

fn reg(address: u32) -> *mut u32 {
    address as *mut u32
}

fn read(address: u32) -> u32 {
    // SAFETY: a fixed, aligned peripheral register address from the reference
    // manual; volatile access is how memory-mapped registers are used.
    unsafe { read_volatile(reg(address)) }
}

fn write(address: u32, value: u32) {
    // SAFETY: as in `read`.
    unsafe { write_volatile(reg(address), value) }
}

fn modify(address: u32, f: impl FnOnce(u32) -> u32) {
    write(address, f(read(address)));
}

#[cfg(feature = "f411")]
mod chip {
    use super::{modify, read, write};

    pub const NAME: &str = "NUCLEO-F411RE (Cortex-M4F)";
    pub const MHZ: u32 = 100;

    const RCC_CR: u32 = 0x4002_3800;
    const RCC_PLLCFGR: u32 = 0x4002_3804;
    const RCC_CFGR: u32 = 0x4002_3808;
    const RCC_AHB1ENR: u32 = 0x4002_3830;
    const RCC_APB1ENR: u32 = 0x4002_3840;
    const FLASH_ACR: u32 = 0x4002_3C00;
    const PWR_CR: u32 = 0x4000_7000;
    const GPIOA_MODER: u32 = 0x4002_0000;
    const GPIOA_AFRL: u32 = 0x4002_0020;
    const USART2_SR: u32 = 0x4000_4400;
    const USART2_DR: u32 = 0x4000_4404;
    const USART2_BRR: u32 = 0x4000_4408;
    const USART2_CR1: u32 = 0x4000_440C;
    const DEMCR: u32 = 0xE000_EDFC;
    const DWT_CTRL: u32 = 0xE000_1000;
    const DWT_CYCCNT: u32 = 0xE000_1004;

    /// HSI 16 MHz -> PLL (M=16, N=400, P=4) -> 100 MHz; APB1 50 MHz; voltage
    /// scale 1; three wait states with prefetch and I/D caches (ART).
    pub fn init() {
        modify(RCC_APB1ENR, |v| v | (1 << 28));
        modify(PWR_CR, |v| (v & !(3 << 14)) | (3 << 14));
        write(FLASH_ACR, 3 | (1 << 8) | (1 << 9) | (1 << 10));
        while read(FLASH_ACR) & 0xF != 3 {}
        write(RCC_PLLCFGR, 16 | (400 << 6) | (1 << 16) | (8 << 24));
        modify(RCC_CR, |v| v | (1 << 24));
        while read(RCC_CR) & (1 << 25) == 0 {}
        modify(RCC_CFGR, |v| (v & !(7 << 10)) | (4 << 10));
        modify(RCC_CFGR, |v| (v & !3) | 2);
        while (read(RCC_CFGR) >> 2) & 3 != 2 {}

        // USART2 on PA2/PA3 (AF7), 115200 baud from PCLK1 = 50 MHz.
        modify(RCC_AHB1ENR, |v| v | 1);
        modify(RCC_APB1ENR, |v| v | (1 << 17));
        modify(GPIOA_MODER, |v| (v & !(0xF << 4)) | (0xA << 4));
        modify(GPIOA_AFRL, |v| (v & !(0xFF << 8)) | (0x77 << 8));
        write(USART2_BRR, (27 << 4) | 2);
        write(USART2_CR1, (1 << 13) | (1 << 3));

        // DWT cycle counter.
        modify(DEMCR, |v| v | (1 << 24));
        write(DWT_CYCCNT, 0);
        modify(DWT_CTRL, |v| v | 1);
    }

    pub fn cycles() -> u32 {
        read(DWT_CYCCNT)
    }

    pub fn put_byte(b: u8) {
        while read(USART2_SR) & (1 << 7) == 0 {}
        write(USART2_DR, u32::from(b));
    }

    pub fn flush() {
        while read(USART2_SR) & (1 << 6) == 0 {}
    }
}

#[cfg(feature = "l031")]
mod chip {
    use super::{modify, read, write};
    use core::sync::atomic::{AtomicU32, Ordering};
    use cortex_m_rt::exception;

    pub const NAME: &str = "NUCLEO-L031K6 (Cortex-M0+)";
    pub const MHZ: u32 = 32;

    const RCC_CR: u32 = 0x4002_1000;
    const RCC_CFGR: u32 = 0x4002_100C;
    const RCC_IOPENR: u32 = 0x4002_102C;
    const RCC_APB1ENR: u32 = 0x4002_1038;
    const FLASH_ACR: u32 = 0x4002_2000;
    const PWR_CR: u32 = 0x4000_7000;
    const PWR_CSR: u32 = 0x4000_7004;
    const GPIOA_MODER: u32 = 0x5000_0000;
    const GPIOA_AFRL: u32 = 0x5000_0020;
    const USART2_CR1: u32 = 0x4000_4400;
    const USART2_BRR: u32 = 0x4000_440C;
    const USART2_ISR: u32 = 0x4000_441C;
    const USART2_TDR: u32 = 0x4000_4428;
    const SYST_CSR: u32 = 0xE000_E010;
    const SYST_RVR: u32 = 0xE000_E014;
    const SYST_CVR: u32 = 0xE000_E018;
    const SCB_ICSR: u32 = 0xE000_ED04;
    const PENDSTSET: u32 = 1 << 26;
    const SYSTICK_MAX: u32 = 0x00FF_FFFF;

    static WRAPS: AtomicU32 = AtomicU32::new(0);

    #[exception]
    fn SysTick() {
        // The only writer; ARMv6-M has no atomic read-modify-write.
        WRAPS.store(
            WRAPS.load(Ordering::Relaxed).wrapping_add(1),
            Ordering::Relaxed,
        );
    }

    /// HSI16 -> PLL (x4, /2) -> 32 MHz; voltage range 1; one wait state with
    /// prefetch. M0+ has no DWT, so SysTick runs over its full 24-bit range at
    /// the core clock and its wrap interrupt carries the upper bits.
    pub fn init() {
        modify(RCC_APB1ENR, |v| v | (1 << 28));
        modify(PWR_CR, |v| (v & !(3 << 11)) | (1 << 11));
        while read(PWR_CSR) & (1 << 4) != 0 {}
        modify(FLASH_ACR, |v| v | 1 | (1 << 1));
        while read(FLASH_ACR) & 1 == 0 {}
        modify(RCC_CR, |v| v | 1);
        while read(RCC_CR) & (1 << 2) == 0 {}
        modify(RCC_CFGR, |v| {
            (v & !((1 << 16) | (0xF << 18) | (3 << 22))) | (1 << 18) | (1 << 22)
        });
        modify(RCC_CR, |v| v | (1 << 24));
        while read(RCC_CR) & (1 << 25) == 0 {}
        modify(RCC_CFGR, |v| v | 3);
        while (read(RCC_CFGR) >> 2) & 3 != 3 {}

        // USART2 TX on PA2 (AF4), 115200 baud from PCLK1 = 32 MHz.
        modify(RCC_IOPENR, |v| v | 1);
        modify(RCC_APB1ENR, |v| v | (1 << 17));
        modify(GPIOA_MODER, |v| (v & !(3 << 4)) | (2 << 4));
        modify(GPIOA_AFRL, |v| (v & !(0xF << 8)) | (4 << 8));
        write(USART2_BRR, 278);
        write(USART2_CR1, (1 << 3) | 1);

        write(SYST_RVR, SYSTICK_MAX);
        write(SYST_CVR, 0);
        write(SYST_CSR, 7);
    }

    pub fn cycles() -> u32 {
        // With interrupts masked, a wrap not yet counted shows as a pending
        // SysTick; re-read the counter after seeing it.
        cortex_m::interrupt::free(|_| {
            let mut high = WRAPS.load(Ordering::Relaxed);
            let mut low = read(SYST_CVR);
            if read(SCB_ICSR) & PENDSTSET != 0 {
                low = read(SYST_CVR);
                high = high.wrapping_add(1);
            }
            (high << 24) | (SYSTICK_MAX - low)
        })
    }

    pub fn put_byte(b: u8) {
        while read(USART2_ISR) & (1 << 7) == 0 {}
        write(USART2_TDR, u32::from(b));
    }

    pub fn flush() {
        while read(USART2_ISR) & (1 << 6) == 0 {}
    }
}

pub use chip::{MHZ, NAME, cycles, flush, init};

/// The ST-LINK virtual COM port, as a `core::fmt` sink (`\n` becomes `\r\n`).
pub struct Uart;

impl fmt::Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            if b == b'\n' {
                chip::put_byte(b'\r');
            }
            chip::put_byte(b);
        }
        Ok(())
    }
}

const PAINT: u32 = 0xA5A5_A5A5;

unsafe extern "C" {
    // cortex-m-rt's link.x symbols.
    static mut __sdata: u32;
    static mut __ebss: u32;
    static mut _stack_start: u32;
}

/// Fill the unused stack with a pattern (C++ `board_stack_paint`).
#[inline(never)]
pub fn stack_paint() {
    let sp = cortex_m::register::msp::read();
    // SAFETY: writes only between the end of .bss and 64 bytes below the
    // current stack pointer, memory nothing else uses yet.
    unsafe {
        let mut p = &raw mut __ebss;
        while (p as u32) < sp - 64 {
            p.write_volatile(PAINT);
            p = p.add(1);
        }
    }
}

/// Bytes of stack used since `stack_paint` (C++ `board_stack_used`).
pub fn stack_used() -> u32 {
    // SAFETY: reads the painted region between .bss and the stack top.
    unsafe {
        let top = &raw mut _stack_start as u32;
        let mut p = &raw mut __ebss;
        while (p as u32) < top && p.read_volatile() == PAINT {
            p = p.add(1);
        }
        top - p as u32
    }
}

/// .data + .bss bytes (C++ `board_static_ram`).
pub fn static_ram() -> u32 {
    (&raw mut __ebss as u32) - (&raw mut __sdata as u32)
}
