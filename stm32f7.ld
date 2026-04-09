MEMORY
{
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 2048K
  RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 384K  /* The TRUTH */
}

_estack = ORIGIN(RAM) + LENGTH(RAM); /* Top of RAM */
_Min_Stack_Size = 0x4000;

SECTIONS
{
 .text :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector))    /* The "Map" */
    KEEP(*(.text.reset_handler)) /* The "Engine Start" */
    KEEP(*(.text.main))          /* The "Destination" */
    *(.text.startup)        /* Catch any compiler-renamed startup code */
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
    _etext = .;
  } > FLASH

  _sidata = .; /* This captures the Flash address for the next section */

  .data : 
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT > FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  ._user_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } > RAM

    _end = .;
  end = .;
}