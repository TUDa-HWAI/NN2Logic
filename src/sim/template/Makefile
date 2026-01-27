SRC = $(wildcard *.c) common/syscalls.c

CC = riscv32-unknown-elf-gcc
LINKER_SCRIPT = common/test.ld
CRT = common/crt.S
CFLAGS = -march=rv32imdczicsr -mabi=ilp32d -static -mcmodel=medany -Wall -fvisibility=hidden -nostdlib -nostartfiles -ffreestanding -MMD -O2 

OBJS := ${SRC:.c=.o} ${CRT:.S=.o}
DEPS = $(OBJS:%.o=%.d)

all: test.elf

.PHONY: test.elf
test.elf: $(OBJS) $(LINKER_SCRIPT)
	$(CC) $(CFLAGS) -T $(LINKER_SCRIPT) $(OBJS) -o $@

hybrid.o: hybrid.c
	@echo "Special hybrid treatment"
	$(CC) -march=rv32imdczicsr -mabi=ilp32d -static -mcmodel=medany -Wall -fvisibility=hidden -nostdlib -nostartfiles -ffreestanding -MMD @hybrid-opt.txt -c -o $@ $<

%.o: %.S
	$(CC) $(CFLAGS) -MMD -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -MMD -c -o $@ $<

.PHONY: clean
clean:
	rm -f test.elf $(OBJS) $(DEPS)
