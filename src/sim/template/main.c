#include "network.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/signal.h>

#include "dataset.h"

#define PCOUNT_READ(name, dst) asm volatile("csrr %0, " #name ";" : "=r"(dst))


void print(size_t num) {
	putchar(num / 100000 + 48);
	num %= 100000;

	putchar(num / 10000 + 48);
	num %= 10000;

	putchar(num / 1000 + 48);
	num %= 1000;

	putchar(num / 100 + 48);
	num %= 100;

	putchar(num / 10 + 48);
	num %= 10;

	putchar(num + 48);
}

void pprint(int8_t result) {
	if (result < 0) {
		putchar('-');
		result = -result;
	} else {
		putchar('+');
	}

	putchar(result / 100 + 48);
	putchar((result % 100) / 10 + 48);
	putchar((result % 10) + 48);
}

// actual code
int main() {
	size_t right = 0, wrong = 0;
	int8_t result[NUM_OUT];
	size_t total = 0;
	for (size_t i = 0; i < datasetSize; ++i) {
		int32_t timestamp, timestamp2;
		PCOUNT_READ(mcycle, timestamp);

		network(dataset[i], result);
		PCOUNT_READ(mcycle, timestamp2);
		total += timestamp2 - timestamp;

		// count right/wrong
		uint8_t winClass = 255;
		int8_t max = -127;
		for (size_t c = 0; c < NUM_OUT; ++c) {
			if (result[c] > max) {
				winClass = c;
				max = result[c];
			}
		}

		print(timestamp2 - timestamp);
		putchar(' ');
		if (winClass == target[i]) {
			putchar('T');
			++right;
		} else {
			putchar('F');
			++wrong;
		}
		putchar('\n');

	}

	return 0;
}
