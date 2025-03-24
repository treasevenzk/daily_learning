#include <stdio.h>

typedef int (*callback_t)(int, double);

void process_data(int *array, int length, callback_t callback) {
    for (int i = 0; i < length; i++) {
        int result = callback(array[i], (double)array[i] * 0.5);
        printf("C: Callback returned %d\n", result);
    }
}