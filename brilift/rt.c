#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void _bril_print_int(int64_t i) {
    printf("%" PRId64, i);
}

void _bril_print_bool(char i) {
    if (i) {
        printf("true");
    } else {
        printf("false");
    }
}

void _bril_print_float(double f) {
    if (isnan(f)) {
        printf("NaN");
    } else if (isinf(f)) {
        if (f < 0) {
            printf("-Infinity");
        } else {
            printf("Infinity");
        }
    } else {
        printf("%.17lf", f);
    }
}

void _bril_print_sep() {
    printf(" ");
}

void _bril_print_end() {
    printf("\n");
}

int64_t _bril_parse_int(char **args, int64_t idx) {
    char *arg = args[idx];
    int64_t res;
    sscanf(arg, "%" SCNd64, &res);
    return res;
}

char _bril_parse_bool(char **args, int64_t idx) {
    char *arg = args[idx];
    return !strcmp(arg, "true");
}

double _bril_parse_float(char **args, int64_t idx) {
    char *arg = args[idx];
    double res;
    sscanf(arg, "%lf", &res);
    return res;
}

void *_bril_alloc(int64_t size, int64_t bytes) {
    return malloc(size * bytes);
}

void _bril_free(void *ptr) {
    free(ptr);
}
