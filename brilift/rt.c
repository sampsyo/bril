#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

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
    return !!strcmp(arg, "true");
}
