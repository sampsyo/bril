#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>

void print_int(int64_t i) {
    printf("%" PRId64, i);
}

void print_bool(char i) {
    if (i) {
        printf("true");
    } else {
        printf("false");
    }
}

void print_sep() {
    printf(" ");
}

void print_end() {
    printf("\n");
}

int64_t parse_int(char **args, int64_t idx) {
    char *arg = args[idx];
    int64_t res;
    sscanf(arg, "%" SCNd64, &res);
    return res;
}

char parse_bool(char **args, int64_t idx) {
    char *arg = args[idx];
    return !!strcmp(arg, "true");
}
