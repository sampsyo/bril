#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

void print_int(int64_t i) {
    printf("%" PRId64 "\n", i);
}

int64_t parse_int(char **args, int64_t idx) {
    char *arg = args[idx];
    int64_t res;
    sscanf(arg, "%" SCNd64, &res);
    return res;
}
