#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void priint(int64_t i)
{
  printf("%ld", i);
}

void pribool(int i)
{
  printf("%s", i ? "true" : "false");
}

void prifloat(double d)
{
  printf("%.17g", d);
}


int64_t int_of_string(const char *str)
{
  return strtol(str, 0, 0);
}

int bool_of_string(const char *str)
{
  if(strcmp(str, "true") == 0) return 1;
  else if(strcmp(str, "false") == 0) return 0;
  fprintf(stderr, "%s should be a boolean!\n", str);
  exit(1);
}

double float_of_string(const char *str)
{
  return strtod(str, 0);
}
