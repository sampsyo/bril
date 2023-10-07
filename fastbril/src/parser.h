#ifndef PARSER_H
#define PARSER_H

#include "libs/json.h"
#include "bril-insns/instrs.h"

/**
 * parses a program...
 */
program_t *parse_program(struct json_object_s *json);
#endif
