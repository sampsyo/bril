#include <ostream>

namespace bril {

inline int utf8Encode(char32_t c, char* tmp) {
  if (c <= 0x7F) {
    tmp[0] = static_cast<char>(c);
    tmp[1] = '\0';
    return 1;
  } else if (c <= 0x7FF) {
    tmp[0] = static_cast<char>(0xC0 | (c >> 6));
    tmp[1] = static_cast<char>(0x80 | (c & 0x3F));
    tmp[2] = '\0';
    return 2;
  } else if (c <= 0xFFFF) {
    tmp[0] = static_cast<char>(0xE0 | (c >> 12));
    tmp[1] = static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    tmp[2] = static_cast<char>(0x80 | (c & 0x3F));
    tmp[3] = '\0';
    return 3;
  } else if (c <= 0x10FFFF) {
    tmp[0] = static_cast<char>(0xF0 | (c >> 18));
    tmp[1] = static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    tmp[2] = static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    tmp[3] = static_cast<char>(0x80 | (c & 0x3F));
    tmp[4] = '\0';
    return 4;
  }
  return 0;
}

inline std::ostream& operator<<(std::ostream& os, char32_t c) {
  char tmp[5];
  utf8Encode(c, tmp);
  os << tmp;
  return os;
}

inline std::string toString(char32_t c) {
  char tmp[5];
  utf8Encode(c, tmp);
  return tmp;
}
}  // namespace bril