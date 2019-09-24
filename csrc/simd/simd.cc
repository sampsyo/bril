#include <iostream>
#include <array>

// vector support
#include <smmintrin.h>
#include <immintrin.h>

#include "nbind/api.h"

// pretend like there are really big vector lengths to offset comm overhead
//#define BIG_VECTOR 1

void print128_num(__m128i var)
{
    uint32_t *val = (uint32_t*) &var;
    std::cout << "Numerical: ";
    for (int i = 0; i < 4; i++) {
      std::cout << val[i] << " ";
    }
    std::cout << "\n";

    //printf("Numerical: %i %i %i %i\n", 
    //       val[0], val[1], val[2], val[3]);
}


struct SIMD {
  static void vecAdd(
    // std::array or std::vector supported here
    // also can use nbind buffer which requires no memcpy
    nbind::Buffer aBuf,
    nbind::Buffer bBuf,
    nbind::Buffer cBuf
  ) {

    unsigned char *a = aBuf.data();
    unsigned char *b = bBuf.data();
    unsigned char *c = cBuf.data();

    // going to lose performance here because doing twice as many vec loads as needed
    __m128i vecA = _mm_loadu_si128((__m128i*)a);
    __m128i vecB = _mm_loadu_si128((__m128i*)b);

    __m128i vecC = _mm_add_epi32(vecA, vecB);

    // going to lose performance here
    _mm_storeu_si128((__m128i*)c, vecC);
  }

  static void vecLoad(
    nbind::Buffer tsMem,
    nbind::Buffer destReg,
    int addr
  ) {
    int *mem = (int*)tsMem.data();
    unsigned char *reg = destReg.data();

    // loadu -> 32 bit(?), si -> signed int, 128 -> 128 bits
    __m128i vec = _mm_loadu_si128((__m128i*)&(mem[addr]));
    _mm_storeu_si128((__m128i*)reg, vec);
  }

  static void vecStore(
    nbind::Buffer tsMem,
    nbind::Buffer srcReg,
    int addr
  ) {
    int *mem = (int*)tsMem.data();
    unsigned char *reg = srcReg.data();
    __m128i src = _mm_loadu_si128((__m128i*)reg);
    _mm_storeu_si128((__m128i*)&(mem[addr]), src);
  }

  static void vecAdd_serial(
    // std::array or std::vector supported here
    // also can use nbind buffer which requires no memcpy?? is it faster in typescript?
    nbind::Buffer aBuf,
    nbind::Buffer bBuf,
    nbind::Buffer cBuf
  ) {

    int *a = (int*)aBuf.data();
    int *b = (int*)bBuf.data();
    int *c = (int*)cBuf.data();

    for (int i = 0; i < 4; i++) {
      c[i] = a[i] + b[i];
    }
  }

  static void vecLoad_serial(
    nbind::Buffer tsMem,
    nbind::Buffer destReg,
    int addr
  ) {
    int *mem = (int*)tsMem.data();
    int *reg = (int*)destReg.data();
    for (int i = 0; i < 4; i++) {
      reg[i] = mem[addr + i];
    }
  }

  static void vecStore_serial(
    nbind::Buffer tsMem,
    nbind::Buffer srcReg,
    int addr
  ) {
    int *mem = (int*)tsMem.data();
    int *reg = (int*)srcReg.data();
    for (int i = 0; i < 4; i++) {
      mem[addr + i] = reg[i];
    }
  }
};

#include "nbind/nbind.h"

NBIND_CLASS(SIMD) {
    method(vecAdd);
    method(vecLoad);
    method(vecStore);
    method(vecAdd_serial);
    method(vecLoad_serial);
    method(vecStore_serial);
}
