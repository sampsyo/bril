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
    //std::array<int,4> a,
    //std::array<int,4> b
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
    cBuf.commit();

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
    destReg.commit();
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
    tsMem.commit();
  }
};

#include "nbind/nbind.h"

NBIND_CLASS(SIMD) {
    method(vecAdd);
    method(vecLoad);
    method(vecStore);
}
