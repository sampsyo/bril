#include <iostream>
#include <array>

// vector support
#include <smmintrin.h>
#include <immintrin.h>

#include "nbind/api.h"

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

// can these be saved in the vector register file?
// possible to specify register keyword here?
static __m128i _vecRegs[4];

struct SIMD {
  static void vecAdd(
    // prob important to do more than one vec add?
    // because calling into binding is expensive?
    int size,
    // std::array or std::vector supported here
    // also can use nbind buffer which requires no memcpy
    //std::array<int,4> a,
    //std::array<int,4> b
    //nbind::Buffer a,
    //nbind::Buffer b,
    //nbind::Buffer c,
    int index0,
    int index1,
    int index2
  ) {

    // https://www.cs.virginia.edu/~cr4bd/3330/F2018/simdref.html
    /*__m256i a = _mm256_loadu_si256((__m256i_u*) array);
    __m256i b = _mm256_loadu_si256((__m256i_u*) array);

    // takes two __m256i types  and treats are 32 bit ints then adds them
    __m256i c = _mm256_add_epi32(a, b);*/

    // loadu -> 32 bit(?), si -> signed int, 128 -> 128 bits
    //__m128i vecA = _mm_loadu_si128((__m128i*)&(a[0]));
    //__m128i vecB = _mm_loadu_si128((__m128i*)&(b[0]));

    // mm -> 128 bits, epi32 -> 32 bit arithmetic
    //_vecRegs[index0] = _mm_add_epi32(_vecRegs[index1], _vecRegs[index2]);
    _vecRegs[index0] = _mm_add_epi32(_vecRegs[index1], _vecRegs[index2]);
    //print128_num(vecC);

    // store into a buffer and return it
    //std::array<int,4> c;
    //_mm_storeu_si128((__m128i*)&(c[0]), vecC);
    //return c;
  }

  static void vecLoad(
    int size,
    nbind::Buffer tsMem,
    int index,
    int addr
  ) {
    int *data = (int*)tsMem.data();

    // loadu -> 32 bit(?), si -> signed int, 128 -> 128 bits
    _vecRegs[index] = _mm_loadu_si128((__m128i*)&(data[addr]));
    //__m128i vecB = _mm_loadu_si128((__m128i*)&(b[0]));

    // just keep in a c++ variable, hopefully a vector register

    //tsMem.commit();
  }

  static void vecStore(
    int size,
    nbind::Buffer tsMem,
    int index,
    int addr
  ) {
    int *data = (int*)tsMem.data();
    _mm_storeu_si128((__m128i*)&(data[addr]), _vecRegs[index]);
    tsMem.commit();
  }
};

#include "nbind/nbind.h"

NBIND_CLASS(SIMD) {
    method(vecAdd);
    method(vecLoad);
    method(vecStore);
}
