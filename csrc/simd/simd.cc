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

struct SIMD {
  static std::array<int, 4> vecAdd(
    // prob important to do more than one vec add?
    // because calling into binding is expensive?
    int size,
    // std::array or std::vector supported here
    // also can use nbind buffer which requires no memcpy
    std::array<int,4> a,
    std::array<int,4> b
    //nbind::Buffer a,
    //nbind::Buffer b,
    //nbind::Buffer c,
  ) {

    // https://www.cs.virginia.edu/~cr4bd/3330/F2018/simdref.html
    /*__m256i a = _mm256_loadu_si256((__m256i_u*) array);
    __m256i b = _mm256_loadu_si256((__m256i_u*) array);

    // takes two __m256i types  and treats are 32 bit ints then adds them
    __m256i c = _mm256_add_epi32(a, b);*/

    // loadu -> 32 bit(?), si -> signed int, 128 -> 128 bits
    __m128i vecA = _mm_loadu_si128((__m128i*)&(a[0]));
    __m128i vecB = _mm_loadu_si128((__m128i*)&(b[0]));

    // mm -> 128 bits, epi32 -> 32 bit arithmetic
    __m128i vecC = _mm_add_epi32(vecA, vecB);
    //print128_num(vecC);

    // store into a buffer and return it
    std::array<int,4> c;
    _mm_storeu_si128((__m128i*)&(c[0]), vecC);
    return c;

    

  }
};

#include "nbind/nbind.h"

NBIND_CLASS(SIMD) {
    method(vecAdd);
}
