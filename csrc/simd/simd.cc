#include <string>
#include <iostream>
#include <immintrin.h>  // portable to all x86 compilers

void print128_num(__m128 var)
{
    float *val = (float*) &var;
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
    std::string name
  ) {
    std::cout
      << "Hello, "
      << name << "!\n";

    // vector https://stackoverflow.com/questions/1389712/getting-started-with-intel-x86-sse-simd-instructions
    __m128 vector1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0); // high element first, opposite of C array order.  Use _mm_setr_ps if you want "little endian" element order in the source.
    __m128 vector2 = _mm_set_ps(7.0, 8.0, 9.0, 0.0);

    __m128 sum = _mm_add_ps(vector1, vector2); // result = vector1 + vector 2

    vector1 = _mm_shuffle_ps(vector1, vector1, _MM_SHUFFLE(0,1,2,3));
    // vector1 is now (1, 2, 3, 4) (above shuffle reversed it)
    //return 0;
    print128_num(sum);

  }
};

#include "nbind/nbind.h"

NBIND_CLASS(SIMD) {
    method(vecAdd);
}
