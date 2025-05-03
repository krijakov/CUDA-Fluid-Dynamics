#pragma once

// If we’re _not_ compiling under nvcc, pretend all CUDA
// qualifiers are empty macros so the host compiler ignores them.
#ifndef __CUDACC__
  #define __device__
  #define __host__
  #define __global__
  #define __constant__
  #define __shared__
#endif