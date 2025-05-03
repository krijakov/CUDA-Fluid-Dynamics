#pragma once
#include "fluid_sim/cuda_compat.hpp"
#include "fluid_sim/simulation.hpp"

// CUDA only declarations:
__global__ void update_particle(Particle *particles, MeshTile *velocities, Params *parameters);
__global__ void update_mesh(Particle *particles, MeshTile *meshes, Params *parameters);


