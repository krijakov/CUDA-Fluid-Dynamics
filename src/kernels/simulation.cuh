#pragma once
#include "fluid_sim/simulation.hpp"

__global__ void step_kernel(Particle* particles, MeshTile* velocities, int N, int M, float dt);



