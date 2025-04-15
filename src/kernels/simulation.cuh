#pragma once
#include "fluid_sim/simulation.hpp"

__global__ void step_kernel(Particle* particles, Velocity* velocities, int count, float dt);



