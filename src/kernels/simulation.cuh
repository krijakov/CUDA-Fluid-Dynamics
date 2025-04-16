#pragma once
#include "fluid_sim/simulation.hpp"

__global__ void update_particle(Particle* particles, MeshTile* velocities, Params* parameters);
__global__ void update_mesh(Particle* particles, MeshTile* meshes, Params* parameters);



