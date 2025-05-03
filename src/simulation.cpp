#include "fluid_sim/simulation.hpp"
#include "fluid_sim/cuda_compat.hpp" // ignores the CUDA specific syntax in case of CPU side compilation
#include <vector>

void simulate_step_gpu(Particle *particles, MeshTile *meshes, Params *parameters);

void simulate_step(Particle *particles, MeshTile *meshes, Params *parameters)
{
    // Call the GPU simulation function
    simulate_step_gpu(particles, meshes, parameters);
}