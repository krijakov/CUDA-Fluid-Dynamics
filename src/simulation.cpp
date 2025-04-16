#include "fluid_sim/simulation.hpp"

void simulate_step_gpu(Particle* particles, MeshTile* meshes, Params* parameters);

void simulate_step(Particle* particles, MeshTile* meshes, Params* parameters) {
    // Call the GPU simulation function
    simulate_step_gpu(particles, meshes, parameters);
}