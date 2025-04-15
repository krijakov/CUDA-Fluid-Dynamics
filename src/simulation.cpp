#include "fluid_sim/simulation.hpp"

void simulate_step_gpu(Particle* particles, MeshTile* meshes, int N, int M, float dt);

void simulate_step(Particle* particles, MeshTile* meshes, int N, int M, float dt) {
    // Call the GPU simulation function
    simulate_step_gpu(particles, meshes, N, M, dt);
}