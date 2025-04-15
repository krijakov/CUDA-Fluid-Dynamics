#include "fluid_sim/simulation.hpp"

void simulate_step_gpu(Particle* particles, Velocity* velocities, int count, float dt);

void simulate_step(Particle* particles, Velocity* velocities, int count, float dt) {
    // Call the GPU simulation function
    simulate_step_gpu(particles, velocities, count, dt);
}