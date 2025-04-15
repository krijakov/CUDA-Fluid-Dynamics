#pragma once
#include <vector>

struct Particle {
    float x, y, z; // Position
};

struct Velocity {
    float vx, vy, vz; // Velocity
};

void simulate_step(Particle* particles, Velocity* velocities, int count, float dt);