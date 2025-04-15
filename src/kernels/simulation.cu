#include "simulation.cuh"

__global__ void step_kernel(Particle* particles, Velocity* velocities, int count, float dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count){
        particles[idx].x += velocities[idx].vx * dt;
        particles[idx].y += velocities[idx].vy * dt;
        particles[idx].z += velocities[idx].vz * dt;
    }
}

void simulate_step_gpu(Particle* particles, Velocity* velocities, int count, float dt){
    Particle* d_particles;
    Velocity* d_velocities;
    size_t size = count * sizeof(Particle);
    size_t vel_size = count * sizeof(Velocity);

    cudaMalloc(&d_particles, size);
    cudaMalloc(&d_velocities, vel_size);

    cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities, vel_size, cudaMemcpyHostToDevice);

    step_kernel<<<(count + 255) / 256, 256>>>(d_particles, d_velocities, count, dt);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, d_velocities, vel_size, cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_velocities);
}