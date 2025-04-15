#include "simulation.cuh"

__global__ void step_kernel(Particle* particles, MeshTile* meshes, int N, int M, float dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N + M){
        // I'm here
        particles[idx].posx += particles[idx].vx * dt;
        particles[idx].posy += particles[idx].vy * dt;
        particles[idx].posz += particles[idx].vz * dt;
    }
}

void simulate_step_gpu(Particle* particles, MeshTile* meshes, int N, int M, float dt){
    Particle* d_particles;
    MeshTile* d_meshes;
    size_t size = N * sizeof(Particle);
    size_t mesh_size = M * sizeof(MeshTile);

    cudaMalloc(&d_particles, size);
    cudaMalloc(&d_meshes, mesh_size);

    cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_meshes, meshes, M, cudaMemcpyHostToDevice);

    step_kernel<<<(N + M + 255) / 256, 256>>>(d_particles, d_meshes, N, M, dt);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(meshes, d_meshes, mesh_size, cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_meshes);
}