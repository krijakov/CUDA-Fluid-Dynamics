/**
 * @file    simulation.cu
 * @brief   Kernel functions and core logic.
 *
 *
 *
 * Notes:
 *     - Threading design: N particle, then M mesh tiles (mesh tiles after particles!!!)
 */

#include "simulation.cuh"

int BLOCKSIZE = 256; // Number of threads per block, from config later!!

// Calculates particle's new velocity and moves them:
__global__ void update_particle(
    Particle *particles, // particles state at the start of the timestep, these get updated here
    MeshTile *meshes,    // mesh state at the start of the timestep
    Params *parameters   // global params
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = parameters->dt;

    if (idx < parameters->num_particles)
    {
        Particle &p = particles[idx];
        // I. Update the particle's velocity based on the mesh tile it is in

        // TODO: fill in force calculation, requires mesh neighbour lookup (spatial hashing)

        // II. Update the particle's position based on its new velocity

        p.posx += p.vx * dt;
        p.posy += p.vy * dt;
        p.posz += p.vz * dt;

        // III. Update the particle's new mesh state

        // TODO: fill in mesh tile update, requires atomic actions (add)
    }
}

__global__ void update_mesh(
    Particle *particles, // particles state at the end of the timestep
    MeshTile *meshes,    // mesh state at the start of the timestep, these get updated here
    Params *parameters   // global params
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < parameters->num_tiles)
    {
        MeshTile &m = meshes[idx];
        // Use equation of state to determine the pressure
        // Assuming a simple equation of state: P = rho * c^2 with c = 1.0
        m.pressure = m.N_particle / m.volume;
    }
}

void simulate_step_gpu(Particle *particles, MeshTile *meshes, Params *parameters)
{
    Particle *d_particles; // particle -> mesh
    MeshTile *d_meshes;    // mesh -> particle
    Params *d_parameters;

    int N = parameters->num_particles;
    int M = parameters->num_tiles;

    // Allocate device memory
    size_t size = N * sizeof(Particle);
    size_t mesh_size = M * sizeof(MeshTile);
    size_t parameters_size = sizeof(Params);

    cudaMalloc(&d_particles, size);
    cudaMalloc(&d_meshes, mesh_size);
    cudaMalloc(&d_parameters, parameters_size);

    cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_meshes, meshes, mesh_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_parameters, parameters, parameters_size, cudaMemcpyHostToDevice);

    for (int t = 0; t < parameters->Nstep; ++t)
    {
        // Launch kernels:

        // Force calculation & particle update:
        update_particle<<<(N + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(
            d_particles, d_meshes, d_parameters);
        cudaDeviceSynchronize();

        // Mesh update:
        int mesh_blocks = (M + BLOCKSIZE - 1) / BLOCKSIZE;
        update_mesh<<<mesh_blocks, BLOCKSIZE>>>(d_particles, d_meshes, d_parameters);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(particles, d_particles, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(meshes, d_meshes, mesh_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(parameters, d_parameters, parameters_size, cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_meshes);
    cudaFree(d_parameters);
}