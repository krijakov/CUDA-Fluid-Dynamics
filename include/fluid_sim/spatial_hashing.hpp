/**
 * @file    spatial_hashing.cu
 * @brief   Mesh ID - position mapping and spatial hashing for various geometries.
 *
 *
 *
 * Notes:
 *
 */

#pragma once
#include "fluid_sim/cuda_compat.hpp"
#include "fluid_sim/types.hpp"
#include <cuda_runtime.h>

// Auxiliary functions:
__host__ __device__ inline bool is_in_bounds(int3 idx, const Params *params)
{
    // Check if the position is within the bounds of the simulation domain
    return (idx.x >= 0 && idx.x < params->N_tiles_x &&
            idx.y >= 0 && idx.y < params->N_tiles_y &&
            idx.z >= 0 && idx.z < params->N_tiles_z);
}

__host__ __device__ inline int3 hash_to_index(int hash, const Params *params)
{
    // Convert a 1D hash to a 3D index
    int x = hash % params->N_tiles_x;
    int y = (hash / params->N_tiles_x) % params->N_tiles_y;
    int z = hash / (params->N_tiles_x * params->N_tiles_y);
    return make_int3(x, y, z);
}

__host__ __device__ inline int3 add_int3(int3 a, int3 b)
{
    // Add two 3D indices
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __constant__ int3 cubic_directions[6] = {
    {-1, 0, 0},
    {1, 0, 0},
    {0, -1, 0},
    {0, 1, 0},
    {0, 0, -1},
    {0, 0, 1}};

// Spatial hashing:

inline __host__ __device__ int cubic_grid_hash(float3 pos, const Params *params)
{
    // Calculate the mesh ID based on the position and grid size
    // NOTE: origin at the bottom left corner of the grid
    int ix = (int)(pos.x / params->tile_size_x); // indexing starts at 0
    int iy = (int)(pos.y / params->tile_size_y);
    int iz = (int)(pos.z / params->tile_size_z);

    // Ensure the indices are within bounds:
    if (!is_in_bounds(make_int3(ix, iy, iz), params) || pos.x < 0 || pos.y < 0 || pos.z < 0)
    {
        return -1; // Out of bounds
    }
    else
    {
        return ix + iy * params->N_tiles_x + iz * params->N_tiles_x * params->N_tiles_y; // might want to precompute offsets later
    };
}

// Spatial hashing:
inline __host__ __device__ int position_to_mesh_id(float3 pos, const Params *params)
{
    switch (params->geometry_type)
    {
    case GEOM_CUBIC:
        return cubic_grid_hash(pos, params);
    // Other geometries can be added here
    default:
        return -1; // Invalid geometry type
    }
}

inline __host__ __device__ void cubic_get_neighbours(int mesh_id, int *neighbour_ids, const Params *params)
{

    for (int i = 0; i < 6; i++)
    {
        int3 idx = add_int3(hash_to_index(mesh_id, params), cubic_directions[i]);
        if (is_in_bounds(idx, params))
        {
            neighbour_ids[i] = idx.x + idx.y * params->N_tiles_x + idx.z * params->N_tiles_x * params->N_tiles_y;
        }
        else
        {
            neighbour_ids[i] = -1; // Out of bounds
        }
    }
}

// Get neighbouring mesh IDs for a given mesh ID:
inline __host__ __device__ void get_neighbour_mesh_ids(int mesh_id, int *neighbour_ids, const Params *params)
{
    switch (params->geometry_type)
    {
    case GEOM_CUBIC:
        cubic_get_neighbours(mesh_id, neighbour_ids, params);
        return;
    // Other geometries can be added here
    default:
        return;
    }
}