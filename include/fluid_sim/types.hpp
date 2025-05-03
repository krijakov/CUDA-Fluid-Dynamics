/**
 * @file    types.hpp
 * @brief   Defines the basic types used in the fluid simulation.
 *
 * This file contains the definitions of the Particle and Velocity structures.
 *
 * Notes:
 *     - [important design decisions, caveats, etc.]
 */

#pragma once

enum GeometryType {
    GEOM_CUBIC = 0,
    // Placeholders for future geometry types
    GEOM_HEX,
    GEOM_CYLINDRICAL,
    GEOM_SPHERICAL,
};

struct Particle {
    float posx, posy, posz; // Position
    float vx, vy, vz; // Velocity
    float mass; // Mass
    int mesh_id; // Mesh ID in which the particle is located
};

struct MeshTile {
    float posx, posy, posz; // Position of the center
    float size; // Size
    float volume; // Size
    int id; // ID

    // Phyical properties
    float N_particle; // number of particles in the tile
    float velx, vely, velz; // total velocity
    float velx2, vely2, velz2; // total velocity squared
    float pressure; // Pressure
};

// Global parameters of the simulation
struct Params {
    // Bounding box:
    float minx, miny, minz; // Minimum coordinates
    float maxx, maxy, maxz; // Maximum coordinates
    
    // Global parameters:
    int num_tiles; // Total number of tiles 
    int num_particles; // Total number of particles
    int geometry_type; // Geometry type (use the enum GeometryType)

    // Integration parameters:
    float dt; // Time step
    int Nstep; // Number of steps

    // Cubic only, may get moved:
    // NOTE: grid always starts at (0,0,0)
    int N_tiles_x, N_tiles_y, N_tiles_z; // Number of tiles in each direction
    float tile_size_x, tile_size_y, tile_size_z; // Size of each tile
};

