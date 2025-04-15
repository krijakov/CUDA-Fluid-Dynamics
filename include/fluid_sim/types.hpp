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

struct Particle {
    float posx, posy, posz; // Position
    float vx, vy, vz; // Velocity
    float mass; // Mass
    int mesh_id; // Mesh ID in which the particle is located
};

struct MeshTile {
    float posx, posy, posz; // Position of the center
    float size; // Size
    int id; // ID

    // Phyical properties
    float density; // Density
    float avg_velx, avg_vely, avg_velz; // Average velocity
    float pressure; // Pressure
};
