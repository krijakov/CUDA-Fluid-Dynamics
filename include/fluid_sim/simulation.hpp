/**
 * @file    simulation.hpp
 * @brief   Defines the high-level simulation functions.
 *
 * This file contains the declaration of the simulate_step function, which is responsible for 
 * stepping the particles forward in time.
 *
 * Notes:
 *     - [important design decisions, caveats, etc.]
 */

#pragma once

#include "types.hpp" // This contains the base structs.

void simulate_step(Particle* particles, MeshTile* meshes, int N, int M, float dt);