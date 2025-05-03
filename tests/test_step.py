#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test for a single timestep of a single particle.
This test is not meant to be comprehensive, but rather to check if the system is working as expected.

"""

import pyfluid

def test_step():
    """
    Test a single step of a single particle.
    """
    
    particles = [pyfluid.Particle()]
    particles[0].posx = 0.0
    particles[0].posy = 0.0
    particles[0].posz = 0.0
    particles[0].vx = 1.0
    particles[0].vy = 0.0
    particles[0].vz = 0.0
    particles[0].mass = 1.0
    particles[0].mesh_id = 1

    meshes = [pyfluid.MeshTile()]
    meshes[0].id = 1
    meshes[0].posx = 0.0
    meshes[0].posy = 0.0
    meshes[0].posz = 0.0

    params = pyfluid.Params()
    params.dt = 0.1
    params.num_particles = 1
    params.num_tiles = 1
    params.geometry_type = 0
    params.Nstep = 1

    x,m = pyfluid.simulate_step(particles, meshes, params)
    
    assert x[0].vx == 1.0
    assert x[0].vy == 0.0
    assert x[0].vz == 0.0
    assert x[0].posx <= 0.1 + 1e-5 and x[0].posx >= 0.1 - 1e-5, f"Position x is not correct: {x[0].posx}"
    assert x[0].posy == 0.0
    assert x[0].posz == 0.0
    
    