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

    x,m = pyfluid.simulate_step(particles, meshes, 1, 1, 0.1)
    
    assert x[0].posx <= 0.1 + 1e-5 and x[0].posx >= 0.1 - 1e-5
    assert x[0].posy == 0.0
    assert x[0].posz == 0.0
    assert x[0].vx == 1.0
    assert x[0].vy == 0.0
    assert x[0].vz == 0.0
    