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
    particles[0].x = 0.0
    particles[0].y = 0.0
    particles[0].z = 0.0

    velocities = [pyfluid.Velocity()]
    velocities[0].vx = 1.0
    velocities[0].vy = 0.0
    velocities[0].vz = 0.0
    
    x,v = pyfluid.simulate_step(particles, velocities, 1, 0.1)
    
    assert x[0].x <= 0.1 + 1e-5 and x[0].x >= 0.1 - 1e-5
    assert x[0].y == 0.0
    assert x[0].z == 0.0
    assert v[0].vx == 1.0
    assert v[0].vy == 0.0
    assert v[0].vz == 0.0
    