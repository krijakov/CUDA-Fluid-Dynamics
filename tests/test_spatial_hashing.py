#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Tests for the spatial hashing logic.

"""

import pytest
import pyfluid

@pytest.fixture
def default_params():
    P = pyfluid.Params()
    P.N_tiles_x =  P.N_tiles_y =  P.N_tiles_z = 4 # 4×4×4 grid: IDs 0…63
    P.tile_size_x = P.tile_size_y =  P.tile_size_z = 1.0
    P.geometry_type = 0
    return P

@pytest.mark.parametrize("pos,expected", [
    ((0.1,0.1,0.1), 0),
    ((3.9,0,0),     3),
    ((0,3.9,0),    12),   # y index 3 → 3*4 + 0
    ((0,0,3.9),    48),   # z index 3 → 3*16
    ((4.0,0,0),    -1),   # out of bounds
    ((-0.1,2,2),   -1),
])
def test_cubic_grid_hash(default_params, pos, expected):
    assert pyfluid.cubic_grid_hash(pos, default_params) == expected
    

def test_position_to_mesh_id_alias(default_params):
    # must match direct hash
    assert pyfluid.position_to_mesh_id((1.2,2.3,0.7), default_params) \
           == pyfluid.cubic_grid_hash((1.2,2.3,0.7), default_params)
           
@pytest.mark.parametrize("mesh_id,expected_neighbours", [
    (0,   [-1,1, -1,4, -1,16]),   # corner cell
    (21,  [20,22, 17,25, 5,37]),  # an internal-ish cell
])
def test_get_neighbours(default_params, mesh_id, expected_neighbours):
    nbs = pyfluid.get_neighbour_mesh_ids(mesh_id, default_params)
    assert list(nbs) == expected_neighbours
    
