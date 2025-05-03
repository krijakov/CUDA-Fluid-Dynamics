#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fluid_sim/simulation.hpp"
#include "fluid_sim/spatial_hashing.hpp"
namespace py = pybind11;

// helper to turn a 3‐tuple into float3:
static float3 to_float3(const py::tuple &t)
{
    return {t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>()};
}

PYBIND11_MODULE(pyfluid, m)
{
    py::class_<Particle>(m, "Particle")
        .def(py::init<>())
        .def_readwrite("posx", &Particle::posx)
        .def_readwrite("posy", &Particle::posy)
        .def_readwrite("posz", &Particle::posz)
        .def_readwrite("vx", &Particle::vx)
        .def_readwrite("vy", &Particle::vy)
        .def_readwrite("vz", &Particle::vz)
        .def_readwrite("mass", &Particle::mass)
        .def_readwrite("mesh_id", &Particle::mesh_id);

    py::class_<MeshTile>(m, "MeshTile")
        .def(py::init<>())
        .def_readwrite("posx", &MeshTile::posx)
        .def_readwrite("posy", &MeshTile::posy)
        .def_readwrite("posz", &MeshTile::posz)
        .def_readwrite("size", &MeshTile::size)
        .def_readwrite("id", &MeshTile::id)
        .def_readwrite("N_particle", &MeshTile::N_particle)
        .def_readwrite("velx", &MeshTile::velx)
        .def_readwrite("vely", &MeshTile::vely)
        .def_readwrite("velz", &MeshTile::velz)
        .def_readwrite("velx2", &MeshTile::velx2)
        .def_readwrite("vely2", &MeshTile::vely2)
        .def_readwrite("velz2", &MeshTile::velz2)
        .def_readwrite("pressure", &MeshTile::pressure);

    py::class_<Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("minx", &Params::minx)
        .def_readwrite("miny", &Params::miny)
        .def_readwrite("minz", &Params::minz)
        .def_readwrite("maxx", &Params::maxx)
        .def_readwrite("maxy", &Params::maxy)
        .def_readwrite("maxz", &Params::maxz)
        .def_readwrite("N_tiles_x", &Params::N_tiles_x)
        .def_readwrite("N_tiles_y", &Params::N_tiles_y)
        .def_readwrite("N_tiles_z", &Params::N_tiles_z)
        .def_readwrite("tile_size_x", &Params::tile_size_x)
        .def_readwrite("tile_size_y", &Params::tile_size_y)
        .def_readwrite("tile_size_z", &Params::tile_size_z)
        .def_readwrite("num_tiles", &Params::num_tiles)
        .def_readwrite("num_particles", &Params::num_particles)
        .def_readwrite("dt", &Params::dt)
        .def_readwrite("Nstep", &Params::Nstep)
        .def_readwrite("geometry_type", &Params::geometry_type);

    m.def("simulate_step", [](std::vector<Particle> &particles, std::vector<MeshTile> &meshes, Params *parameters)
          {
        simulate_step(particles.data(), meshes.data(), parameters);
        return std::make_pair(particles, meshes); });

    //--------------------------------------Testing--------------------------------------
    // Spatial hashing:
    m.def("cubic_grid_hash",
        [](py::tuple pos, const Params &P) {
          return cubic_grid_hash(to_float3(pos), &P);
        },
        py::arg("pos"), py::arg("params"));

        m.def("position_to_mesh_id",
            [](py::tuple pos, const Params &P) {
              return position_to_mesh_id(to_float3(pos), &P);
            },
            py::arg("pos"), py::arg("params"));
    
      m.def("get_neighbour_mesh_ids",
            [](int mesh_id, const Params &P) {
              std::array<int,6> out;
              get_neighbour_mesh_ids(mesh_id, out.data(), &P);
              return out;   // pybind will convert std::array→tuple/list
            },
            py::arg("mesh_id"), py::arg("params"));
}