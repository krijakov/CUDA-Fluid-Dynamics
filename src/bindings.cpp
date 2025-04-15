#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fluid_sim/simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyfluid, m) {
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
        .def_readwrite("density", &MeshTile::density)
        .def_readwrite("avg_velx", &MeshTile::avg_velx)
        .def_readwrite("avg_vely", &MeshTile::avg_vely)
        .def_readwrite("avg_velz", &MeshTile::avg_velz)
        .def_readwrite("pressure", &MeshTile::pressure);

    m.def("simulate_step", [](std::vector<Particle>& particles, std::vector<MeshTile>& meshes, int N, int M, float dt) {
        simulate_step(particles.data(), meshes.data(), static_cast<int>(particles.size()), static_cast<int>(meshes.size()), dt);
        return std::make_pair(particles, meshes);
    });

}