#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fluid_sim/simulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyfluid, m) {
    py::class_<Particle>(m, "Particle")
        .def(py::init<>())
        .def_readwrite("x", &Particle::x)
        .def_readwrite("y", &Particle::y)
        .def_readwrite("z", &Particle::z);

    py::class_<Velocity>(m, "Velocity")
        .def(py::init<>())
        .def_readwrite("vx", &Velocity::vx)
        .def_readwrite("vy", &Velocity::vy)
        .def_readwrite("vz", &Velocity::vz);

    m.def("simulate_step", [](std::vector<Particle>& particles, std::vector<Velocity>& velocities, int count, float dt) {
        simulate_step(particles.data(), velocities.data(), static_cast<int>(particles.size()), dt);
        return std::make_pair(particles, velocities);
    });

}