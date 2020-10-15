#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "hal_core/python_bindings/python_bindings.h"

#include "netlist_simulator/plugin_netlist_simulator.h"
#include "netlist_simulator/netlist_simulator.h"

#include "hal_core/python_bindings/python_bindings.h"

namespace py = pybind11;

namespace hal
{

// the name in PYBIND11_MODULE/PYBIND11_PLUGIN *MUST* match the filename of the output library (without extension),
    // otherwise you will get "ImportError: dynamic module does not define module export function" when importing the module

    #ifdef PYBIND11_MODULE
    PYBIND11_MODULE(libnetlist_simulator, m)
    {
        m.doc() = "hal netlist_simulator python bindings";
#else
    PYBIND11_PLUGIN(libnetlist_simulator)
    {
        py::module m("netlist_simulator", "hal netlist_simulator python bindings");
#endif    // ifdef PYBIND11_MODULE

        py::class_<NetlistSimulatorPlugin, RawPtrWrapper<NetlistSimulatorPlugin>, BasePluginInterface, RawPtrWrapper<NetlistSimulatorPlugin>>(m, "NetlistSimulatorPlugin")
            .def_property_readonly("name", &NetlistSimulatorPlugin::get_name)
            .def("get_name", &NetlistSimulatorPlugin::get_name)
            .def_property_readonly("version", &NetlistSimulatorPlugin::get_version)
            .def("get_version", &NetlistSimulatorPlugin::get_version)
            .def("create_simulator", &NetlistSimulatorPlugin::create_simulator);

        py::class_<NetlistSimulator>(m, "NetlistSimulator")
            .def("add_gates", &NetlistSimulator::add_gates, py::arg("gates"), R"(
                Add gates to the simulation set.
                Only elements in the simulation set are considered during simulation.

                :param list[hal_py.Gate] gates: The gates to add.
            )")

            .def("add_clock_frequency", &NetlistSimulator::add_clock_frequency, py::arg("clock_net"), py::arg("frequency"), py::arg("start_at_zero"), R"(
                Specify a net that carries the clock signal and set the clock frequency in hertz.

        py::class_<NetlistSimulator>(m, "NetlistSimulator")
            .def("add_gates", &NetlistSimulator::add_gates)
            .def("add_clock_hertz", &NetlistSimulator::add_clock_hertz)
            .def("add_clock_period", &NetlistSimulator::add_clock_period)
            .def("get_gates", &NetlistSimulator::get_gates)
            .def("get_input_nets", &NetlistSimulator::get_input_nets)
            .def("get_output_nets", &NetlistSimulator::get_output_nets)
            .def("set_input", &NetlistSimulator::set_input)
            .def("load_initial_values", &NetlistSimulator::load_initial_values)
            .def("simulate", &NetlistSimulator::simulate)
            .def("reset", &NetlistSimulator::reset)
            .def("set_simulation_state", &NetlistSimulator::set_simulation_state)
            .def("get_simulation_state", &NetlistSimulator::get_simulation_state)
            .def("set_iteration_timeout", &NetlistSimulator::set_iteration_timeout)
            .def("get_simulation_timeout", &NetlistSimulator::get_simulation_timeout)
            ;

            .def("add_clock_period", &NetlistSimulator::add_clock_period, py::arg("clock_net"), py::arg("period"), py::arg("start_at_zero"), R"(
                Specify a net that carries the clock signal and set the clock period in picoseconds.
        
                :param hal_py.Net clock_net: The net that carries the clock signal.
                :param int period: The clock period from rising edge to rising edge in picoseconds.
                :param bool start_at_zero: Initial clock state is 0 if true, 1 otherwise.
            )")

            .def("get_gates", &NetlistSimulator::get_gates, R"(
                Get all gates that are in the simulation set.

                :returns: The name.
                :rtype: set[hal_py.Gate]
            )")

            .def("get_input_nets", &NetlistSimulator::get_input_nets, R"(
                Get all nets that are considered inputs, i.e., not driven by a gate in the simulation set or global inputs.

                :returns: The input nets.
                :rtype: list[hal_py.Net]
            )")

            .def("get_output_nets", &NetlistSimulator::get_output_nets, R"(
                Get all output nets of gates in the simulation set that have a destination outside of the set or that are global outputs.

                :returns: The output nets.
                :rtype: list[hal_py.Net]
            )")

            .def("set_input", &NetlistSimulator::set_input, py::arg("net"), py::arg("value"), R"(
                Set the signal for a specific wire to control input signals between simulation cycles.
            
                :param hal_py.Net net: The net to set a signal value for.
                :param netlist_simulator.SignalValue value: The value to set.
            )")

            .def("load_initial_values", &NetlistSimulator::load_initial_values, R"(
                Load the initial values for all sequential elements into the current state.
                For example, a Flip Flop may be initialized with HIGH output in FPGAs.
                This is not done automatically!
            )")

            .def("simulate", &NetlistSimulator::simulate, py::arg("picoseconds"), R"(
                Simulate for a specific period, advancing the internal state.
                Use \p set_input to control specific signals.
         
                :param int picoseconds: The duration to simulate.
            )")

            .def("reset", &NetlistSimulator::reset, R"(
                Reset the simulator state, i.e., treat all signals as unknown.
                Does not remove gates/nets from the simulation set.
            )")

            .def("set_simulation_state", &NetlistSimulator::set_simulation_state, py::arg("state"), R"(
                Set the simulator state, i.e., net signals, to a given state.
                Does not influence gates/nets added to the simulation set.
        
                :param netlist_simulator.Simulation state: The state to apply.
            )")

            .def("get_simulation_state", &NetlistSimulator::get_simulation_state, R"(
                Get the current simulation state.
        
                :returns: The current simulation state.
                :rtype: libnetlist_simulator.Simulation
            )")

            .def("set_iteration_timeout", &NetlistSimulator::set_iteration_timeout, py::arg("iterations"), R"(
                Set the iteration timeout, i.e., the maximum number of events processed for a single point in time.
                Useful to abort in case of infinite loops.
                A value of 0 disables the timeout.
        
                :param int iterations: The iteration timeout.
            )")

            .def("get_simulation_timeout", &NetlistSimulator::get_simulation_timeout, R"(
                Get the current iteration timeout value.

                :returns: The iteration timeout.
                :rtype: int
            )");

        py::class_<Simulation>(m, "Simulation")
            .def(py::init<>())

            .def("get_net_value", &Simulation::get_net_value, py::arg("net"), py::arg("time"), R"(
                Get the signal value of a specific net at a specific point in time specified in picoseconds.
         
                :param hal_py.Net net: The net to inspect.
                :param int time: The time in picoseconds.
                :returns: The net's signal value.
                :rtype: netlist_simulator.SignalValue
            )")

            .def("add_event", &Simulation::add_event, py::arg("event"), R"(
                Adds a custom event to the simulation.
         
                :param netlist_simulator.Event event: The event to add.
            )")

            .def("get_events", &Simulation::get_events, R"(
                Get all events of the simulation.

                :returns: A map from net to associated events for that net sorted by time.
            )");

        py::enum_<SignalValue>(m, "SignalValue")
            .value("X", SignalValue::X)
            .value("ZERO", SignalValue::ZERO)
            .value("ONE", SignalValue::ONE)
            .value("Z", SignalValue::Z)
            .export_values();

        py::class_<Event>(m, "Event")
            .def(py::init<>())
            
            .def(py::self == py::self, R"(
                Tests whether two events are equal.

                :returns: True when both events are equal, false otherwise.
                :rtype: bool
            )")

            .def(py::self < py::self, R"(
                Tests whether one event happened before the other.
         
                :returns: True when this event happened before the other, false otherwise.
                :rtype: bool
            )");

    #ifndef PYBIND11_MODULE
        return m.ptr();
    #endif    // PYBIND11_MODULE
    }
}
