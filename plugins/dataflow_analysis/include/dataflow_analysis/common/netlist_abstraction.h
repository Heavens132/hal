#pragma once

#include "hal_core/defines.h"

#include "gate_library_specific_utils/include/plugin_gate_library_specific_utils.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hal
{
    /* forward declaration */
    class Gate;
    class Netlist;

    namespace dataflow
    {
        struct Grouping;

        struct NetlistAbstraction
        {
            NetlistAbstraction(Netlist* nl_arg);

            std::shared_ptr<Grouping> create_initial_grouping() const;

            // netlist
            Netlist* nl;

            // utils
            std::shared_ptr<gate_library_specific_utils::Utils> utils;
            bool yosys;

            // all ffs
            std::vector<Gate*> all_sequential_gates;

            /* pre_processed_data */
            std::unordered_map<u32, std::vector<u32>> gate_to_fingerprint;
            std::unordered_map<u32, std::unordered_set<u32>> gate_to_clock_signals;
            std::unordered_map<u32, std::unordered_set<u32>> gate_to_enable_signals;
            std::unordered_map<u32, std::unordered_set<u32>> gate_to_reset_signals;
            std::unordered_map<u32, std::unordered_set<u32>> gate_to_register_stages;
            std::unordered_map<u32, std::unordered_set<u32>> gate_to_predecessors;
            std::unordered_map<u32, std::unordered_set<u32>> gate_to_successors;
            std::unordered_map<u32, std::vector<std::vector<u32>>> gate_to_output_shape;
            std::unordered_map<u32, std::vector<std::vector<u32>>> gate_to_input_shape;
        };
    }    // namespace dataflow
}    // namespace hal