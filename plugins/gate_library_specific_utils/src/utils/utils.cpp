#include "utils/utils.h"

#include "hal_core/netlist/gate.h"
#include "hal_core/netlist/net.h"
#include "hal_core/netlist/netlist.h"
#include "hal_core/utilities/log.h"

#include <queue>

namespace hal
{
    namespace gate_library_specific_utils
    {
        /* find all successor/predecessor (forward = true/false) sequential gates of start_gate */
        std::unordered_set<Gate*> Utils::get_sequential_successors(Gate* start_gate)
        {
            std::unordered_set<Gate*> found_ffs;
            for (const auto& n : start_gate->get_fan_out_nets())
            {
                auto suc = get_sequential_successors(n);
                found_ffs.insert(suc.begin(), suc.end());
            }
            /*
            if (auto it = found_ffs.find(start_gate); it != found_ffs.end())
            {
                found_ffs.erase(it);
            }
            */
            return found_ffs;
        }

        std::unordered_set<Gate*> Utils::get_sequential_successors_internal(Net* start_net, std::unordered_set<u32>& seen)
        {
            if (auto it = m_successor_cache.find(start_net->get_id()); it != m_successor_cache.end())
            {
                return it->second;
            }

            if (seen.find(start_net->get_id()) != seen.end())
            {
                return {};
            }

            seen.insert(start_net->get_id());

            std::unordered_set<Gate*> found_ffs;

            for (const auto& dst : start_net->get_destinations())
            {
                auto next_gate = dst->get_gate();

                /* skip if entry to clk */
                if (is_sequential(next_gate))
                {
                    auto clock_ports = get_clock_ports(next_gate);
                    if (clock_ports.find(dst->get_pin()) != clock_ports.end())
                    {
                        continue;
                    }

                    found_ffs.insert(next_gate);
                }
                else
                {
                    for (const auto& n : next_gate->get_fan_out_nets())
                    {
                        auto successors = get_sequential_successors_internal(n, seen);
                        found_ffs.insert(successors.begin(), successors.end());
                    }
                }
            }

            m_successor_cache.emplace(start_net->get_id(), found_ffs);
            return found_ffs;
        }

        std::unordered_set<Gate*> Utils::get_sequential_successors(Net* start_net)
        {
            std::unordered_set<u32> seen;
            return get_sequential_successors_internal(start_net, seen);
        }

        /*
std::unordered_set<Gate*> Utils::get_sequential_successors(Net* start_net)
{
    std::unordered_set<Gate*> found_ffs;
    std::unordered_set<Net*> seen;
    std::queue<Net*> q;
    q.push(start_net);

    while (!q.empty())
    {
        auto current_net = q.front();
        q.pop();

        if (seen.find(current_net) != seen.end())
        {
            continue;
        }
        seen.insert(current_net);

        for (const auto& dst : current_net->get_destinations())
        {
            auto& next_gate = dst->get_gate();

            if (is_sequential(next_gate))
            {
                auto clock_ports = get_clock_ports(next_gate);
                if (clock_ports.find(dst->get_pin()) != clock_ports.end())
                {
                    continue;
                }

                found_ffs.insert(next_gate);
            }
            else
            {
                for (const auto& n : next_gate->get_fan_out_nets())
                {
                    q.push(n);
                }
            }
        }
    }
    return found_ffs;
}
*/

        void Utils::clear_successor_cache()
        {
            m_successor_cache.clear();
        }

        std::unordered_set<Net*> Utils::get_clock_signals_of_gate(Gate* sg)
        {
            std::unordered_set<Net*> clk_nets;

            for (const auto& pin_type : get_clock_ports(sg))
            {
                if (sg->get_fan_in_net(pin_type) != nullptr)
                {
                    clk_nets.insert(sg->get_fan_in_net(pin_type));
                }
            }
            return clk_nets;
        }

        std::unordered_set<Net*> Utils::get_enable_signals_of_gate(Gate* sg)
        {
            std::unordered_set<Net*> enable_nets;

            for (const auto& pin_type : get_enable_ports(sg))
            {
                if (sg->get_fan_in_net(pin_type) != nullptr)
                {
                    enable_nets.insert(sg->get_fan_in_net(pin_type));
                }
            }
            return enable_nets;
        }

        std::unordered_set<Net*> Utils::get_reset_signals_of_gate(Gate* sg)
        {
            std::unordered_set<Net*> reset_nets;

            for (const auto& pin_type : get_reset_ports(sg))
            {
                if (sg->get_fan_in_net(pin_type) != nullptr)
                {
                    reset_nets.insert(sg->get_fan_in_net(pin_type));
                }
            }
            return reset_nets;
        }

        std::unordered_set<Net*> Utils::get_set_signals_of_gate(Gate* sg)
        {
            std::unordered_set<Net*> set_nets;

            for (const auto& pin_type : get_set_ports(sg))
            {
                if (sg->get_fan_in_net(pin_type) != nullptr)
                {
                    set_nets.insert(sg->get_fan_in_net(pin_type));
                }
            }
            return set_nets;
        }

        std::unordered_set<Net*> Utils::get_data_signals_of_gate(Gate* sg)
        {
            std::unordered_set<Net*> data_nets;

            for (const auto& pin_type : get_data_ports(sg))
            {
                if (sg->get_fan_in_net(pin_type) != nullptr)
                {
                    data_nets.insert(sg->get_fan_in_net(pin_type));
                }
            }
            return data_nets;
        }
    }        // namespace dataflow
}    // namespace hal