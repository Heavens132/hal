//  MIT License
//
//  Copyright (c) 2019 Ruhr University Bochum, Chair for Embedded Security. All Rights reserved.
//  Copyright (c) 2021 Max Planck Institute for Security and Privacy. All Rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "hal_core/plugin_system/plugin_interface_base.h"
#include "netlist_simulator/netlist_simulator.h"
#include <unordered_map>
#include <memory>

namespace hal
{
    class PLUGIN_API NetlistSimulatorPlugin : public BasePluginInterface
    {
        std::unordered_map<std::string, std::shared_ptr<NetlistSimulator> > m_shared_simulator_map;
    public:
        /**
         * Get the name of the plugin.
         *
         * @returns The name of the plugin.
         */
        std::string get_name() const override;

        /**
         * Get the version of the plugin.
         *
         * @returns The version of the plugin.
         */
        std::string get_version() const override;

        /**
         * Create a netlist simulator instance.
         * 
         * @returns The simulator instance.
         */
        std::unique_ptr<NetlistSimulator> create_simulator() const;

        /**
         * Get simulator that is shared with external module
         *
         * @param user Name of external module
         * @return Shared pointer to simulator instance
         */
        std::shared_ptr<NetlistSimulator> get_shared_simulator(const std::string& module_name);
    };
}    // namespace hal
