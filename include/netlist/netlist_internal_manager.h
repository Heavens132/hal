//  MIT License
//
//  Copyright (c) 2019 Ruhr-University Bochum, Germany, Chair for Embedded Security. All Rights reserved.
//  Copyright (c) 2019 Marc Fyrbiak, Sebastian Wallat, Max Hoffmann ("ORIGINAL AUTHORS"). All rights reserved.
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

#include "def.h"

namespace hal
{
    // forward declaration
    class Netlist;
    class Gate;
    class GateType;
    class Net;
    class Module;
    class Endpoint;

    /**
     * @ingroup netlist
     */
    class NETLIST_API NetlistInternalManager
    {
        friend class Netlist;
        friend class Net;
        friend class Module;

    private:
        Netlist* m_netlist;

        explicit NetlistInternalManager(Netlist* nl);

        ~NetlistInternalManager() = default;

        // gate functions
        std::shared_ptr<Gate> create_gate(u32 id, const std::shared_ptr<const GateType>& gt, const std::string& name, float x, float y);
        bool delete_gate(std::shared_ptr<Gate> gate);
        bool is_gate_type_invalid(const std::shared_ptr<const GateType>& gt) const;

        // net functions
        std::shared_ptr<Net> create_net(u32 id, const std::string& name);
        bool delete_net(const std::shared_ptr<Net>& net);
        bool net_add_source(const std::shared_ptr<Net>& net, const Endpoint& ep);
        bool net_remove_source(const std::shared_ptr<Net>& net, const Endpoint& ep);
        bool net_add_destination(const std::shared_ptr<Net>& net, const Endpoint& ep);
        bool net_remove_destination(const std::shared_ptr<Net>& net, const Endpoint& ep);

        // module functions
        std::shared_ptr<Module> create_module(u32 id, const std::shared_ptr<Module>& parent, const std::string& name);
        bool delete_module(const std::shared_ptr<Module>& module);
        bool module_assign_gate(const std::shared_ptr<Module>& m, const std::shared_ptr<Gate>& g);
        bool module_remove_gate(const std::shared_ptr<Module>& m, const std::shared_ptr<Gate>& g);
    };
}    // namespace hal
