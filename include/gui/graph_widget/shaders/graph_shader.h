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

#include "gui/graph_widget/items/nets/graphics_net.h"
#include "gui/graph_widget/items/nodes/graphics_node.h"

#include <QColor>
#include <QMap>
#include <QSet>
#include <QVector>
namespace hal{
class graph_context;

class graph_shader
{
public:
    struct shading
    {
        QMap<u32, graphics_node::visuals> module_visuals;
        QMap<u32, graphics_node::visuals> gate_visuals;
        QMap<u32, graphics_net::visuals> net_visuals;
    };

    explicit graph_shader(const graph_context* const context);
    virtual ~graph_shader() = default;

    virtual void add(const QSet<u32> modules, const QSet<u32> gates, const QSet<u32> nets) = 0;
    virtual void remove(const QSet<u32> modules, const QSet<u32> gates, const QSet<u32> nets) = 0;
    virtual void update() = 0;

    const shading& get_shading();

protected:
    const graph_context* const m_context;

    shading m_shading;
};
}
