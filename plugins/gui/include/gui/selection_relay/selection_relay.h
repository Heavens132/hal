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

#include "hal_core/defines.h"

#include <QObject>
#include <QPair>
#include <QVector>
#include <QSet>

namespace hal
{
    class Gate;
    class Module;
    class Net;

    class SelectionRelay : public QObject
    {
        Q_OBJECT

    public:
        enum class ItemType
        {
            None   = 0,
            Gate   = 1,
            Net    = 2,
            Module = 3
        };

        enum class Subfocus
        {
            None  = 0,
            Left  = 1,
            Right = 2
            //        up     = 3,
            //        down   = 4,
            //        center = 5,
        };

        enum class Mode
        {
            Override = 0,
            Add      = 1,
            Remove   = 2
        };

        explicit SelectionRelay(QObject* parent = nullptr);

        void clear(); // does not emit the "update" signal!
        void clear_and_update();

        void register_sender(void* sender, QString name);
        void remove_sender(void* sender);

        // TEST METHOD
        // USE RELAY METHODS OR ACCESS SIGNALS DIRECTLY ???
        void relay_selection_changed(void* sender);
        void relay_subfocus_changed(void* sender);

        void navigate_up();
        void navigate_down();
        void navigate_left();
        void navigate_right();

        void handle_module_removed(const u32 id);
        void handle_gate_removed(const u32 id);
        void handle_net_removed(const u32 id);

        bool isModuleSelected(u32 id) const;
        bool isGateSelected(u32 id) const;
        bool isNetSelected(u32 id) const;

        void suppressedByFilter(const QList<u32>& modIds = QList<u32>(),
                                const QList<u32>& gatIds = QList<u32>(),
                                const QList<u32>& netIds = QList<u32>());

    Q_SIGNALS:
        // TEST SIGNAL
        // ADD ADDITIONAL INFORMATION (LIKE PREVIOUS FOCUS) OR LEAVE THAT TO SUBSCRIBERS ???
        // USE SEPARATE OR COMBINED SIGNALS ??? MEANING DOES A SELECTION CAHNGE FIRE A SUBSELECTION CHANGED SIGNAL OR IS THAT IMPLICIT
        void selection_changed(void* sender);
        //void focus_changed(void* sender); // UNCERTAIN
        void subfocus_changed(void* sender);

    public:
        QSet<u32> m_selected_gates;
        QSet<u32> m_selected_nets;
        QSet<u32> m_selected_modules;

        // MAYBE UNNECESSARY
        ItemType m_current_type;
        u32 m_current_id;

        // USE ARRAY[0] INSTEAD OF MEMBER ???
        ItemType m_focus_type;
        u32 m_focus_id;

        Subfocus m_subfocus;
        u32 m_subfocus_index;    // HANDLE VIA INT OR STRING ?? INDEX HAS TO BE KNOWN ANYWAY TO FIND NEXT / PREVIOUS BOTH OPTIONS KIND OF BAD

    private:
        QSet<u32> mModulesSuppressedByFilter;
        QSet<u32> mGatesSuppressedByFilter;
        QSet<u32> mNetsSuppressedByFilter;

        static bool s_navigation_skips_enabled;    // DOES THIS HAVE ANY USE ???

        // RENAME THESE METHODS ???
        void follow_gate_input_pin(Gate* g, u32 input_pin_index);
        void follow_gate_output_pin(Gate* g, u32 output_pin_index);
        void follow_module_input_pin(Module* m, u32 input_pin_index);
        void follow_module_output_pin(Module* m, u32 output_pin_index);

        void follow_net_to_source(Net* n);
        void follow_net_to_destination(Net* n, u32 dst_index);

        void subfocus_none();
        void subfocus_left();
        void subfocus_right();

        //    bool try_subfocus_left();
        //    bool try_subfocus_right();
        //    bool try_subfocus_up();
        //    bool try_subfocus_down();

        QVector<QPair<void*, QString>> m_sender_register;
    };
}
