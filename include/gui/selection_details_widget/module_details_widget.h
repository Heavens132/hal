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

#include "gui/gui_def.h"
#include "netlist/endpoint.h"
#include "netlist_relay/netlist_relay.h"

#include <QWidget>
/* forward declaration */
class QLabel;
class QTableWidget;
class QTableWidgetItem;
class QTreeWidget;
class QTreeWidgetItem;
class QVBoxLayout;
class QHBoxLayout;
class QScrollArea;
class QGridLayout;
class QModelIndex;
class QFont;
class QPushButton;

namespace hal
{
    class ModuleDetailsWidget : public QWidget
    {
        Q_OBJECT
    public:
        ModuleDetailsWidget(QWidget* parent = nullptr);

        void update(const u32 module_id);

    public Q_SLOTS:

        void handle_netlist_marked_global_input(std::shared_ptr<Netlist> netlist, u32 associated_data);
        void handle_netlist_marked_global_output(std::shared_ptr<Netlist> netlist, u32 associated_data);
        void handle_netlist_marked_global_inout(std::shared_ptr<Netlist> netlist, u32 associated_data);
        void handle_netlist_unmarked_global_input(std::shared_ptr<Netlist> netlist, u32 associated_data);
        void handle_netlist_unmarked_global_output(std::shared_ptr<Netlist> netlist, u32 associated_data);
        void handle_netlist_unmarked_global_inout(std::shared_ptr<Netlist> netlist, u32 associated_data);

        void handle_module_name_changed(std::shared_ptr<Module> module);
        void handle_submodule_added(std::shared_ptr<Module> module, u32 associated_data);
        void handle_submodule_removed(std::shared_ptr<Module> module, u32 associated_data);
        void handle_module_gate_assigned(std::shared_ptr<Module> module, u32 associated_data);
        void handle_module_gate_removed(std::shared_ptr<Module> module, u32 associated_data);
        void handle_module_input_port_name_changed(std::shared_ptr<Module> module, u32 associated_data);
        void handle_module_output_port_name_changed(std::shared_ptr<Module> module, u32 associated_data);
        void handle_module_type_changed(std::shared_ptr<Module> module);

        void handle_net_name_changed(std::shared_ptr<Net> net);
        void handle_net_source_added(std::shared_ptr<Net> net, const u32 src_gate_id);
        void handle_net_source_removed(std::shared_ptr<Net> net, const u32 src_gate_id);
        void handle_net_destination_added(std::shared_ptr<Net> net, const u32 dst_gate_id);
        void handle_net_destination_removed(std::shared_ptr<Net> net, const u32 dst_gate_id);

    private:
        QFont m_key_font;

        QScrollArea* m_scroll_area;
        QWidget* m_top_lvl_container;
        QVBoxLayout* m_top_lvl_layout;
        QVBoxLayout* m_content_layout;

        QPushButton* m_general_info_button;
        QPushButton* m_input_ports_button;
        QPushButton* m_output_ports_button;

        QTableWidget* m_general_table;

        QTableWidgetItem* m_name_item;
        QTableWidgetItem* m_id_item;
        QTableWidgetItem* m_type_item;
        QTableWidgetItem* m_number_of_gates_item;
        QTableWidgetItem* m_number_of_submodules_item;
        QTableWidgetItem* m_number_of_nets_item;

        QTableWidget* m_input_ports_table;

        QTableWidget* m_output_ports_table;

        void handle_buttons_clicked();

        QSize calculate_table_size(QTableWidget* table);

        u32 m_current_id;

        void add_general_table_static_item(QTableWidgetItem* item);
        void add_general_table_dynamic_item(QTableWidgetItem* item);
        void style_table(QTableWidget* table);
    };
}
