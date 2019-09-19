#include "selection_details_widget/module_details_widget.h"
#include "gui_globals.h"
#include "netlist/gate.h"
#include "netlist/module.h"
#include <QLabel>
#include <QVBoxLayout>
#include <QDebug>

module_details_widget::module_details_widget(QWidget* parent) : QWidget(parent), m_treeview(new QTreeView(this)), m_tree_module_model(new tree_module_model(this))
{
    m_treeview->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_treeview->setModel(m_tree_module_model);

    m_content_layout = new QVBoxLayout(this);
    m_content_layout->setContentsMargins(0, 0, 0, 0);
    m_content_layout->setSpacing(0);
    m_content_layout->setAlignment(Qt::AlignTop);
    m_content_layout->addWidget(m_treeview);

    connect(&g_netlist_relay, &netlist_relay::module_event, this, &module_details_widget::handle_module_event);
}

void module_details_widget::handle_module_event(module_event_handler::event ev, std::shared_ptr<module> module, u32 associated_data)
{
    Q_UNUSED(ev)
    Q_UNUSED(associated_data)
    if (m_current_id == module->get_id())
    {
        update(module->get_id());
    }
}

void module_details_widget::update(u32 module_id)
{
    m_current_id = module_id;
    m_tree_module_model->update(module_id);
}
