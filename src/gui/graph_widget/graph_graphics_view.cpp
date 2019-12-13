#include "graph_widget/graph_graphics_view.h"

#include "core/log.h"

#include "netlist/gate.h"
#include "netlist/module.h"
#include "netlist/net.h"

#include "gui/graph_widget/contexts/graph_context.h"
#include "gui/graph_widget/graph_widget.h"
#include "gui/graph_widget/graph_widget_constants.h"
#include "gui/graph_widget/graphics_scene.h"
#include "gui/graph_widget/items/graphics_gate.h"
#include "gui/graph_widget/items/graphics_item.h"
#include "gui/graph_widget/items/io_graphics_net.h"
#include "gui/graph_widget/items/separated_graphics_net.h"
#include "gui/graph_widget/items/standard_graphics_gate.h"
#include "gui/graph_widget/items/standard_graphics_module.h"
#include "gui/graph_widget/items/standard_graphics_net.h"
#include "gui/graph_widget/items/utility_items/drag_shadow_gate.h"
#include "gui/gui_globals.h"
#include "gui/gui_utils/netlist.h"

#include <QAction>
#include <QApplication>
#include <QColorDialog>
#include <QDrag>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QScrollBar>
#include <QStyleOptionGraphicsItem>
#include <QWheelEvent>
#include <QWidgetAction>
#include <qmath.h>

graph_graphics_view::graph_graphics_view(graph_widget* parent)
    : QGraphicsView(parent), m_graph_widget(parent), m_minimap_enabled(false), m_grid_enabled(true), m_grid_clusters_enabled(true), m_grid_type(graph_widget_constants::grid_type::lines),
      m_zoom_modifier(Qt::NoModifier), m_zoom_factor_base(1.0015)
{
    connect(&g_selection_relay, &selection_relay::subfocus_changed, this, &graph_graphics_view::conditional_update);
    connect(this, &graph_graphics_view::customContextMenuRequested, this, &graph_graphics_view::show_context_menu);
    connect(&g_settings_relay, &settings_relay::setting_changed, this, &graph_graphics_view::handle_global_setting_changed);

    initialize_settings();

    setContextMenuPolicy(Qt::CustomContextMenu);
    setOptimizationFlags(QGraphicsView::DontSavePainterState);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);

    setAcceptDrops(true);
    setMouseTracking(true);
}

void graph_graphics_view::initialize_settings()
{
    unsigned int drag_modifier_setting = g_settings_manager.get("graph_view/drag_mode_modifier").toUInt();
    m_drag_modifier                    = Qt::KeyboardModifier(drag_modifier_setting);
    unsigned int move_modifier_setting = g_settings_manager.get("graph_view/move_modifier").toUInt();
    m_move_modifier                    = Qt::KeyboardModifier(move_modifier_setting);
}

void graph_graphics_view::conditional_update()
{
    if (QStyleOptionGraphicsItem::levelOfDetailFromTransform(transform()) >= graph_widget_constants::gate_min_lod)
        update();
}

void graph_graphics_view::handle_change_color_action()
{
    QColor color = QColorDialog::getColor();

    if (!color.isValid())
        return;
}

void graph_graphics_view::handle_isolation_view_action()
{
    u32 cnt = 0;
    while (true)
    {
        ++cnt;
        QString name = "Isolated View " + QString::number(cnt);
        if (!g_graph_context_manager.context_with_name_exists(name))
        {
            auto context = g_graph_context_manager.create_new_context(name);
            context->add(g_selection_relay.m_selected_modules, g_selection_relay.m_selected_gates);
            return;
        }
    }
}

void graph_graphics_view::handle_move_action(QAction* action)
{
    const u32 mod_id          = action->data().toInt();
    std::shared_ptr<module> m = g_netlist->get_module_by_id(mod_id);
    for (const auto& id : g_selection_relay.m_selected_gates)
    {
        m->assign_gate(g_netlist->get_gate_by_id(id));
    }
    for (const auto& id : g_selection_relay.m_selected_modules)
    {
        g_netlist->get_module_by_id(id)->set_parent_module(m);
    }

    auto gates   = g_selection_relay.m_selected_gates;
    auto modules = g_selection_relay.m_selected_modules;
    g_selection_relay.clear();
    g_selection_relay.relay_selection_changed(this);
}

void graph_graphics_view::handle_move_new_action()
{
    std::unordered_set<std::shared_ptr<gate>> gate_objs;
    std::unordered_set<std::shared_ptr<module>> module_objs;
    for (const auto& id : g_selection_relay.m_selected_gates)
    {
        gate_objs.insert(g_netlist->get_gate_by_id(id));
    }
    for (const auto& id : g_selection_relay.m_selected_modules)
    {
        module_objs.insert(g_netlist->get_module_by_id(id));
    }
    std::shared_ptr<module> parent = gui_utility::first_common_ancestor(module_objs, gate_objs);
    QString parent_name            = QString::fromStdString(parent->get_name());
    bool ok;
    QString name = QInputDialog::getText(nullptr, "", "New module will be created under \"" + parent_name + "\"\nModule Name:", QLineEdit::Normal, "", &ok);
    if (!ok || name.isEmpty())
        return;
    std::shared_ptr<module> m = g_netlist->create_module(g_netlist->get_unique_module_id(), name.toStdString(), parent);

    for (const auto& id : g_selection_relay.m_selected_gates)
    {
        m->assign_gate(g_netlist->get_gate_by_id(id));
    }
    for (const auto& id : g_selection_relay.m_selected_modules)
    {
        g_netlist->get_module_by_id(id)->set_parent_module(m);
    }

    auto gates   = g_selection_relay.m_selected_gates;
    auto modules = g_selection_relay.m_selected_modules;
    g_selection_relay.clear();
    g_selection_relay.relay_selection_changed(this);
}

void graph_graphics_view::handle_rename_action()
{
    if (m_item->item_type() == hal::item_type::gate)
    {
        std::shared_ptr<gate> g = g_netlist->get_gate_by_id(m_item->id());
        const QString name      = QString::fromStdString(g->get_name());
        bool confirm;
        const QString new_name = QInputDialog::getText(this, "Rename gate", "New name:", QLineEdit::Normal, name, &confirm);
        if (confirm)
        {
            g->set_name(new_name.toStdString());
        }
    }
    else if (m_item->item_type() == hal::item_type::module)
    {
        std::shared_ptr<module> m = g_netlist->get_module_by_id(m_item->id());
        const QString name        = QString::fromStdString(m->get_name());
        bool confirm;
        const QString new_name = QInputDialog::getText(this, "Rename module", "New name:", QLineEdit::Normal, name, &confirm);
        if (confirm)
        {
            m->set_name(new_name.toStdString());
        }
    }
}

void graph_graphics_view::adjust_min_scale()
{
    if (!scene())
        return;

    m_min_scale = std::min(viewport()->width() / scene()->width(), viewport()->height() / scene()->height());
}

void graph_graphics_view::paintEvent(QPaintEvent* event)
{
    qreal lod = QStyleOptionGraphicsItem::levelOfDetailFromTransform(transform());

    // USE CONSISTENT METHOD NAMES
    graphics_scene::set_lod(lod);
    graphics_scene::set_grid_enabled(m_grid_enabled);
    graphics_scene::set_grid_clusters_enabled(m_grid_clusters_enabled);
    graphics_scene::set_grid_type(m_grid_type);

    graphics_item::set_lod(lod);
    drag_shadow_gate::set_lod(lod);

    standard_graphics_module::update_alpha();
    standard_graphics_gate::update_alpha();
    standard_graphics_net::update_alpha();
    separated_graphics_net::update_alpha();
    io_graphics_net::update_alpha();

    QGraphicsView::paintEvent(event);
}

void graph_graphics_view::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() != Qt::LeftButton)
        return;

    graphics_item* item = static_cast<graphics_item*>(itemAt(event->pos()));

    if (!item)
        return;

    if (item->item_type() == hal::item_type::module)
        Q_EMIT module_double_clicked(item->id());
}

void graph_graphics_view::drawForeground(QPainter* painter, const QRectF& rect)
{
    Q_UNUSED(rect)

    if (!m_minimap_enabled)
        return;

    QRectF map(viewport()->width() - 210, viewport()->height() - 130, 200, 120);
    painter->resetTransform();
    painter->fillRect(map, QColor(0, 0, 0, 170));
}

void graph_graphics_view::mousePressEvent(QMouseEvent* event)
{
    if (event->modifiers() == m_move_modifier)
    {
        if (event->button() == Qt::LeftButton)
            m_move_position = event->pos();
    }
    else if (event->button() == Qt::LeftButton)
    {
        graphics_item* item = static_cast<graphics_item*>(itemAt(event->pos()));
        if (item && item_draggable(item))
        {
            m_drag_item               = static_cast<graphics_gate*>(item);
            m_drag_mousedown_position = event->pos();
            m_drag_cursor_offset      = m_drag_mousedown_position - mapFromScene(item->pos());
        }
        else
        {
            m_drag_item = nullptr;
        }

        // we still need the normal mouse logic for single clicks
        QGraphicsView::mousePressEvent(event);
    }
    else
        QGraphicsView::mousePressEvent(event);
}

void graph_graphics_view::mouseMoveEvent(QMouseEvent* event)
{
    if (!scene())
        return;

    QPointF delta = target_viewport_pos - event->pos();

    if (qAbs(delta.x()) > 5 || qAbs(delta.y()) > 5)
    {
        target_viewport_pos = event->pos();
        target_scene_pos    = mapToScene(event->pos());
    }

    if (event->buttons().testFlag(Qt::LeftButton))
    {
        if (event->modifiers() == m_move_modifier)
        {
            QScrollBar* hBar  = horizontalScrollBar();
            QScrollBar* vBar  = verticalScrollBar();
            QPoint delta_move = event->pos() - m_move_position;
            m_move_position   = event->pos();
            hBar->setValue(hBar->value() + (isRightToLeft() ? delta_move.x() : -delta_move.x()));
            vBar->setValue(vBar->value() - delta_move.y());
        }
        else
        {
            if (m_drag_item && (event->pos() - m_drag_mousedown_position).manhattanLength() >= QApplication::startDragDistance())
            {
                QDrag* drag         = new QDrag(this);
                QMimeData* mimeData = new QMimeData;

                // TODO set MIME type and icon
                mimeData->setText("dragTest");
                drag->setMimeData(mimeData);
                // drag->setPixmap(iconPixmap);

                // enable DragMoveEvents until mouse released
                drag->exec(Qt::MoveAction);
            }
        }
    }
    QGraphicsView::mouseMoveEvent(event);
}

void graph_graphics_view::dragEnterEvent(QDragEnterEvent* event)
{
    if (event->source() == this && event->proposedAction() == Qt::MoveAction)
    {
        event->acceptProposedAction();
        QSizeF size(m_drag_item->width(), m_drag_item->height());
        QPointF mouse = event->posF();
        QPointF pos   = mapToScene(mouse.x(), mouse.y()) - m_drag_cursor_offset;
        if (g_selection_relay.m_selected_gates.size() > 1)
        {
            // if we are in multi-select mode, reduce the selection to the
            // item we are dragging
            g_selection_relay.clear();
            g_selection_relay.m_selected_gates.insert(m_drag_item->id());
            g_selection_relay.m_focus_type = selection_relay::item_type::gate;
            g_selection_relay.m_focus_id   = m_drag_item->id();
            g_selection_relay.m_subfocus   = selection_relay::subfocus::none;
            g_selection_relay.relay_selection_changed(nullptr);
        }
        static_cast<graphics_scene*>(scene())->start_drag_shadow(pos, size, m_drag_item);
    }
    else
    {
        QGraphicsView::dragEnterEvent(event);
    }
}

void graph_graphics_view::dragLeaveEvent(QDragLeaveEvent* event)
{
    Q_UNUSED(event)
    static_cast<graphics_scene*>(scene())->stop_drag_shadow();
}

void graph_graphics_view::dragMoveEvent(QDragMoveEvent* event)
{
    if (event->source() == this && event->proposedAction() == Qt::MoveAction)
    {
        QPoint mouse         = event->pos();
        bool modifierPressed = event->keyboardModifiers() == m_drag_modifier;
        QPoint shadow        = mouse - m_drag_cursor_offset;
        static_cast<graphics_scene*>(scene())->move_drag_shadow(mapToScene(shadow.x(), shadow.y()), modifierPressed ? graphics_scene::drag_mode::swap : graphics_scene::drag_mode::move);
    }
}

void graph_graphics_view::dropEvent(QDropEvent* event)
{
    if (event->source() == this && event->proposedAction() == Qt::MoveAction)
    {
        event->acceptProposedAction();
        graphics_scene* s = static_cast<graphics_scene*>(scene());
        bool success      = s->stop_drag_shadow();
        if (success)
        {
            bool modifierPressed = event->keyboardModifiers() == m_drag_modifier;
            // TODO: Once the layouter data structures are defined & stable,
            // add code to move the gates to the correct layouter boxes
            // TODO: Also add a mechanism to insert rows and columns of boxes,
            // like in a table calculation software
            if (modifierPressed)
            {
                // swap mode; swap gates
                QPointF targetPos = s->drop_target_item()->pos();
                s->drop_target_item()->setPos(m_drag_item->pos());
                m_drag_item->setPos(targetPos);
            }
            else
            {
                // move mode; move gate to the selected location
                m_drag_item->setPos(s->drop_target());
            }
            // TODO: Once available, trigger a re-layout of all nets here
        }
    }
    else
    {
        QGraphicsView::dropEvent(event);
    }
}

void graph_graphics_view::wheelEvent(QWheelEvent* event)
{
    if (QApplication::keyboardModifiers() == m_zoom_modifier)
    {
        if (event->orientation() == Qt::Vertical)
        {
            qreal angle  = event->angleDelta().y();
            qreal factor = qPow(m_zoom_factor_base, angle);
            gentle_zoom(factor);
        }
    }
}

void graph_graphics_view::keyPressEvent(QKeyEvent* event)
{
    switch (event->key())
    {
        case Qt::Key_Space:
        {
            //qDebug() << "Space pressed";
        }
        break;
    }

    event->ignore();
}

void graph_graphics_view::keyReleaseEvent(QKeyEvent* event)
{
    switch (event->key())
    {
        case Qt::Key_Space:
        {
            //qDebug() << "Space released";
        }
        break;
    }

    event->ignore();
}

void graph_graphics_view::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    adjust_min_scale();
}

void graph_graphics_view::show_context_menu(const QPoint& pos)
{
    graphics_scene* s = static_cast<graphics_scene*>(scene());

    if (!s)
        return;

    QMenu context_menu(this);
    QAction* action;

    QGraphicsItem* item = itemAt(pos);
    bool isGate = false, isModule = false, isNet = false;
    if (item)
    {
        m_item   = static_cast<graphics_item*>(item);
        isGate   = m_item->item_type() == hal::item_type::gate;
        isModule = m_item->item_type() == hal::item_type::module;
        isNet    = m_item->item_type() == hal::item_type::net;

        if (isGate)
        {
            if (g_selection_relay.m_selected_gates.find(m_item->id()) == g_selection_relay.m_selected_gates.end())
            {
                g_selection_relay.clear();
                g_selection_relay.m_selected_gates.insert(m_item->id());
                g_selection_relay.m_focus_type = selection_relay::item_type::gate;
                g_selection_relay.m_focus_id   = m_item->id();
                g_selection_relay.m_subfocus   = selection_relay::subfocus::none;
                g_selection_relay.relay_selection_changed(this);
            }

            context_menu.addAction("This gate:")->setEnabled(false);

            action = context_menu.addAction("  Rename …");
            QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_rename_action);

            action = context_menu.addAction("  Fold parent module");
            QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_fold_single_action);
        }
        else if (isModule)
        {
            if (g_selection_relay.m_selected_modules.find(m_item->id()) == g_selection_relay.m_selected_modules.end())
            {
                g_selection_relay.clear();
                g_selection_relay.m_selected_modules.insert(m_item->id());
                g_selection_relay.m_focus_type = selection_relay::item_type::module;
                g_selection_relay.m_focus_id   = m_item->id();
                g_selection_relay.m_subfocus   = selection_relay::subfocus::none;
                g_selection_relay.relay_selection_changed(this);
            }

            context_menu.addAction("This module:")->setEnabled(false);

            action = context_menu.addAction("  Rename …");
            QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_rename_action);

            action = context_menu.addAction("  Unfold module");
            QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_unfold_single_action);
        }

        if (g_selection_relay.m_selected_gates.size() + g_selection_relay.m_selected_modules.size() > 1)
        {
            context_menu.addSeparator();
            context_menu.addAction("Entire selection:")->setEnabled(false);
        }

        if (isGate || isModule)
        {
            action = context_menu.addAction("  Isolate In New View");
            QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_isolation_view_action);

            action = context_menu.addAction("  Add successors to view");
            connect(action, &QAction::triggered, this, &graph_graphics_view::handle_select_outputs);

            action = context_menu.addAction("  Add predecessors to view");
            connect(action, &QAction::triggered, this, &graph_graphics_view::handle_select_inputs);

            std::shared_ptr<gate> g   = isGate ? g_netlist->get_gate_by_id(m_item->id()) : nullptr;
            std::shared_ptr<module> m = isModule ? g_netlist->get_module_by_id(m_item->id()) : nullptr;

            // only allow move actions on anything that is not the top module
            if (!(isModule && m == g_netlist->get_top_module()))
            {
                QMenu* module_submenu = context_menu.addMenu("  Move to module …");

                action = module_submenu->addAction("New module …");
                QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_move_new_action);
                module_submenu->addSeparator();

                QActionGroup* module_actions = new QActionGroup(module_submenu);
                for (auto& module : g_netlist->get_modules())
                {
                    // don't allow a gate to be moved into its current module
                    // && don't allow a module to be moved into its current module
                    // && don't allow a module to be moved into itself
                    // (either check automatically passes if g respective m is nullptr, so we
                    // don't have to create two loops)
                    if (!module->contains_gate(g) && !module->contains_module(m) && module != m)
                    {
                        QString mod_name = QString::fromStdString(module->get_name());
                        const u32 mod_id = module->get_id();
                        action           = module_submenu->addAction(mod_name);
                        module_actions->addAction(action);
                        action->setData(mod_id);
                    }
                }
                QObject::connect(module_actions, SIGNAL(triggered(QAction*)), this, SLOT(handle_move_action(QAction*)));
            }
        }

        if (g_selection_relay.m_selected_gates.size() + g_selection_relay.m_selected_modules.size() > 1)
        {
            if (!g_selection_relay.m_selected_gates.empty())
            {
                context_menu.addSeparator();
                context_menu.addAction("All selected gates:")->setEnabled(false);

                action = context_menu.addAction("  Fold all parent modules");
                QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_fold_all_action);
            }
            if (!g_selection_relay.m_selected_modules.empty())
            {
                context_menu.addSeparator();
                context_menu.addAction("All selected modules:")->setEnabled(false);

                action = context_menu.addAction("  Unfold all");
                QObject::connect(action, &QAction::triggered, this, &graph_graphics_view::handle_unfold_all_action);
            }
        }
    }

    // if (!item || isNet)
    // {
        // QAction* antialiasing_action = context_menu.addAction("Antialiasing");
        // QAction* cosmetic_action     = context_menu.addAction("Cosmetic Nets");
        // QMenu* grid_menu = context_menu.addMenu("Grid");
        // QMenu* type_menu = grid_menu->addMenu("Type");
        // QMenu* cluster_menu          = grid_menu->addMenu("Clustering");
        // QAction* lines_action        = type_menu->addAction("Lines");
        // QAction* dots_action         = type_menu->addAction("Dots");
        // QAction* none_action         = type_menu->addAction("None");
        // connect(action, &QAction::triggered, this, SLOT);
    // }

    context_menu.exec(mapToGlobal(pos));
    update();
}

void graph_graphics_view::update_matrix(const int delta)
{
    qreal scale = qPow(2.0, delta / 100.0);

    QMatrix matrix;
    matrix.scale(scale, scale);
    setMatrix(matrix);
}

void graph_graphics_view::toggle_antialiasing()
{
    setRenderHint(QPainter::Antialiasing, !(renderHints() & QPainter::Antialiasing));
}

bool graph_graphics_view::item_draggable(graphics_item* item)
{
    hal::item_type type = item->item_type();
    return type == hal::item_type::gate || type == hal::item_type::module;
}

void graph_graphics_view::gentle_zoom(const qreal factor)
{
    scale(factor, factor);
    centerOn(target_scene_pos);
    QPointF delta_viewport_pos = target_viewport_pos - QPointF(viewport()->width() / 2.0, viewport()->height() / 2.0);
    QPointF viewport_center    = mapFromScene(target_scene_pos) - delta_viewport_pos;
    centerOn(mapToScene(viewport_center.toPoint()));
}

void graph_graphics_view::handle_select_outputs()
{
    auto context           = m_graph_widget->get_context();
    QAction* sender_action = dynamic_cast<QAction*>(sender());
    if (sender_action)
    {
        QSet<u32> gates;
        for (auto sel_id : g_selection_relay.m_selected_gates)
        {
            auto gate = g_netlist->get_gate_by_id(sel_id);
            for (const auto& net : gate->get_fan_out_nets())
            {
                for (const auto& suc : net->get_dsts())
                {
                    bool found = false;
                    for (const auto& id : context->modules())
                    {
                        auto m = g_netlist->get_module_by_id(id);
                        if (m->contains_gate(suc.gate, true))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        gates.insert(suc.gate->get_id());
                    }
                }
            }
        }
        for (auto sel_id : g_selection_relay.m_selected_modules)
        {
            auto module = g_netlist->get_module_by_id(sel_id);
            for (const auto& net : module->get_output_nets())
            {
                for (const auto& suc : net->get_dsts())
                {
                    bool found = false;
                    for (const auto& id : context->modules())
                    {
                        auto m = g_netlist->get_module_by_id(id);
                        if (m->contains_gate(suc.gate, true))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        gates.insert(suc.gate->get_id());
                    }
                }
            }
        }

        context->add({}, gates);
    }
}
void graph_graphics_view::handle_select_inputs()
{
    auto context           = m_graph_widget->get_context();
    QAction* sender_action = dynamic_cast<QAction*>(sender());
    if (sender_action)
    {
        QSet<u32> gates;
        for (auto sel_id : g_selection_relay.m_selected_gates)
        {
            auto gate = g_netlist->get_gate_by_id(sel_id);
            for (const auto& net : gate->get_fan_in_nets())
            {
                if (net->get_src().gate != nullptr)
                {
                    bool found = false;
                    for (const auto& id : context->modules())
                    {
                        auto m = g_netlist->get_module_by_id(id);
                        if (m->contains_gate(net->get_src().gate, true))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        gates.insert(net->get_src().gate->get_id());
                    }
                }
            }
        }
        for (auto sel_id : g_selection_relay.m_selected_modules)
        {
            auto module = g_netlist->get_module_by_id(sel_id);
            for (const auto& net : module->get_input_nets())
            {
                if (net->get_src().gate != nullptr)
                {
                    bool found = false;
                    for (const auto& id : context->modules())
                    {
                        auto m = g_netlist->get_module_by_id(id);
                        if (m->contains_gate(net->get_src().gate, true))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        gates.insert(net->get_src().gate->get_id());
                    }
                }
            }
        }
        context->add({}, gates);
    }
}

void graph_graphics_view::handle_global_setting_changed(void* sender, const QString& key, const QVariant& value)
{
    UNUSED(sender);
    if (key == "graph_view/drag_mode_modifier")
    {
        unsigned int modifier = value.toUInt();
        m_drag_modifier       = Qt::KeyboardModifier(modifier);
    }
    else if (key == "graph_view/move_modifier")
    {
        unsigned int modifier = value.toUInt();
        m_move_modifier       = Qt::KeyboardModifier(modifier);
    }
}

void graph_graphics_view::handle_fold_single_action()
{
    auto context = m_graph_widget->get_context();
    context->fold_module_of_gate(m_item->id());
}

void graph_graphics_view::handle_unfold_single_action()
{
    auto context = m_graph_widget->get_context();
    auto m       = g_netlist->get_module_by_id(m_item->id());
    if (m->get_gates().empty() && m->get_submodules().empty())
    {
        QMessageBox msg;
        msg.setText("This module is empty.\nYou can't unfold it.");
        msg.setWindowTitle("Error");
        msg.exec();
        return;
        // We would otherwise unfold the empty module into nothing, so the user
        // would have nowhere to click to get their module back
    }
    context->unfold_module(m_item->id());
}

void graph_graphics_view::handle_fold_all_action()
{
    auto context = m_graph_widget->get_context();

    context->begin_change();
    for (u32 id : g_selection_relay.m_selected_gates)
    {
        context->fold_module_of_gate(id);
    }
    context->end_change();
}

void graph_graphics_view::handle_unfold_all_action()
{
    auto context = m_graph_widget->get_context();

    context->begin_change();
    for (u32 id : g_selection_relay.m_selected_modules)
    {
        context->unfold_module(id);
    }
    context->end_change();
}
