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

#include "gui/content_widget/content_widget.h"

#include "gui/graph_widget/contexts/graph_context.h"
#include "gui/graph_widget/graph_widget.h"
#include "gui/graph_widget/graphics_scene.h"

#include <QMap>

class QTabWidget;
class QVBoxLayout;

namespace hal
{
    class SettingsItemDropdown;

    class GraphTabWidget : public ContentWidget
    {
        Q_OBJECT

    public:
        GraphTabWidget(QWidget* parent = nullptr);

        virtual QList<QShortcut*> createShortcuts() override;

        int addTab(QWidget* tab, QString tab_name = "default");
        void showContext(GraphContext* context);

        void ensureSelectionVisible();

        enum KeyboardModifier{Alt, Ctrl, Shift};
        Q_ENUM(KeyboardModifier)

    public Q_SLOTS:
        void handleContextCreated(GraphContext* context);
        void handleContextRenamed(GraphContext* context);
        void handleContextRemoved(GraphContext* context);

        void handleTabChanged(int index);

        void handleGateFocus(u32 gateId);
        void handleNetFocus(u32 netId);
        void handleModuleFocus(u32 moduleId);

    private:
        QTabWidget* mTabWidget;
        QVBoxLayout* mLayout;

        float mZoomFactor;

        QMap<GraphContext*, QWidget*> mContextWidgetMap;

        int getContextTabIndex(GraphContext* context) const;

        //functions
        void handleTabCloseRequested(int index);

        void addGraphWidgetTab(GraphContext* context);

        void zoomInShortcut();
        void zoomOutShortcut();

        SettingsItemDropdown* mSettingGridType;
        SettingsItemDropdown* mSettingDragModifier;
        SettingsItemDropdown* mSettingPanModifier;

        QMap<KeyboardModifier, Qt::KeyboardModifier> mKeyModifierMap;
    };
}
