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

#include "hal_core/defines.h"

#include "gui/graph_widget/contexts/graph_context.h"
#include "gui/context_manager_widget/models/context_table_model.h"
#include "gui/context_manager_widget/models/context_table_proxy_model.h"
#include "gui/searchbar/searchbar.h"

#include <QListWidget>
#include <QPoint>
#include <QTableView>
#include <QPushButton>
#include <QMenu>


namespace hal
{
    class GraphContext;
    class GraphTabWidget;

    /**
     * @ingroup gui
     * TODO
     */
    class ContextManagerWidget : public ContentWidget
    {
        Q_OBJECT
        Q_PROPERTY(QString disabledIconStyle READ disabledIconStyle WRITE setDisabledIconStyle)
        Q_PROPERTY(QString newViewIconPath READ newViewIconPath WRITE setNewViewIconPath)
        Q_PROPERTY(QString newViewIconStyle READ newViewIconStyle WRITE setNewViewIconStyle)
        Q_PROPERTY(QString renameIconPath READ renameIconPath WRITE setRenameIconPath)
        Q_PROPERTY(QString renameIconStyle READ renameIconStyle WRITE setRenameIconStyle)
        Q_PROPERTY(QString deleteIconPath READ deleteIconPath WRITE setDeleteIconPath)
        Q_PROPERTY(QString deleteIconStyle READ deleteIconStyle WRITE setDeleteIconStyle)
        Q_PROPERTY(QString duplicateIconPath READ duplicateIconPath WRITE setDuplicateIconPath)
        Q_PROPERTY(QString duplicateIconStyle READ duplicateIconStyle WRITE setDuplicateIconStyle)
        Q_PROPERTY(QString openIconPath READ openIconPath WRITE setOpenIconPath)
        Q_PROPERTY(QString openIconStyle READ openIconStyle WRITE setOpenIconStyle)
        Q_PROPERTY(QString searchIconPath READ searchIconPath WRITE setSearchIconPath)
        Q_PROPERTY(QString searchIconStyle READ searchIconStyle WRITE setSearchIconStyle)

    public:
        ContextManagerWidget(GraphTabWidget* tab_view, QWidget* parent = nullptr);

        void selectViewContext(GraphContext* context);
        GraphContext* getCurrentContext();

        virtual void setupToolbar(Toolbar* toolbar) override;

        QString disabledIconStyle() const;
        QString newViewIconPath() const;
        QString newViewIconStyle() const;
        QString renameIconPath() const;
        QString renameIconStyle() const;
        QString deleteIconPath() const;
        QString deleteIconStyle() const;
        QString duplicateIconPath() const;
        QString duplicateIconStyle() const;
        QString openIconPath() const;
        QString openIconStyle() const;
        QString searchIconPath() const;
        QString searchIconStyle() const;

        void setDisabledIconStyle(const QString &path);
        void setNewViewIconPath(const QString &path);
        void setNewViewIconStyle(const QString &style);
        void setRenameIconPath(const QString &path);
        void setRenameIconStyle(const QString &style);
        void setDeleteIconPath(const QString &path);
        void setDeleteIconStyle(const QString &style);
        void setDuplicateIconPath(const QString &path);
        void setDuplicateIconStyle(const QString &style);
        void setOpenIconPath(const QString &path);
        void setOpenIconStyle(const QString &style);
        void setSearchIconPath(const QString &path);
        void setSearchIconStyle(const QString &style);

        void handleOpenContextClicked();
    public Q_SLOTS:
        //void handleContextCreated(GraphContext* context);
        //void handleContextRenamed(GraphContext* context);
        //void handleContextRemoved(GraphContext* context);

    private:
        GraphTabWidget* mTabView;

        QTableView* mContextTableView;
        ContextTableModel* mContextTableModel;
        ContextTableProxyModel* mContextTableProxyModel;

        Searchbar mSearchbar;

        QString mDisabledIconStyle;

        QAction* mNewViewAction;
        QString mNewViewIconPath;
        QString mNewViewIconStyle;

        QAction* mRenameAction;
        QString mRenameIconPath;
        QString mRenameIconStyle;

        QAction* mDuplicateAction;
        QString mDuplicateIconPath;
        QString mDuplicateIconStyle;

        QAction* mDeleteAction;
        QString mDeleteIconPath;
        QString mDeleteIconStyle;

        QAction* mOpenAction;
        QString mOpenIconPath;
        QString mOpenIconStyle;

        QString mSearchIconPath;
        QString mSearchIconStyle;

        void handleCreateContextClicked();
        void handleRenameContextClicked();
        void handleDuplicateContextClicked();
        void handleDeleteContextClicked();

        void handleContextMenuRequest(const QPoint& point);
        void handleSelectionChanged(const QItemSelection &selected, const QItemSelection &deselected);
        void handleFilterTextChanged(const QString& filter_text);

        void setToolbarButtonsEnabled(bool enabled);

        void toggleSearchbar();

        QList<QShortcut*> createShortcuts() override;
    };
}
