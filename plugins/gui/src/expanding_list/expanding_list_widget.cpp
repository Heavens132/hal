#include "gui/expanding_list/expanding_list_widget.h"

#include "gui/expanding_list/expanding_list_button.h"
#include "gui/expanding_list/expanding_list_item.h"

#include <QFrame>
#include <QStyle>
#include <QVBoxLayout>

namespace hal
{

    ExpandingListWidget::ExpandingListWidget(QWidget* parent)
        : QScrollArea(parent), mContent(new QFrame()), mContentLayout(new QVBoxLayout()), mSpacer(new QFrame()), mSelectedButton(nullptr), mOffset(0)
    {
        setFrameStyle(QFrame::NoFrame);
        setWidget(mContent);
        setWidgetResizable(true);
        mContent->setObjectName("content");
        mContent->setFrameStyle(QFrame::NoFrame);
        mContent->setLayout(mContentLayout);
        mContentLayout->setAlignment(Qt::AlignTop);
        mContentLayout->setContentsMargins(0, 0, 0, 0);
        mContentLayout->setSpacing(0);
        mSpacer->setObjectName("spacer");
        mSpacer->setFrameStyle(QFrame::NoFrame);
        mContentLayout->addWidget(mSpacer);
    }

    void ExpandingListWidget::appendItem(ExpandingListButton* button, const QString& groupName)
    {
        ExpandingListItem* item = new ExpandingListItem(button);
        mItems.append(item);
        mContentLayout->addWidget(item);

        if (!groupName.isEmpty())
            mButtonGroup[groupName].append(item);
        connect(button, &ExpandingListButton::clicked, this, &ExpandingListWidget::handleClicked);
    }

    void ExpandingListWidget::selectButton(ExpandingListButton* button)
    {
        if (button == mSelectedButton)
            return;

        if (!button)
        {
            if (mSelectedButton)
            {
                mSelectedButton->setSelected(false);
                mSelectedButton = nullptr;
            }

            return;
        }

        mSelectedButton = button;
        for (ExpandingListItem* item : mItems)
        {
            ExpandingListButton* but = item->button();
            but->setSelected(but == mSelectedButton);
        }

        Q_EMIT buttonSelected(button);
    }

    void ExpandingListWidget::selectItem(int index)
    {
        if (index < 0 || index >= mItems.size())
            return;

        selectButton(mItems.at(index)->button());
    }

    void ExpandingListWidget::repolish()
    {
        QStyle* s = style();

        s->unpolish(this);
        s->polish(this);

        for (ExpandingListItem* item : mItems)
            item->repolish();
    }

    void ExpandingListWidget::handleClicked()
    {
        QObject* obj                  = sender();
        ExpandingListButton* button = static_cast<ExpandingListButton*>(obj);

        auto grpIt = mButtonGroup.find(button->text());
        if (grpIt != mButtonGroup.end())
        {
            grpIt->toggleCollapsed(mSelectedButton); // except selected
            return;
        }

        selectButton(button);
    }

    bool ExpandingListWidget::hasGroup(const QString& groupName) const
    {
        return mButtonGroup.contains(groupName);
    }

    void ExpandingListGroup::toggleCollapsed(ExpandingListButton* exceptSelected)
    {
        mCollapsed = !mCollapsed;
        for (ExpandingListItem* item : *this)
        {
            if (mCollapsed)
            {
                if (item->button() != exceptSelected)
                    item->collapse();
            }
            else
                item->expand();
        }
    }
}
