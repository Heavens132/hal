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

#include <QFont>
#include <QWidget>
#include <QLabel>

namespace hal
{
    class DetailsWidget : public QWidget
    {
        Q_OBJECT
    public:
        enum DetailsType
        {
            ModuleDetails,
            GateDetails,
            NetDetails
        };

        explicit DetailsWidget(DetailsType tp, QWidget* parent = nullptr);

        QFont keyFont() const;
        u32 currentId() const;
        QString detailsTypeName() const;
        QLabel* bigIcon();

        virtual void hideSectionsWhenEmpty(bool hide);

    protected:
        DetailsType mDetailsType;
        u32 mCurrentId;
        QFont mKeyFont;
        bool mHideEmptySections;
        QLabel* mBigIcon;
    };
}    // namespace hal
