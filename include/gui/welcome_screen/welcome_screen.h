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

#include "QFrame"

class QHBoxLayout;

namespace hal
{
    class GetInTouchWidget;
    class OpenFileWidget;
    class RecentFilesWidget;
    class LabeledFrame;

    class WelcomeScreen : public QFrame
    {
        Q_OBJECT

    public:
        explicit WelcomeScreen(QWidget* parent = nullptr);

        void repolish();

        //    void intro();
        //    void reset();

    private:
        QHBoxLayout* m_layout;

        LabeledFrame* m_recent_files_frame;
        LabeledFrame* m_open_file_frame;
        LabeledFrame* m_get_in_touch_frame;

        RecentFilesWidget* m_recent_files_widget;
        OpenFileWidget* m_open_file_widget;
        GetInTouchWidget* m_get_in_touch_widget;
    };
}