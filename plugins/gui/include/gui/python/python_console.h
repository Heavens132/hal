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

#include "gui/python/python_context_subscriber.h"

#include <QTextEdit>
#include <memory>

namespace hal
{
    class PythonConsoleHistory;

    class PythonConsole : public QTextEdit, public PythonContextSubscriber
    {
        Q_OBJECT

    public:
        PythonConsole(QWidget* parent = nullptr);

        void keyPressEvent(QKeyEvent* e) override;
        void mousePressEvent(QMouseEvent* event) override;

        virtual void handleStdout(const QString& output) override;
        virtual void handleError(const QString& output) override;
        virtual void clear() override;

        void displayPrompt();

        void interpretCommand();
        QString getCurrentCommand();
        void replaceCurrentCommand(const QString& new_command);
        void appendToCurrentCommand(const QString& new_command);

        bool selectionEditable();

        void handleUpKeyPressed();
        void handleDownKeyPressed();

        void handleTabKeyPressed();

        void insertAtEnd(const QString& text, QColor textColor);

    private:
        QColor mPromptColor;
        QColor mStandardColor;
        QColor mErrorColor;

        QString mStandardPrompt;
        QString mCompoundPrompt;

        int mPromptBlockNumber;
        int mPromptLength;
        int mPromptEndPosition;
        int mCompoundPromptEndPosition;

        bool mInCompoundPrompt;
        bool mInCompletion;

        QString mCurrentCompoundInput;
        QString mCurrentInput;

        int mCurrentHistoryIndex;
        int mCurrentCompleterIndex;

        std::shared_ptr<PythonConsoleHistory> mHistory;
    };
}
