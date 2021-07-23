#include "gui/study_questionnaire/study_questionnaire.h"
#include "gui/user_action/action_study_questionnaire.h"

#include <QPushButton>
#include <QPlainTextEdit>

#include <QDateTime>

#include <QDebug>

namespace hal
{
    StudyQuestionnaire* StudyQuestionnaire::inst = nullptr;
    const uint mMinInactivityTime = 10 * 60; // min inactivity time is 10 minutes

    StudyQuestionnaire::StudyQuestionnaire(QWidget *parent)
        : QDialog(parent), mHalFocusLost(false), mLastDialogShown(0), mLastHALFocusLost(0), mLastUserActionExecutedTime(0), mMainWindowActivated(true), mMacroPlay(false)
    {
        connect(this, SIGNAL(finished(int)), this, SLOT(questionnaireClosed(int)));
    }

    StudyQuestionnaire* StudyQuestionnaire::instance()
    {
        if (!StudyQuestionnaire::inst)
        {
            StudyQuestionnaire::inst = new StudyQuestionnaire;
            StudyQuestionnaire::inst->initDialog();
            StudyQuestionnaire::inst->mLastDialogShown = QDateTime::currentDateTime().toSecsSinceEpoch();
            StudyQuestionnaire::inst->mLastUserActionExecutedTime = QDateTime::currentDateTime().toSecsSinceEpoch();
        }

        return StudyQuestionnaire::inst;
    }

    void StudyQuestionnaire::initDialog()
    {
        setWindowTitle("HAL Study Questionnaire:");

        // remove questionmark from title bar, it is not implemented and only confusing
        setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
        mLayout = new QVBoxLayout(this);
        QString titleString = QString("No action has been taken in HAL for a while.\nPlease tell us, if you have worked on the task outside of HAL.");
        mLayout->addWidget(new QLabel(titleString,this));

        mCheckBoxLayout = new QVBoxLayout();
        mLayout->addLayout(mCheckBoxLayout);
        addCheckBox(QString("research_library"), QString("I did some research on the library used."));
        addCheckBox(QString("research_hal"), QString("I did some research on the HAL Python API."));
        addCheckBox(QString("python_code_external"), QString("I worked on the python code in an external editor."));
        addCheckBox(QString("explore_netlist_external"), QString("I manually explored the netlist in an external editor."));
        addCheckBox(QString("break"), QString("I took a break."));

        mLayout->addWidget(new QLabel(QString("If you like, please add further information:"),this));
        mFurtherInformation = new QPlainTextEdit();
        mLayout->addWidget(mFurtherInformation);

        QPushButton* sendButton = new QPushButton("Save", this);
        mLayout->addWidget(sendButton);

        connect(sendButton, SIGNAL(clicked()), this, SLOT(accept()));
    }

    void StudyQuestionnaire::addCheckBox(QString shortName, QString description)
    {
        QCheckBox* checkBox = new QCheckBox(description, this);
        mCheckBoxLayout->addWidget(checkBox);
        mQuestionnaireCheckboxes.insert(shortName, checkBox);
    }

    void StudyQuestionnaire::setContentWidgetDeactivated(QString widgetName)
    {
        mContentWidgetsActivated.removeAll(widgetName);
        checkDialog();
    }

    void StudyQuestionnaire::setWindowDeactivated()
    {
        mMainWindowActivated = false;
        checkDialog();
    }

    void StudyQuestionnaire::setContentWidgetActivated(QString widgetName)
    {
        mContentWidgetsActivated.append(widgetName);
        checkDialog();
    }

    void StudyQuestionnaire::setWindowActivated()
    {
        mMainWindowActivated = true;
        checkDialog();
    }

    void StudyQuestionnaire::setUserActionDone(const QString &userActionName)
    {
        if(userActionName != "StudyQuestionnaire") {
            checkDialog();
            mLastUserActionExecutedTime = QDateTime::currentDateTime().toSecsSinceEpoch();
        }
    }

    void StudyQuestionnaire::setMacroPlay(const bool macroPlay_)
    {
        mMacroPlay = macroPlay_;
    }

    void StudyQuestionnaire::checkDialog()
    {
        // don't show if macro is currently executed
        if(mMacroPlay) return;

        uint diffSecs = QDateTime::currentDateTime().toSecsSinceEpoch() - mMinInactivityTime;

        if(!mMainWindowActivated && mContentWidgetsActivated.isEmpty()) {
            mLastHALFocusLost = QDateTime::currentDateTime().toSecsSinceEpoch();
            mHalFocusLost = true;
        } else if(!mHalFocusLost && mLastUserActionExecutedTime < diffSecs && mLastDialogShown < diffSecs) {
            mLastDialogShown = QDateTime::currentDateTime().toSecsSinceEpoch();
            exec();
        } else if(mHalFocusLost) {
            mHalFocusLost = false;
            if(mLastDialogShown < diffSecs && mLastHALFocusLost < diffSecs)
            {
                mLastDialogShown = QDateTime::currentDateTime().toSecsSinceEpoch();
                exec();
            }
        }
    }

    void StudyQuestionnaire::questionnaireClosed(int resultCode)
    {
        mLastDialogShown = QDateTime::currentDateTime().toSecsSinceEpoch();
        if(QDialog::Accepted == resultCode) {
            QMap<QString, QCheckBox*>::iterator it;
            QStringList* listCheckboxes = new QStringList();
            for(it = mQuestionnaireCheckboxes.begin(); it != mQuestionnaireCheckboxes.end(); ++it) {
                QString key = it.key();
                QCheckBox* checkBox = it.value();

                if(checkBox->isChecked()) {
                    listCheckboxes->append(key);
                    checkBox->setChecked(false);
                }
            }

            ActionStudyQuestionnaire* act = new ActionStudyQuestionnaire(*listCheckboxes, mFurtherInformation->toPlainText());
            act->exec();

            mFurtherInformation->setPlainText("");
            delete listCheckboxes;
        }
    }
}
