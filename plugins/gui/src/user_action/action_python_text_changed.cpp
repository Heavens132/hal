#include "gui/user_action/action_python_text_changed.h"
#include "gui/user_action/user_action_manager.h"
#include <QTabWidget>
#include "gui/gui_globals.h"
#include "gui/python/python_editor.h"
#include "gui/study_questionnaire/study_questionnaire.h"

namespace hal
{
    ActionPythonTextChangedFactory::ActionPythonTextChangedFactory()
       : UserActionFactory("PythonTextChanged") {;}

    ActionPythonTextChangedFactory* ActionPythonTextChangedFactory::sFactory = new ActionPythonTextChangedFactory;

    UserAction* ActionPythonTextChangedFactory::newAction() const
    {
        return new ActionPythonTextChanged;
    }

    QString ActionPythonTextChanged::tagname() const
    {
        return ActionPythonTextChangedFactory::sFactory->tagname();
    }

    // maximum duration for single text change 10 sec
    qint64 ActionPythonTextChanged::sRecentTextChangeMsec = 10000;

    ActionPythonTextChanged::ActionPythonTextChanged(const u32 &id_, const QString oldtext_, const QString &text_)
        : mOldText(oldtext_), mText(text_), mPythonCodeEditorId(id_), mTextCursorPosition(0), mLastCursorPosition(0),
          mLastKeyIsReturn(false), mMerged(false), mDuration(0)
    {
        if (id_)
            setObject(UserActionObject(id_,UserActionObjectType::PythonCodeEditor));
    }

    bool ActionPythonTextChanged::exec()
    {
        PythonCodeEditor* pythonCodeEditor = gContentManager->getPythonEditorWidget()->getPythonCodeEditorById(mPythonCodeEditorId);
        if (pythonCodeEditor)
            pythonCodeEditor->setOldPlainText(mText);

        if (UserActionManager::instance()->isUserTriggeredAction() && mergeWithRecent())
        {
            // caller should delete this action to avoid memory leak
            mMerged = true;
            return true;
        }
        if(!mUndoAction) {
            ActionPythonTextChanged* act = new ActionPythonTextChanged(mPythonCodeEditorId, "", mOldText);
            act->setTextCursorPosition(mTextCursorPosition,mLastCursorPosition);
            mUndoAction = act;
        }

        if(!UserActionManager::instance()->isUserTriggeredAction()) {
            if(!pythonCodeEditor) {
                gContentManager->getPythonEditorWidget()->newTab(mPythonCodeEditorId);
                pythonCodeEditor = gContentManager->getPythonEditorWidget()->getPythonCodeEditorById(mPythonCodeEditorId);
                if(!pythonCodeEditor) return false;
                pythonCodeEditor->setOldPlainText(mText);
            }
            int tabId = gContentManager->getPythonEditorWidget()->getTabIndexByPythonCodeEditorId(pythonCodeEditor->id());
            // set current index, if wrong tab is selected
            if(gContentManager->getPythonEditorWidget()->getTabWidget()->currentIndex() != tabId)
                gContentManager->getPythonEditorWidget()->getTabWidget()->setCurrentIndex(tabId);

            // set text
            pythonCodeEditor->setPlainText(mText);

            // set text cursor position
            pythonCodeEditor->setCursorPosition(mTextCursorPosition);
        }

        return UserAction::exec();
    }

    bool ActionPythonTextChanged::mergeWithRecent()
    {
        if(mLastKeyIsReturn) return false;

        UserActionManager* uam = UserActionManager::instance();
        Q_ASSERT(uam);
        if (uam->mActionHistory.isEmpty()) return false;
        UserAction* lastAction = uam->mActionHistory.last();
        qint64 msecSinceLastAction = uam->timeStamp() - lastAction->timeStamp();
        if (msecSinceLastAction > sRecentTextChangeMsec) return false;
        ActionPythonTextChanged* lastTextChanged = dynamic_cast<ActionPythonTextChanged*>(lastAction);
        if (!lastTextChanged) return false;
        if (lastTextChanged->mPythonCodeEditorId != mPythonCodeEditorId) return false;
        lastTextChanged->mTimeStamp = uam->timeStamp();
        lastTextChanged->mText      = mText;
        lastTextChanged->mTextCursorPosition = mTextCursorPosition;
        lastTextChanged->mDuration += msecSinceLastAction;

        #ifdef HAL_STUDY
            StudyQuestionnaire::instance()->setUserActionDone(tagname());
        #endif

        return true;
    }

    void ActionPythonTextChanged::addToHash(QCryptographicHash& cryptoHash) const
    {
        cryptoHash.addData(mText.toUtf8());
        cryptoHash.addData(QByteArray::number(mPythonCodeEditorId));
    }

    void ActionPythonTextChanged::writeToXml(QXmlStreamWriter& xmlOut) const
    {
        xmlOut.writeTextElement("pythonscript", mText);
        xmlOut.writeTextElement("uid", QString::number(mPythonCodeEditorId));
        if (mDuration > 0)
            xmlOut.writeTextElement("duration", QString::number(mDuration));
        if (mTextCursorPosition)
            xmlOut.writeTextElement("textcursorposition", QString::number(mTextCursorPosition));
    }

    void ActionPythonTextChanged::readFromXml(QXmlStreamReader& xmlIn)
    {
        while (xmlIn.readNextStartElement())
        {
            if (xmlIn.name() == "pythonscript")
                mText = xmlIn.readElementText();
            if (xmlIn.name() == "uid")
                mPythonCodeEditorId = xmlIn.readElementText().toInt();
            if (xmlIn.name() == "duration")
                // we don't need this value,
                // but we have to read it, to read next start element
                xmlIn.readElementText();
            if (xmlIn.name() == "textcursorposition")
                mTextCursorPosition = xmlIn.readElementText().toInt();
        }
    }

    void ActionPythonTextChanged::setLastKeyIsReturn()
    {
        mLastKeyIsReturn = true;
    }

    void ActionPythonTextChanged::setTextCursorPosition(int lastCursorPosition, int textCursorPosition)
    {
        mLastCursorPosition = lastCursorPosition;
        mTextCursorPosition = textCursorPosition;
    }
}
