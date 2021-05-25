#pragma once
#include "user_action.h"

namespace hal
{
    /**
     * @ingroup user_action
     */
    class ActionPythonExecuteFile : public UserAction
    {
    public:
        ActionPythonExecuteFile();
        QString tagname() const override;
        bool exec() override;
        void writeToXml(QXmlStreamWriter& xmlOut) const override;
        void readFromXml(QXmlStreamReader& xmlIn) override;
        void addToHash(QCryptographicHash& cryptoHash) const override;
    };

    /**
     * @ingroup user_action
     */
    class ActionPythonExecuteFileFactory : public UserActionFactory
    {
    public:
        ActionPythonExecuteFileFactory();
        UserAction* newAction() const;
        static ActionPythonExecuteFileFactory* sFactory;
    };
}
