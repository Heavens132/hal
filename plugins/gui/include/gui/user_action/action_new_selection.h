#pragma once
#include "user_action.h"

namespace hal
{
    class ActionNewSelection : public UserAction
    {
    public:
        ActionNewSelection() {;}
        QString tagname() const override;
        void writeToXml(QXmlStreamWriter& xmlOut) const override;
        void readFromXml(QXmlStreamReader& xmlIn) override;
        void exec() override;
    };

    class ActionNewSelectionFactory : public UserActionFactory
    {
    public:
        ActionNewSelectionFactory();
        UserAction* newAction() const;
        static ActionNewSelectionFactory* sFactory;
    };
}
