#pragma once
#include "user_action.h"

namespace hal
{
    class ActionDeleteObject : public UserAction
    {
    public:
        ActionDeleteObject() {;}
        bool exec() override;
        QString tagname() const override;
    };

    class ActionDeleteObjectFactory : public UserActionFactory
    {
    public:
        ActionDeleteObjectFactory();
        UserAction* newAction() const;
        static ActionDeleteObjectFactory* sFactory;
    };
}
