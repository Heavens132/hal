#pragma once
#include "user_action.h"

namespace hal
{
    class ActionOpenNetlistFile : public UserAction
    {
        QString mFilename;
    public:
        ActionOpenNetlistFile(const QString& filename_ = QString());
        void exec() override;
        virtual void writeToXml(QXmlStreamWriter& xmlOut) const override;
        virtual void readFromXml(QXmlStreamReader& xmlIn) override;
    };
}
