#include "gui/gui_globals.h"
#include "gui/user_action/action_rename_object.h"
#include "gui/graph_widget/contexts/graph_context.h"
#include "gui/grouping/grouping_manager_widget.h"
#include "gui/grouping/grouping_table_model.h"

namespace hal
{
    ActionRenameObjectFactory::ActionRenameObjectFactory()
       : UserActionFactory("RenameObject") {;}

    ActionRenameObjectFactory* ActionRenameObjectFactory::sFactory = new ActionRenameObjectFactory;

    UserAction* ActionRenameObjectFactory::newAction() const
    {
        return new ActionRenameObject;
    }

    QString ActionRenameObject::tagname() const
    {
        return ActionRenameObjectFactory::sFactory->tagname();
    }

    void ActionRenameObject::addToHash(QCryptographicHash& cryptoHash) const
    {
        cryptoHash.addData(mNewName.toUtf8());
    }

    void ActionRenameObject::writeToXml(QXmlStreamWriter& xmlOut) const
    {
        xmlOut.writeTextElement("name", mNewName);
    }

    void ActionRenameObject::readFromXml(QXmlStreamReader& xmlIn)
    {
        while (xmlIn.readNextStartElement())
        {
            if (xmlIn.name() == "name")
                mNewName = xmlIn.readElementText();
        }
    }

    bool ActionRenameObject::exec()
    {
        QString       oldName;
        Module*       mod;
        Gate*         gat;
        GraphContext* ctx;
        switch (mObject.type()) {
        case UserActionObjectType::Module:
            mod = gNetlist->get_module_by_id(mObject.id());
            if (mod)
            {
                oldName = QString::fromStdString(mod->get_name());
                mod->set_name(mNewName.toStdString());
            }
            else
                return false;
            break;
        case UserActionObjectType::Gate:
            gat = gNetlist->get_gate_by_id(mObject.id());
            if (gat)
            {
                oldName = QString::fromStdString(gat->get_name());
                gat->set_name(mNewName.toStdString());
            }
            else
                return false;
            break;
        case UserActionObjectType::Grouping:
            oldName = gContentManager->getGroupingManagerWidget()->getModel()->
                    renameGrouping(mObject.id(),mNewName);
            break;
        case UserActionObjectType::Context:
            ctx = gGraphContextManager->getContextById(mObject.id());
            if (ctx)
            {
                oldName = ctx->name();
                gGraphContextManager->renameGraphContextAction(ctx,mNewName);
            }
            else
                return false;
            break;
        default:
            return false;
        }
        mUndoAction = new ActionRenameObject(oldName);
        mUndoAction->setObject(mObject);
        return UserAction::exec();
    }
}
