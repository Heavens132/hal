#include "gui/user_action/action_remove_items_from_object.h"
#include "gui/user_action/action_add_items_to_object.h"
#include "gui/graph_widget/contexts/graph_context.h"
#include "gui/gui_globals.h"
#include "hal_core/netlist/grouping.h"

namespace hal
{
    ActionRemoveItemsFromObjectFactory::ActionRemoveItemsFromObjectFactory()
       : UserActionFactory("RemoveItemsFromObject") {;}

    ActionRemoveItemsFromObjectFactory* ActionRemoveItemsFromObjectFactory::sFactory = new ActionRemoveItemsFromObjectFactory;

    UserAction* ActionRemoveItemsFromObjectFactory::newAction() const
    {
        return new ActionRemoveItemsFromObject;
    }

    QString ActionRemoveItemsFromObject::tagname() const
    {
        return ActionRemoveItemsFromObjectFactory::sFactory->tagname();
    }

    void ActionRemoveItemsFromObject::addToHash(QCryptographicHash& cryptoHash) const
    {
        cryptoHash.addData("mod",3);
        cryptoHash.addData(setToText(mModules).toUtf8());
        cryptoHash.addData("gat",3);
        cryptoHash.addData(setToText(mGates).toUtf8());
        cryptoHash.addData("net",3);
        cryptoHash.addData(setToText(mNets).toUtf8());
    }

    void ActionRemoveItemsFromObject::writeToXml(QXmlStreamWriter& xmlOut) const
    {
        if (!mModules.isEmpty()) xmlOut.writeTextElement("modules",setToText(mModules));
        if (!mGates.isEmpty())   xmlOut.writeTextElement("gates",setToText(mGates));
        if (!mNets.isEmpty())    xmlOut.writeTextElement("nets",setToText(mNets));
    }

    void ActionRemoveItemsFromObject::readFromXml(QXmlStreamReader& xmlIn)
    {
        while (xmlIn.readNextStartElement())
        {
            if (xmlIn.name()=="modules")
                mModules = setFromText(xmlIn.readElementText());
            else if (xmlIn.name()=="gates")
                mGates = setFromText(xmlIn.readElementText());
            else if (xmlIn.name()=="nets")
                mNets = setFromText(xmlIn.readElementText());
        }
    }

    bool ActionRemoveItemsFromObject::exec()
    {
        mUndoAction = nullptr;
        GraphContext* ctx;
        Grouping*     grp;
        switch (mObject.type())
        {
        case UserActionObjectType::Context:
            ctx = gGraphContextManager->getContextById(mObject.id());
            if (ctx)
            {
                ActionAddItemsToObject* undo =
                        new ActionAddItemsToObject(mModules,mGates,mNets);
                undo->setObject(mObject);
                PlacementHint plc(PlacementHint::GridPosition);
                for (u32 id : mModules)
                {
                    Node nd(id,Node::Module);
                    QPoint p = ctx->getLayouter()->nodeToPositionMap().value(nd);
                    plc.addGridPosition(nd,p);
                }
                for (u32 id : mGates)
                {
                    Node nd(id,Node::Gate);
                    QPoint p = ctx->getLayouter()->nodeToPositionMap().value(nd);
                    plc.addGridPosition(nd,p);
                }
                undo->setPlacementHint(plc);
                mUndoAction = undo;
                ctx->beginChange();
                ctx->remove(mModules, mGates);
                ctx->endChange();
            }
            else
                return false;
            break;
        case UserActionObjectType::Grouping:
            grp = gNetlist->get_grouping_by_id(mObject.id());
            if (grp)
            {
                for (u32 id : mModules)
                    grp->remove_module_by_id(id);
                for (u32 id : mGates)
                    grp->remove_gate_by_id(id);
                for (u32 id : mNets)
                    grp->remove_net_by_id(id);
            }
            else
                return false;
            break;
        default:
            return false;
        }

        if (!mUndoAction)
        {
            mUndoAction = new ActionAddItemsToObject(mModules,mGates,mNets);
            mUndoAction->setObject(mObject);
        }
        return UserAction::exec();
    }
}
