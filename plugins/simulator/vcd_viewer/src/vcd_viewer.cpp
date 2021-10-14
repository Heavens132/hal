#include "vcd_viewer/vcd_viewer.h"

#include "vcd_viewer/wave_widget.h"
#include "netlist_simulator_controller/wave_data.h"
#include "netlist_simulator_controller/plugin_netlist_simulator_controller.h"
#include "netlist_simulator_controller/simulation_engine.h"

#include "netlist_simulator_controller/vcd_serializer.h"
#include "vcd_viewer/gate_selection_dialog.h"
#include "vcd_viewer/plugin_vcd_viewer.h"
#include "vcd_viewer/clock_set_dialog.h"
#include "hal_core/netlist/module.h"
#include "hal_core/netlist/gate.h"
#include "hal_core/netlist/net.h"
#include "hal_core/netlist/netlist.h"
#include "hal_core/utilities/log.h"
#include "hal_core/plugin_system/plugin_manager.h"
#include "hal_version.h"
#include "gui/netlist_relay/netlist_relay.h"
#include "gui/gui_globals.h"
#include "gui/gui_utils/graphics.h"
#include "gui/toolbar/toolbar.h"

#include <QFile>
#include <QDate>
#include <QDebug>
#include <QColor>
#include <QFileDialog>
#include <QStatusBar>
#include <QAction>
#include <QMenu>
#include <QVBoxLayout>
#include "hal_core/plugin_system/plugin_manager.h"
#include "netlist_simulator/plugin_netlist_simulator.h"
#include "netlist_simulator/netlist_simulator.h"
#include "netlist_simulator_controller/simulation_input.h"

namespace hal
{

    ContentWidget* VcdViewerFactory::contentFactory() const
    {
        return new VcdViewer;
    }

    VcdViewer::VcdViewer(QWidget *parent)
        : ContentWidget("VcdViewer",parent),
          mVisualizeNetState(false)
    {
        mCreateControlAction = new QAction(this);
        mSimulSettingsAction = new QAction(this);
        mOpenInputfileAction = new QAction(this);
        mRunSimulationAction = new QAction(this);

        mCreateControlAction->setIcon(gui_utility::getStyledSvgIcon("all->#808080",":/icons/plus"));
        mSimulSettingsAction->setIcon(gui_utility::getStyledSvgIcon("all->#808080",":/icons/preferences"));
        mOpenInputfileAction->setIcon(gui_utility::getStyledSvgIcon("all->#FFFFFF",":/icons/folder"));
        mRunSimulationAction->setIcon(gui_utility::getStyledSvgIcon("all->#20FF80",":/icons/run"));

        mCreateControlAction->setToolTip("Create simulation controller");
        mSimulSettingsAction->setToolTip("Simulation settings");
        mOpenInputfileAction->setToolTip("Open input file");
        mRunSimulationAction->setToolTip("Run Simulation");

        connect(mCreateControlAction, &QAction::triggered, this, &VcdViewer::handleCreateControl);
        connect(mSimulSettingsAction, &QAction::triggered, this, &VcdViewer::handleSimulSettings);
        connect(mOpenInputfileAction, &QAction::triggered, this, &VcdViewer::handleOpenInputFile);
        connect(mRunSimulationAction, &QAction::triggered, this, &VcdViewer::handleRunSimulation);

        mTabWidget = new QTabWidget(this);
        mTabWidget->setTabsClosable(true);
        mTabWidget->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
        connect(mTabWidget,&QTabWidget::tabCloseRequested,this,&VcdViewer::handleTabClosed);
        mContentLayout->addWidget(mTabWidget);
//        mWaveWidget = new WaveWidget(this);
//        mWaveWidget->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
//        mContentLayout->addWidget(mWaveWidget);
        mStatusBar = new QStatusBar(this);
        mStatusBar->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Preferred);
        mContentLayout->addWidget(mStatusBar);
        if (gSelectionRelay)
            connect(gSelectionRelay,&SelectionRelay::selectionChanged,this,&VcdViewer::handleSelectionChanged);

        NetlistSimulatorControllerMap* nscm = NetlistSimulatorControllerMap::instance();
        connect(nscm, &NetlistSimulatorControllerMap::controllerAdded, this, &VcdViewer::handleControllerAdded);
        connect(nscm, &NetlistSimulatorControllerMap::controllerRemoved, this, &VcdViewer::handleControllerRemoved);
        displayStatusMessage();
    }

    void VcdViewer::displayStatusMessage(const QString& msg)
    {
        if (msg.isEmpty())
        {
            if (mTabWidget->count())
            {
                WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->currentWidget());
                mStatusBar->showMessage("Setup simulation controller " + ww->controller()->name());
            }
            else
                mStatusBar->showMessage("Create new simulation controller");
        }
        else
            mStatusBar->showMessage(msg);
    }

    void VcdViewer::initSimulator()
    {
        NetlistSimulatorControllerPlugin* simControlPlug = static_cast<NetlistSimulatorControllerPlugin*>(plugin_manager::get_plugin_instance("netlist_simulator_controller"));
        if (!simControlPlug)
        {
            qDebug() << "Plugin 'netlist_simulator_controller' not found";
            return;
        }
        qDebug() << "access to plugin" << simControlPlug->get_name().c_str() << simControlPlug->get_version().c_str();

        mController = simControlPlug->create_simulator_controller();

        /*
        mSimulator = simP->get_shared_simulator("vcd_viewer");
        if (!mSimulator)
        {
            qDebug() << "Cannot create new simulator";
            return;
        }
        */
        mController->reset();
        qDebug() << "sim has gates " << mController->get_gates().size();
        if (!gNetlist)
        {
            qDebug() << "No netlist loaded";
            return;
        }
        qDebug() << "net has gates " << gNetlist->get_gates().size();
        mController->add_gates(mSimulateGates);

        // TODO : WaveData from WaveDataList

        for (const Net* inet : mController->get_input_nets())
            mInputNets.append(inet);

        for (const Net* n : mInputNets)
        {
            WaveData* wd = new WaveData(n);
            wd->insert(0,0);
// TODO            mWaveWidget->addOrReplaceWave(wd);
        }

        // mController->set_iteration_timeout(1000);
 //       setState(SimulationClockSet);
    }

    void VcdViewer::setupToolbar(Toolbar* toolbar)
    {
        toolbar->addAction(mCreateControlAction);
        toolbar->addAction(mSimulSettingsAction);
        toolbar->addAction(mOpenInputfileAction);
        toolbar->addAction(mRunSimulationAction);

    }

    void VcdViewer::handleTabClosed(int inx)
    {
        WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->widget(inx));
        if (!ww->triggerClose())
            log_warning(ww->controller()->get_name(), "Cannot close tab for externally owned controller.");
    }

    void VcdViewer::handleCreateControl()
    {
        NetlistSimulatorControllerPlugin* ctrlPlug = static_cast<NetlistSimulatorControllerPlugin*>(plugin_manager::get_plugin_instance("netlist_simulator_controller"));
        if (!ctrlPlug)
        {
            log_warning("vcd_viewer", "Plugin 'netlist_simulator_controller' not found");
            return;
        }
        std::unique_ptr<NetlistSimulatorController> ctrlRef = ctrlPlug->create_simulator_controller();
        u32 ctrlId = ctrlRef.get()->get_id();
        for (int inx=0; inx<mTabWidget->count(); inx++)
        {
            WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->widget(inx));
            qDebug() << "search controller" << ctrlId << ww->controllerId();
            if (ctrlId == ww->controllerId())
            {
                ww->takeOwnership(ctrlRef);
                break;
            }
        }
    }

    void VcdViewer::handleSimulSettings()
    {
         QMenu* settingMenu = new QMenu(this);
         QAction* act;
         act = new QAction("Select gates for simulation", settingMenu);
         connect(act, &QAction::triggered, this, &VcdViewer::handleSelectGates);
         settingMenu->addAction(act);
         act = new QAction("Select clock net", settingMenu);
         connect(act, &QAction::triggered, this, &VcdViewer::handleClockSet);
  // TODO       act->setEnabled(mState==SimulationClockSet);
         settingMenu->addAction(act);
         settingMenu->addSeparator();
         QMenu* engineMenu = settingMenu->addMenu("Select engine ...");
         QActionGroup* engineGroup = new QActionGroup(this);
         engineGroup->setExclusive(true);
         for (SimulationEngineFactory* sef : *SimulationEngineFactories::instance())
         {
             act = new QAction(QString::fromStdString(sef->name()), engineMenu);
             act->setCheckable(true);
             engineMenu->addAction(act);
             engineGroup->addAction(act);
         }
         settingMenu->addSeparator();
         act = new QAction("Visualize net state by color", settingMenu);
         act->setCheckable(true);
         act->setChecked(mVisualizeNetState);
         connect (act, &QAction::triggered, this, &VcdViewer::setVisualizeNetState);
         settingMenu->addAction(act);

         settingMenu->exec(mapToGlobal(QPoint(10,3)));
    }

    void VcdViewer::setVisualizeNetState(bool state)
    {
        if (state == mVisualizeNetState) return;
        mVisualizeNetState = state;
        for (int inx=0; inx < mTabWidget->count(); inx++)
        {
            WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->widget(inx));
            ww->setVisualizeNetState(mVisualizeNetState, inx==mTabWidget->currentIndex());
        }
    }

    void VcdViewer::handleControllerAdded(u32 controllerId)
    {
        NetlistSimulatorController* nsc = NetlistSimulatorControllerMap::instance()->controller(controllerId);
        if (!nsc) return;
        WaveWidget* ww = new WaveWidget(nsc, mTabWidget);
        mTabWidget->addTab(ww,nsc->name());
        displayStatusMessage();
    }

    void VcdViewer::handleControllerRemoved(u32 controllerId)
    {
        for (int inx=0; inx<mTabWidget->count(); inx++)
        {
            WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->widget(inx));
            if (ww->controllerId() == controllerId)
            {
                mTabWidget->removeTab(inx);
                ww->deleteLater();
            }
        }
        displayStatusMessage();
    }

    void VcdViewer::handleOpenInputFile()
    {
        if (!mTabWidget->count()) return;
        WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->currentWidget());
        if (!ww) return;
        QString filename =
                QFileDialog::getOpenFileName(this, "Load input wave file", ".", ("VCD Files (*.vcd)") );
        if (filename.isEmpty()) return;
        ww->controller()->parse_vcd(filename.toStdString());
    }

    void VcdViewer::handleRunSimulation()
    {
        mResultMap.clear();
        // TODO
        /*
        if (mState != SimulationInputGenerate)
        {
            qDebug() << "wrong state" << mState;
            return;
        }
        QMultiMap<int,QPair<const Net*, BooleanFunction::Value>> inputMap;
        for (const Net* n : mController->get_input_nets())
        {
            if (n==mClkNet) continue;
            const WaveData* wd = mWaveWidget->waveDataByNetId(n->get_id());
            for (auto it=wd->constBegin(); it != wd->constEnd(); ++it)
            {
                BooleanFunction::Value sv;
                switch (it.value())
                {
                case -2: sv = BooleanFunction::Value::Z; break;
                case -1: sv = BooleanFunction::Value::X; break;
                case  0: sv = BooleanFunction::Value::ZERO; break;
                case  1: sv = BooleanFunction::Value::ONE; break;
                default: continue;
                }
                inputMap.insertMulti(it.key(),QPair<const Net*,BooleanFunction::Value>(n,sv));
            }
        }

         * TODO start simulation
        int t=0;
        for (auto it = inputMap.begin(); it != inputMap.end(); ++it)
        {
            if (it.key() != t)
            {
                mSimulator->simulate(it.key() - t);
                t = it.key();
            }
            mSimulator->set_input(it.value().first,it.value().second);
        }

        for (Net* n : gNetlist->get_nets())
        {
             TODO get data from engine
            WaveData* wd = WaveData::simulationResultFactory(n, mSimulator.get());
            if (wd) mResultMap.insert(wd->id(),wd);

        }


        mSimulator->generate_vcd("result.vcd",0,t);

        VcdSerializer reader(this);
        for (WaveData* wd : reader.deserialize("result.vcd"))
            mResults.insert(wd->name(),wd);
            */
        qDebug() << "results" << mResultMap.size();
     //   setState(SimulationShowResults);
    }

    void VcdViewer::handleClockSet()
    {
        ClockSetDialog csd(mInputNets, this);
        if (csd.exec() != QDialog::Accepted) return;

        int period = csd.period();
        if (period <= 0) return;

        const Net* clk = mInputNets.at(csd.netIndex());

        WaveWidget* ww = static_cast<WaveWidget*>(mTabWidget->currentWidget());
        ww->controller()->add_clock_period(clk,csd.duration(),csd.startValue()==0);

        // TODO : ww->update() ?
   }

    void VcdViewer::handleSelectGates()
    {
        GateSelectionDialog gsd(this);
        if (gsd.exec() != QDialog::Accepted) return;

        mSimulateGates = gsd.selectedGates();
        initSimulator();
    }

    void VcdViewer::handleSelectionChanged(void* sender)
    {
        Q_UNUSED(sender);
        for (u32 nid : gSelectionRelay->selectedNetsList())
        {
            Net* n = gNetlist->get_net_by_id(nid);
            if (!n) continue;
            const WaveData* wd = mResultMap.value(n->get_id());
            if (!wd) continue;
            WaveData* wdCopy = new WaveData(*wd);
// TODO             mWaveWidget->addOrReplaceWave(wdCopy);
        }
    }
}    // namespace hal