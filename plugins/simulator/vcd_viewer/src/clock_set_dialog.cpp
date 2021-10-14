#include "vcd_viewer/clock_set_dialog.h"
#include "hal_core/netlist/net.h"
#include <QGridLayout>
#include <QLabel>

namespace hal {

    ClockSetDialog::ClockSetDialog(const QList<const Net*>& inputs, QWidget *parent)
        : QDialog(parent)
    {
        QGridLayout* layout = new QGridLayout(this);
        mComboNet = new QComboBox(this);
        int j = 0;
        int iclk = -1;
        for (const Net* n : inputs)
        {
            QString netName = QString::fromStdString(n->get_name());
            if (netName.toUpper().contains("CLK")) iclk = j;
            mComboNet->insertItem(j++,QString("%1[%2]").arg(netName).arg(n->get_id()));
        }
        if (iclk >= 0) mComboNet->setCurrentIndex(iclk);
        layout->addWidget(new QLabel("Select clock net:",this),0,0);
        layout->addWidget(mComboNet,0,1);

        layout->addWidget(new QLabel("Clock period:",this),1,0);
        mSpinPeriod = new QSpinBox(this);
        mSpinPeriod->setMinimum(0);
        mSpinPeriod->setMaximum(1000000);
        mSpinPeriod->setValue(10);
        layout->addWidget(mSpinPeriod,1,1);

        layout->addWidget(new QLabel("Start value:",this),2,0);
        mSpinStartValue = new QSpinBox(this);
        mSpinStartValue->setMinimum(0);
        mSpinStartValue->setMaximum(1);
        layout->addWidget(mSpinStartValue,2,1);

        layout->addWidget(new QLabel("Duration:",this),3,0);
        mSpinDuration = new QSpinBox(this);
        mSpinDuration->setMinimum(0);
        mSpinDuration->setMaximum(1000000);
        mSpinDuration->setValue(2000);
        layout->addWidget(mSpinDuration,3,1);

        mButtonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        connect(mButtonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(mButtonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
        layout->addWidget(mButtonBox,4,1);
    }
}