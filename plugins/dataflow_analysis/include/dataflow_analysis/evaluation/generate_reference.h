#pragma once

#include "dataflow_analysis/common/grouping.h"
#include "hal_core/defines.h"

namespace hal
{
    namespace evaluation
    {
        std::shared_ptr<hal::dataflow::Grouping> generate_reference(const hal::dataflow::NetlistAbstraction& netlist_abstr);

    }    // namespace evaluation
}    // namespace hal

