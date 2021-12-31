#pragma once

#include "dataflow_analysis/common/grouping.h"
#include "dataflow_analysis/evaluation/result.h"
#include "dataflow_analysis/evaluation/similarity_score.h"
#include "dataflow_analysis/processing/result.h"

namespace hal
{
    namespace evaluation
    {
        SimilarityScore compare_to_reference(const std::shared_ptr<hal::dataflow::Grouping>& test_state, const std::shared_ptr<hal::dataflow::Grouping>& ref_state);

        std::unordered_map<std::shared_ptr<hal::dataflow::Grouping>, SimilarityScore>
            compare_to_reference(const hal::dataflow::processing::Result& processing_result, const hal::dataflow::evaluation::Result& eval_result, const std::shared_ptr<hal::dataflow::Grouping>& ref_state);
    }    // namespace evaluation
}    // namespace hal
