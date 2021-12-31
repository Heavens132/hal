//  MIT License
//
//  Copyright (c) 2019 Ruhr University Bochum, Chair for Embedded Security. All Rights reserved.
//  Copyright (c) 2021 Max Planck Institute for Security and Privacy. All Rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "dataflow_analysis/common/grouping.h"
#include "dataflow_analysis/processing/pass_collection.h"
#include "hal_core/defines.h"

#include <map>

<<<<<<< HEAD


=======
>>>>>>> f9875986779e3da88a60f66fc7667a162ca19e66
namespace hal
{
    namespace dataflow
    {
        namespace processing
        {
            struct Result
            {
                std::vector<std::shared_ptr<Grouping>> unique_groupings;
                std::map<std::shared_ptr<Grouping>, std::vector<std::vector<pass_id>>> pass_combinations_leading_to_grouping;
                std::map<std::vector<pass_id>, std::shared_ptr<Grouping>> groupings;
<<<<<<< HEAD
                // a vector containing each group of every grouping 
                // each group is a vector containing 1's only in the corresponding indexes(gates)
                std::vector<std::vector<u32>> groups_embedding;
                std::vector<std::set<u32>> all_results;
                std::vector<u32> s_gates;
                std::map<u32, u32> s_gates_maping;

=======
>>>>>>> f9875986779e3da88a60f66fc7667a162ca19e66
            };

        }    // namespace processing
    }        // namespace dataflow
<<<<<<< HEAD
}    // namespace hal

=======
}    // namespace hal
>>>>>>> f9875986779e3da88a60f66fc7667a162ca19e66
