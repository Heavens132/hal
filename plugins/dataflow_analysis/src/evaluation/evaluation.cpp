#include "dataflow_analysis/evaluation/evaluation.h"

#include "dataflow_analysis/common/grouping.h"
#include "dataflow_analysis/common/netlist_abstraction.h"
#include "dataflow_analysis/output_generation/textual_output.h"
#include "dataflow_analysis/processing/passes/group_by_control_signals.h"
#include "dataflow_analysis/utils/parallel_for_each.h"
#include "dataflow_analysis/utils/progress_printer.h"
#include "dataflow_analysis/utils/timing_utils.h"
#include "hal_core/netlist/gate.h"
#include "hal_core/netlist/netlist.h"
#include "hal_core/utilities/log.h"

#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <filesystem>
#include<iostream>
//#include <experimental/filesystem>

namespace hal
{
    namespace dataflow
    {
        namespace evaluation
        {
            namespace
            {
            	 std::shared_ptr<Grouping> generate_output(const Configuration& config, const std::shared_ptr<Grouping>& initial_grouping, const processing::Result& result)
                {
                    measure_block_time("majority voting");
                    auto& netlist_abstr = initial_grouping->netlist_abstr;

                    const u32 bad_group_size = 7;
                    
                    std::vector<std::set<u32>> all_results (result.all_results);
                    std::vector<std::vector<u32>> groups_embedding = result.groups_embedding;
                    std::vector<u32> s_gates = result.s_gates;
                    std::map<u32, u32> s_gates_maping = result.s_gates_maping;
                    int gates_num = s_gates.size();

                    // mark all sequential gates as unassigned gates
                    std::vector<u32> unassigned_gates;
                    unassigned_gates.reserve(netlist_abstr.all_sequential_gates.size());
                    std::transform(netlist_abstr.all_sequential_gates.begin(), netlist_abstr.all_sequential_gates.end(), std::back_inserter(unassigned_gates), [](auto& g) { return g->get_id(); });

                    // sort unassignes gates to be able to use std::algorithms
                    std::sort(unassigned_gates.begin(), unassigned_gates.end());

                    std::shared_ptr<Grouping> output = std::make_shared<Grouping>(netlist_abstr);

                    u32 id_counter = -1;


                    // copy known groups to final result and erase from unassignes gates
                    for (const auto& [group_id, gates] : initial_grouping->gates_of_group)
                    {
                        if (!initial_grouping->operations_on_group_allowed.at(group_id))
                        {
                            u32 new_group_id = ++id_counter;

                            output->group_control_fingerprint_map[new_group_id] = initial_grouping->netlist_abstr.gate_to_fingerprint.at(*gates.begin());
                            output->operations_on_group_allowed[new_group_id]   = false;

                            output->gates_of_group[new_group_id].insert(gates.begin(), gates.end());
                            for (const auto& sg : gates)
                            {
                                output->parent_group_of_gate[sg] = new_group_id;   
                            }

                            std::set<u32> sorted_gates(gates.begin(), gates.end());
                            std::remove(all_results.begin(), all_results.end(), sorted_gates);
                            groups_embedding.erase(std::remove_if(groups_embedding.begin(),groups_embedding.end(),
                                                            [&gates,&s_gates_maping](auto check_group) {
                                                                return std::any_of(gates.begin(), gates.end(), [&check_group, &s_gates_maping](auto check_group_gate) {
                                                                    return check_group[s_gates_maping[check_group_gate]]==1;
                                                                });
                                                            }),
                                             groups_embedding.end());
                            unassigned_gates.erase(std::remove_if(unassigned_gates.begin(), unassigned_gates.end(), [&sorted_gates](auto id) { return sorted_gates.find(id) != sorted_gates.end(); }),
                                               unassigned_gates.end());     
                        }
                    }
                    
                    const float percent_scan = 0.1f;

                    log_info("dataflow", "got {} voting groups", groups_embedding.size());
			
                    // scan groups until all or done
                    float original_size = all_results.size();
                    progress_printer progress_bar;
                    bool Kmeans_needed = true;
                    // ifstream ifile;
                    // ifile.open("b.txt");
                    // if(ifile) {
                    //    cout<<"file exists";
                    // }
                    //std::filesystem::remove("../plugins/dataflow_analysis/src/evaluation/the_saved_model.pth");
                    
                    while (!all_results.empty())
                    {
                        progress_bar.print_progress((original_size - all_results.size()) / original_size);

                        std::cout << "voting groups num " << groups_embedding.size() << "\n"; 

                        // precompute the group indices of each gate
                        std::unordered_map<u32, u32> max_group_size_of_gate;
                        std::unordered_map<u32, std::vector<u32>> groups_of_gate;

                        for (auto g : unassigned_gates)
                        {
                            max_group_size_of_gate.emplace(g, 0);
                            groups_of_gate.emplace(g, std::vector<u32>{});
                        }

                        for (u32 i = 0; i < all_results.size(); ++i)
                        {
                            auto size = (u32)all_results[i].size();
                            for (auto g : all_results[i])
                            {
                                auto it    = max_group_size_of_gate.find(g);
                                it->second = std::max(it->second, size);
                                groups_of_gate.at(g).push_back(i);
                            }
                        }
                        std::vector<std::set<u32>> chosen_groups;
                        if(Kmeans_needed)
                        {
                            //do Kmeans and get the candidates to choose from
                            std::ofstream embedding_file;
                            embedding_file.open("group_embedding.txt", std::ofstream::trunc);	
                            for (const auto &element : groups_embedding)
                            {
                                for (const auto &num : element)
                                {
                                    embedding_file << num << " ";
                                }
                                embedding_file << "\n";
                            } 
                            embedding_file.close();
                            std::string cmd_line="python3 /home/osboxes/hal/plugins/dataflow_analysis/src/evaluation/Kmeans.py --groups_embedding=./group_embedding.txt --gates_num="+std::to_string(gates_num);
                            cmd_line+=" --sizes=\" ";
                            for(int i=0;i<config.prioritized_sizes.size();i++)
                            {
                                cmd_line+=std::to_string(config.prioritized_sizes[i]);
                                cmd_line+=" ";
                            }
                            cmd_line+="\"";
                            std::cout<<cmd_line<<std::endl;
                            system(cmd_line.data());
                            std::fstream chosen_gs;
                            chosen_gs.open("chosen_group.txt", std::ios::in);
                            std::string line;
                            int pos=0;
                            while (std::getline(chosen_gs, line))
                            {
                                chosen_groups.push_back(std::set<u32>{});
                                std::stringstream ss;
                                ss << line;
                                float number; 
                                while (!ss.eof())
                                {
                                    ss >> number;
                                    chosen_groups[pos].insert(s_gates[number]);
                                }
                                pos++;	
                            }
                            chosen_gs.close();

                            if(chosen_groups.empty())
                            {
                                Kmeans_needed=false;
                                continue;
                            }
                        }
                        else
                        {
                            for (const auto &g : all_results)
                            {
                                    chosen_groups.push_back(g);
                            }
                        }

                    	 // counts sequential gates that would end up in bad groups for each scanned candidate
                        u32 num_scanned_groups = chosen_groups.size();
                        std::vector<std::vector<u32>> badness_of_group(num_scanned_groups);

                        utils::parallel_for_each(0, num_scanned_groups, [&](u32 scanned_group_idx) {
                            auto& scanned_group = chosen_groups[scanned_group_idx];
                            // get all unassigned gates that are not in this group
                            std::vector<u32> unaffected_gates;
                            unaffected_gates.reserve(unassigned_gates.size() - scanned_group.size());
                            std::set_difference(unassigned_gates.begin(), unassigned_gates.end(), scanned_group.begin(), scanned_group.end(), std::back_inserter(unaffected_gates));

                            // ##############################################

                            std::unordered_set<u32> affected_groups;
                            for (auto g : scanned_group)
                            {
                                auto& gg = groups_of_gate.at(g);
                                affected_groups.insert(gg.begin(), gg.end());
                            }
                            affected_groups.erase(scanned_group_idx);
                            for (auto gr : affected_groups)
                            {
                                if (std::find(config.prioritized_sizes.begin(), config.prioritized_sizes.end(), all_results[gr].size()) != config.prioritized_sizes.end())
                                {
                                    badness_of_group[scanned_group_idx].push_back(1);
                                }
                            }
                            // get the maximum size of the groups of each unaffected gate
                            for (auto g : unaffected_gates)
                            {
                                if (max_group_size_of_gate.at(g) <= bad_group_size)
                                {
                                    badness_of_group[scanned_group_idx].push_back(1);
                                }
                            }

                            // ##############################################
                        });

                        // log_info("dataflow", "scanned {}/{} groups in this iteration", num_scanned_groups, sorted_results.size());

                        // get the scanned group that would result in the least amount of bad groups
                        std::vector<u32> group_badness_score;
                        for(auto g : badness_of_group)
                        {
                        	group_badness_score.push_back(g.size());
                        }
                        u32 best_choice = std::distance(group_badness_score.begin(), std::min_element(group_badness_score.begin(), group_badness_score.end()));
                        auto best_group = chosen_groups[best_choice];
                        std::cout << "best_group size " << best_group.size() << "\n";

                        // add this group to the final output
                        {
                            u32 new_group_id = ++id_counter;

                            output->group_control_fingerprint_map[new_group_id] = netlist_abstr.gate_to_fingerprint.at(*best_group.begin());
                            output->operations_on_group_allowed[new_group_id]   = true;

                            output->gates_of_group[new_group_id].insert(best_group.begin(), best_group.end());
                            for (const auto& sg : best_group)
                            {
                                output->parent_group_of_gate[sg] = new_group_id;
                            }
                        }

                        unassigned_gates.erase(std::remove_if(unassigned_gates.begin(), unassigned_gates.end(), [&best_group](auto id) { return best_group.find(id) != best_group.end(); }),
                                               unassigned_gates.end());
                        // remove all candidate groupings that contain any of the newly assigned gates
                        groups_embedding.erase(std::remove_if(groups_embedding.begin(), groups_embedding.end(),
                                                            [&best_group, &s_gates_maping](auto check_group) {
                                                                return std::any_of(best_group.begin(), best_group.end(), [&check_group, &s_gates_maping](auto check_group_gate) {
                                                                    return check_group[s_gates_maping[check_group_gate]]==1;
                                                                });
                                                            }),
                                             groups_embedding.end());
                        
                        all_results.erase(std::remove_if(all_results.begin(),
                                                            all_results.end(),
                                                            [&best_group](auto check_group) {
                                                                return std::any_of(check_group.begin(), check_group.end(), [&best_group](auto check_group_gate) {
                                                                    return best_group.find(check_group_gate) != best_group.end();
                                                                });
                                                            }),
                                             all_results.end());
                    }
                    progress_bar.clear();


                    for (auto g : unassigned_gates)
                    {
                        u32 new_group_id = ++id_counter;

                        output->group_control_fingerprint_map[new_group_id] = netlist_abstr.gate_to_fingerprint.at(g);
                        output->operations_on_group_allowed[new_group_id]   = true;

                        output->gates_of_group[new_group_id].insert(g);
                        output->parent_group_of_gate[g] = new_group_id;
                    }

                    return output;
                }

            }    // namespace

            evaluation::Result run(const Configuration& config, Context& ctx, const std::shared_ptr<Grouping>& initial_grouping, const processing::Result& result)
            {
                evaluation::Result output;
                output.is_final_result = false;

                output.merged_result = generate_output(config, initial_grouping, result);

                if (std::any_of(ctx.partial_results.begin(), ctx.partial_results.end(), [&](auto& seen) { return *seen == *output.merged_result; }))
                {
                    if (*ctx.partial_results.back() == *output.merged_result)
                    {
                        log_info("dataflow", "result does not improve anymore");
                    }
                    else
                    {
                        log_info("dataflow", "found a cycle");
                    }

                    output.is_final_result = true;
                    log_info("dataflow", "voting done");
                }

                ctx.partial_results.push_back(output.merged_result);

                return output;
            }
        }    // namespace evaluation
    }        // namespace dataflow
}    // namespace hal