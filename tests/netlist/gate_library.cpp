#include "netlist_test_utils.h"
#include "gtest/gtest.h"
#include "hal_core/utilities/log.h"
#include <iostream>
#include "hal_core/netlist/gate_library/gate_library.h"
#include "hal_core/netlist/gate_library/gate_type/gate_type.h"
#include "hal_core/netlist/gate_library/gate_type/gate_type_lut.h"
#include "hal_core/netlist/gate_library/gate_type/gate_type_sequential.h"

namespace hal {
    /*
     * Testing the GateLibrary class, as well as the different gate_types (GateType, GateTypeLut, GateTypeSequential)
     */

    class GateLibraryTest : public ::testing::Test {
    protected:

        virtual void SetUp() {
            test_utils::init_log_channels();
        }

        virtual void TearDown() {
        }
    };

    // ========= GateType tests ===========

    /**
     * Testing the constructor and the access to the gates name and BaseType (combinatorial for GateType class)
     *
     * Functions: <constructor>, get_name, get_base_type, to_string, operator<<
     */
    TEST_F(GateLibraryTest, check_constructor) {
        TEST_START
            GateType gt("gt_name");
            EXPECT_EQ(gt.get_name(), "gt_name");
            EXPECT_EQ(gt.to_string(), "gt_name");
            std::stringstream ss;
            ss << gt;
            EXPECT_EQ(ss.str(), "gt_name");
            EXPECT_EQ(gt.get_base_type(), GateType::BaseType::combinational);
        TEST_END
    }

    /**
     * Testing the addition of input and output pins and the access to them
     *
     * Functions: add_input_pin, add_input_pins, get_input_pins, add_output_pin, add_output_pins, get_output_pins
     */
    TEST_F(GateLibraryTest, check_pin_management) {
        TEST_START
            {
                // Add some input pins
                GateType gt("gt_name");
                gt.add_input_pin("IN_0"); // Single
                gt.add_input_pins(std::vector<std::string>({"IN_1", "IN_2"})); // Multiple
                EXPECT_EQ(gt.get_input_pins(), std::vector<std::string>({"IN_0", "IN_1", "IN_2"}));
            }
            {
                // Add some output pins
                GateType gt("gt_name");
                gt.add_output_pin("OUT_0"); // Single
                gt.add_output_pins(std::vector<std::string>({"OUT_1", "OUT_2"})); // Multiple
                EXPECT_EQ(gt.get_output_pins(), std::vector<std::string>({"OUT_0", "OUT_1", "OUT_2"}));
            }
        TEST_END
    }

    /**
     * Testing the usage of power and ground pins
     *
     * Functions: assign_power_pin, get_power_pins, assign_ground_pin, get_ground_pins
     */
    TEST_F(GateLibraryTest, check_power_ground_pins) {
        TEST_START
            {
                // assign and get some pins
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.add_input_pins(std::vector<std::string>({"I_0", "I_1"}));
                gts.assign_power_pin("I_0");
                gts.assign_ground_pin("I_1");
                EXPECT_EQ(gts.get_power_pins(), std::unordered_set<std::string>({"I_0"}));
                EXPECT_EQ(gts.get_ground_pins(), std::unordered_set<std::string>({"I_1"}));
            }
            // Negative
            {
                // Try to assign pins that were not registered as input pins
                NO_COUT_TEST_BLOCK;
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.assign_power_pin("I_0");
                gts.assign_ground_pin("I_1");
                EXPECT_EQ(gts.get_power_pins(), std::unordered_set<std::string>());
                EXPECT_EQ(gts.get_ground_pins(), std::unordered_set<std::string>());
            }
        TEST_END
    }

    /**
     * Testing the usage of Gate types with pin groups
     *
     * Functions: add_input_pin_group, get_input_pin_groups, add_output_pin_groups, get_output_pin_groups
     */
    TEST_F(GateLibraryTest, check_pin_groups) {
        TEST_START
            {
                // Add input pin groups
                GateType gt("gt_name");
                std::map<u32, std::string> pin_group_a =
                    {{0, "pin_group_a(0)"}, {1, "pin_group_a(1)"}, {2, "pin_group_a(2)"}, {3, "pin_group_a(3)"}};
                std::map<u32, std::string> pin_group_b = {{0, "pin_group_b(0)"}, {1, "pin_group_b(1)"}};
                std::unordered_map<std::string, std::map<u32, std::string>>
                    pin_groups = {{"pin_group_a", pin_group_a}, {"pin_group_b", pin_group_b}};

                gt.add_input_pins({"pin_group_a(0)", "pin_group_a(1)", "pin_group_a(2)", "pin_group_a(3)",
                                   "pin_group_b(0)", "pin_group_b(1)"});
                gt.assign_input_pin_group("pin_group_a", pin_group_a);
                gt.assign_input_pin_group("pin_group_b", pin_group_b);
                EXPECT_EQ(gt.get_input_pin_groups(), pin_groups);
                EXPECT_EQ(gt.get_input_pins(),
                          std::vector<std::string>({"pin_group_a(0)", "pin_group_a(1)", "pin_group_a(2)",
                                                    "pin_group_a(3)", "pin_group_b(0)", "pin_group_b(1)"}));
            }
            {
                // Add output pin groups
                GateType gt("gt_name");
                std::map<u32, std::string> pin_group_a =
                    {{0, "pin_group_a(0)"}, {1, "pin_group_a(1)"}, {2, "pin_group_a(2)"}, {3, "pin_group_a(3)"}};
                std::map<u32, std::string> pin_group_b = {{0, "pin_group_b(0)"}, {1, "pin_group_b(1)"}};
                std::unordered_map<std::string, std::map<u32, std::string>>
                    pin_groups = {{"pin_group_a", pin_group_a}, {"pin_group_b", pin_group_b}};

                gt.add_output_pins({"pin_group_a(0)", "pin_group_a(1)", "pin_group_a(2)", "pin_group_a(3)",
                                    "pin_group_b(0)", "pin_group_b(1)"});
                gt.assign_output_pin_group("pin_group_a", pin_group_a);
                gt.assign_output_pin_group("pin_group_b", pin_group_b);
                EXPECT_EQ(gt.get_output_pin_groups(), pin_groups);
                EXPECT_EQ(gt.get_output_pins(),
                          std::vector<std::string>({"pin_group_a(0)", "pin_group_a(1)", "pin_group_a(2)",
                                                    "pin_group_a(3)", "pin_group_b(0)", "pin_group_b(1)"}));
            }
            // NEGATIVE TESTS
            {
                // Try to add an already added pin group
                GateType gt("gt_name");

                // Output Pin Group
                std::map<u32, std::string> out_pin_group = {{0, "out_pin_group(0)"}, {1, "out_pin_group(1)"}};
                std::unordered_map<std::string, std::map<u32, std::string>> out_pin_groups = {{"out_pin_group", out_pin_group}};

                gt.add_output_pins({"out_pin_group(0)", "out_pin_group(1)"});
                gt.assign_output_pin_group("out_pin_group", out_pin_group);
                EXPECT_EQ(gt.get_output_pin_groups(), out_pin_groups);
                gt.assign_output_pin_group("out_pin_group", out_pin_group);
                EXPECT_EQ(gt.get_output_pin_groups(), out_pin_groups);
                EXPECT_EQ(gt.get_output_pins(), std::vector<std::string>({"out_pin_group(0)", "out_pin_group(1)"}));

                // Input Pin Group
                std::map<u32, std::string> in_pin_group = {{0, "in_pin_group(0)"}, {1, "in_pin_group(1)"}};
                std::unordered_map<std::string, std::map<u32, std::string>> in_pin_groups = {{"in_pin_group", in_pin_group}};

                gt.add_input_pins({"in_pin_group(0)", "in_pin_group(1)"});
                gt.assign_input_pin_group("in_pin_group", in_pin_group);
                EXPECT_EQ(gt.get_input_pin_groups(), in_pin_groups);
                gt.assign_input_pin_group("in_pin_group", in_pin_group);
                EXPECT_EQ(gt.get_input_pin_groups(), in_pin_groups);
                EXPECT_EQ(gt.get_input_pins(), std::vector<std::string>({"in_pin_group(0)", "in_pin_group(1)"}));
            }
            if(test_utils::known_issue_tests_active())
            {
                // ISSUE: Is possible though the pins are not added...
                // Add a pin group that contains previously unregistered pins
                GateType gt("gt_name");
                std::map<u32, std::string> pin_group = {{0, "pin_group(0)"}, {1, "pin_group(1)"}};
                std::unordered_map<std::string, std::map<u32, std::string>> empty_pin_groups;
                gt.assign_output_pin_group("out_pin", pin_group);
                gt.assign_input_pin_group("in_pin", pin_group);
                EXPECT_EQ(gt.get_output_pins(), std::vector<std::string>());
                EXPECT_EQ(gt.get_input_pins(), std::vector<std::string>());
                EXPECT_EQ(gt.get_output_pin_groups(), empty_pin_groups);
                EXPECT_EQ(gt.get_input_pin_groups(), empty_pin_groups);
            }
        TEST_END
    }

    /**
     * Testing the assignment of boolean functions to the pins. Also, boolean functions for non existing pin names
     * can be created, since they are used to store group attributes like next_state, clocked_on, etc.
     *
     * Functions: add_boolean_function, get_boolean_functions
     */
    TEST_F(GateLibraryTest, check_boolean_function_assignment) {
        TEST_START
            {
                // Add a boolean function for an output pin
                GateType gt("gt_name");

                gt.add_input_pins(std::vector<std::string>({"IN_0", "IN_1"}));
                gt.add_output_pin("OUT");

                BooleanFunction bf_out = BooleanFunction::from_string("IN_0 ^ IN_1");
                gt.add_boolean_function("OUT", bf_out);
                std::unordered_map<std::string, BooleanFunction> gt_bf_map = gt.get_boolean_functions();
                EXPECT_EQ(gt_bf_map.size(), 1);
                ASSERT_FALSE(gt_bf_map.find("OUT") == gt_bf_map.end());
                EXPECT_EQ(gt_bf_map.at("OUT"), bf_out);
            }
        TEST_END
    }

    /**
     * Testing the equal_to operator
     *
     * Functions: operator==, operator!=
     */
    TEST_F(GateLibraryTest, check_equal_to_operator) {
        TEST_START
            {
                // Testing the comparison of Gate types (compared by id)
                GateType gt_0("gt_name");
                GateType gt_1("gt_name");

                EXPECT_TRUE(gt_0 == gt_0);
                EXPECT_FALSE(gt_0 == gt_1);
                EXPECT_FALSE(gt_0 != gt_0);
                EXPECT_TRUE(gt_0 != gt_1);
            }
        TEST_END
    }

    // ======== GateTypeLut tests ========

    /**
     * Testing the addition of output_from_init_string pins
     *
     * Functions: add_output_from_init_string_pin, get_output_from_init_string_pin
     */
    TEST_F(GateLibraryTest, check_output_from_init_string_pins) {
        TEST_START
            {
                // Add and get some output_from_init_string pins
                GateTypeLut gtl("gtl_name");
                gtl.add_output_pins(std::vector<std::string>({"O0", "OFIS_0", "OFIS_1"}));
                gtl.assign_lut_pin("OFIS_0");
                gtl.assign_lut_pin("OFIS_1");
                EXPECT_EQ(gtl.get_lut_pins(), std::unordered_set<std::string>({"OFIS_0", "OFIS_1"}));
            }
            // Negative
            if(test_utils::known_issue_tests_active())
            {
                // Try to add output from init string pins, that were not registered as outputs
                // ISSUE: pin is added anyway, but documentation says it shouldn't.
                //  It is only checked that the passed pin is no input pin
                GateTypeLut gtl("gtl_name");
                gtl.assign_lut_pin("OFIS_0");
                EXPECT_EQ(gtl.get_lut_pins(), std::unordered_set<std::string>());
            }
        TEST_END
    }

    /**
     * Testing the access on config data fields (config_data_category, config_data_identifier, config_data_ascending_order)
     * as well as the acess on the base type (should be lut)
     *
     * Functions: get_base_type, get_config_data_category, set_config_data_category, get_config_data_identifier,
     *            set_config_data_identifier, set_config_data_ascending_order, is_config_data_ascending_order
     */
    TEST_F(GateLibraryTest, check_config_data) {
        TEST_START
            GateTypeLut gtl("gtl_name");
            {
                // Base Type
                EXPECT_EQ(gtl.get_base_type(), GateType::BaseType::lut);
            }
            {
                // Config Data Category
                gtl.set_config_data_category("category");
                EXPECT_EQ(gtl.get_config_data_category(), "category");
            }
            {
                // Identifiers
                gtl.set_config_data_identifier("identifier");
                EXPECT_EQ(gtl.get_config_data_identifier(), "identifier");
            }
            {
                // Ascending Order
                EXPECT_TRUE(gtl.is_config_data_ascending_order()); // Set true by constructor
                gtl.set_config_data_ascending_order(false);
                EXPECT_FALSE(gtl.is_config_data_ascending_order());
                gtl.set_config_data_ascending_order(true);
                EXPECT_TRUE(gtl.is_config_data_ascending_order());
            }
        TEST_END
    }

    // ===== GateTypeSequential tests ====

    /**
     * Testing the usage of state_pin and negated_state_pin
     *
     * Functions: add_state_pins, get_state_pins,
     *           add_negated_state_pin, get_negated_state_pin
     */
    TEST_F(GateLibraryTest, check_state_output_pins) {
        TEST_START
            {
                // Add and get some state pins
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.add_output_pins(std::vector<std::string>({"SO_0", "SO_1"}));
                gts.assign_state_pin("SO_0");
                gts.assign_state_pin("SO_1");
                EXPECT_EQ(gts.get_state_pins(), std::unordered_set<std::string>({"SO_0", "SO_1"}));
            }
            {
                // Add and get some negated_state_pin pins
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.add_output_pins(std::vector<std::string>({"ISO_0", "ISO_1"}));
                gts.assign_negated_state_pin("ISO_0");
                gts.assign_negated_state_pin("ISO_1");
                EXPECT_EQ(gts.get_negated_state_pins(), std::unordered_set<std::string>({"ISO_0", "ISO_1"}));
                EXPECT_EQ(gts.get_output_pins(), std::vector<std::string>({"ISO_0", "ISO_1"}));
            }
            // Negative
            {
                // Try to add a state_pin that was not registered as an output pin
                NO_COUT_TEST_BLOCK;
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.assign_state_pin("SO_0");
                EXPECT_EQ(gts.get_state_pins(), std::unordered_set<std::string>());
            }
            {
                // Try to add an negated_state_pin that was not registered as an output pin
                NO_COUT_TEST_BLOCK;
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.assign_negated_state_pin("ISO_0");
                EXPECT_EQ(gts.get_negated_state_pins(), std::unordered_set<std::string>());
            }
        TEST_END
    }

    /**
     * Testing the usage of special pins
     *
     * Functions: assign_clock_pin, get_clock_pins, assign_enable_pin, get_enable_pins, assign_reset_pin, get_reset_pins, assign_set_pin, get_set_pins, assign_data_pin, get_data_pins
     */
    TEST_F(GateLibraryTest, check_special_pins) {
        TEST_START
            {
                // assign and get some pins
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.add_input_pins(std::vector<std::string>({"I_0", "I_1", "I_2", "I_3", "I_4"}));
                gts.assign_clock_pin("I_0");
                gts.assign_enable_pin("I_1");
                gts.assign_reset_pin("I_2");
                gts.assign_set_pin("I_3");
                gts.assign_data_pin("I_4");
                EXPECT_EQ(gts.get_clock_pins(), std::unordered_set<std::string>({"I_0"}));
                EXPECT_EQ(gts.get_enable_pins(), std::unordered_set<std::string>({"I_1"}));
                EXPECT_EQ(gts.get_reset_pins(), std::unordered_set<std::string>({"I_2"}));
                EXPECT_EQ(gts.get_set_pins(), std::unordered_set<std::string>({"I_3"}));
                EXPECT_EQ(gts.get_data_pins(), std::unordered_set<std::string>({"I_4"}));
            }
            // Negative
            {
                // Try to assign pins that were not registered as input pins
                NO_COUT_TEST_BLOCK;
                GateTypeSequential gts("gts_name", GateType::BaseType::ff);
                gts.assign_clock_pin("I_0");
                gts.assign_enable_pin("I_1");
                gts.assign_reset_pin("I_2");
                gts.assign_set_pin("I_3");
                gts.assign_data_pin("I_4");
                EXPECT_EQ(gts.get_clock_pins(), std::unordered_set<std::string>());
                EXPECT_EQ(gts.get_enable_pins(), std::unordered_set<std::string>());
                EXPECT_EQ(gts.get_reset_pins(), std::unordered_set<std::string>());
                EXPECT_EQ(gts.get_set_pins(), std::unordered_set<std::string>());
                EXPECT_EQ(gts.get_data_pins(), std::unordered_set<std::string>());
            }
        TEST_END
    }

    /**
     * Testing the access on init data fields (init_data_category, init_data_category)
     * as well as the access on the base type and the clear-preset behavior
     *
     * Functions: get_base_type, get_config_data_category, set_config_data_category, get_config_data_identifier,
     *            set_config_data_identifier, set_config_data_ascending_order, is_config_data_ascending_order
     */
    TEST_F(GateLibraryTest, check_init_data) {
        TEST_START
            GateTypeSequential gts("gtl_name", GateType::BaseType::ff);
            {
                // Base Type
                EXPECT_EQ(gts.get_base_type(), GateType::BaseType::ff);
            }
            {
                // Init Data Category
                gts.set_init_data_category("category");
                EXPECT_EQ(gts.get_init_data_category(), "category");
            }
            {
                // Identifiers
                gts.set_init_data_identifier("identifier");
                EXPECT_EQ(gts.get_init_data_identifier(), "identifier");
            }
            {
                // Clear-Preset Behavior
                gts.set_clear_preset_behavior(GateTypeSequential::ClearPresetBehavior::L,
                                           GateTypeSequential::ClearPresetBehavior::H);
                EXPECT_EQ(gts.get_clear_preset_behavior(),
                          std::make_pair(GateTypeSequential::ClearPresetBehavior::L,
                                         GateTypeSequential::ClearPresetBehavior::H));
            }
            {
                // Get uninitialized clear_preset behavior
                GateTypeSequential gts_2("gtl_name", GateType::BaseType::ff);
                EXPECT_EQ(gts_2.get_clear_preset_behavior(),
                          std::make_pair(GateTypeSequential::ClearPresetBehavior::U,
                                         GateTypeSequential::ClearPresetBehavior::U));
            }
        TEST_END
    }

    // ======== GateLibrary tests =========

    /**
     * Testing the creation of a new GateLibrary and the addition of Gate types and includes to it
     *
     * Functions: constructor, create_gate_type, get_name, get_gate_types, get_vcc_gate_types, get_gnd_gate_types,
     *            add_include, get_includes
     */
    TEST_F(GateLibraryTest, check_library) {
        TEST_START
            {
                auto gl = std::make_unique<GateLibrary>("imaginary_path", "gl_name");

                // Create some Gate types beforehand
                // AND gate type
                auto gt_and = gl->create_gate_type("gt_and");
                ASSERT_TRUE(gt_and != nullptr);
                gt_and->add_input_pins(std::vector<std::string>({"I0", "I1"}));
                gt_and->add_output_pins(std::vector<std::string>({"O"}));
                gt_and->add_boolean_function("O", BooleanFunction::from_string("I0 & I1"));

                // OR gate type
                auto gt_or = gl->create_gate_type("gt_or", GateType::BaseType::combinational);
                ASSERT_TRUE(gt_or != nullptr);
                gt_or->add_input_pins(std::vector<std::string>({"I0", "I1"}));
                gt_or->add_output_pins(std::vector<std::string>({"O"}));
                gt_or->add_boolean_function("O", BooleanFunction::from_string("I0 | I1"));

                // GND gate type
                auto gt_gnd = gl->create_gate_type("gt_gnd");
                ASSERT_TRUE(gt_gnd != nullptr);
                gt_gnd->add_output_pins(std::vector<std::string>({"O"}));
                gt_gnd->add_boolean_function("O", BooleanFunction(BooleanFunction::ZERO));
                gl->mark_gnd_gate_type(gt_gnd);

                // VCC gate type
                auto gt_vcc = gl->create_gate_type("gt_vcc");
                ASSERT_TRUE(gt_vcc != nullptr);
                gt_vcc->add_output_pins(std::vector<std::string>({"O"}));
                gt_vcc->add_boolean_function("O", BooleanFunction(BooleanFunction::ONE));
                gl->mark_vcc_gate_type(gt_vcc);

                // FF gate type
                auto gt_ff = gl->create_gate_type("gt_ff", GateType::BaseType::ff);
                ASSERT_TRUE(gt_ff != nullptr);
                auto gt_latch = gl->create_gate_type("gt_latch", 
                
                // Latch gate type
                GateType::BaseType::latch);
                ASSERT_TRUE(gt_latch != nullptr);

                // LUT gate type
                auto gt_lut = gl->create_gate_type("gt_lut", GateType::BaseType::lut);
                ASSERT_TRUE(gt_lut != nullptr);

                // Check the name
                EXPECT_EQ(gl->get_name(), "gl_name");

                // check if all gate types contained in library
                EXPECT_EQ(gl->get_gate_types(),
                          (std::unordered_map<std::string, GateType*>({{"gt_and", gt_and},
                          {"gt_gnd", gt_gnd},
                          {"gt_vcc", gt_vcc},
                          {"gt_or", gt_or},
                          {"gt_ff", gt_ff},
                          {"gt_latch", gt_latch},
                          {"gt_lut", gt_lut}})));
                EXPECT_EQ(gl->get_vcc_gate_types(),
                          (std::unordered_map<std::string, GateType*>({{"gt_vcc", gt_vcc}})));
                EXPECT_EQ(gl->get_gnd_gate_types(),
                          (std::unordered_map<std::string, GateType*>({{"gt_gnd", gt_gnd}})));

                // check base types
                EXPECT_EQ(gt_and->get_base_type(), GateType::BaseType::combinational);
                EXPECT_EQ(gt_or->get_base_type(), GateType::BaseType::combinational);
                EXPECT_EQ(gt_ff->get_base_type(), GateType::BaseType::ff);
                EXPECT_EQ(gt_latch->get_base_type(), GateType::BaseType::latch);
                EXPECT_EQ(gt_lut->get_base_type(), GateType::BaseType::lut);

                // check contains_gate_type and contains_gate_type_by_name
                EXPECT_TRUE(gl->contains_gate_type(gt_and));
                EXPECT_FALSE(gl->contains_gate_type(nullptr));
                GateType gt_nil("not_in_library");
                EXPECT_FALSE(gl->contains_gate_type(&gt_nil));

                EXPECT_TRUE(gl->contains_gate_type_by_name(gt_and->get_name()));
                EXPECT_FALSE(gl->contains_gate_type_by_name(""));
                EXPECT_FALSE(gl->contains_gate_type_by_name("not_in_library"));

                // check get_gate_type_by_name
                EXPECT_EQ(gl->get_gate_type_by_name("gt_and"), gt_and);
                EXPECT_EQ(gl->get_gate_type_by_name(""), nullptr);
                EXPECT_EQ(gl->get_gate_type_by_name("not_in_library"), nullptr);

                // Check the addition of includes
                gl->add_include("in.clu.de");
                gl->add_include("another.include");
                gl->add_include("last.include");
                EXPECT_EQ(gl->get_includes(),
                          std::vector<std::string>({"in.clu.de", "another.include", "last.include"}));
            }
        TEST_END
    }
} //namespace hal
