#include "hal_core/netlist/netlist_utils.h"

#include "netlist_test_utils.h"

namespace hal
{
    using test_utils::MIN_GATE_ID;
    using test_utils::MIN_MODULE_ID;
    using test_utils::MIN_NET_ID;

    class NetlistUtilsTest : public ::testing::Test
    {
    protected:
        virtual void SetUp()
        {
            test_utils::init_log_channels();
        }

        virtual void TearDown()
        {
        }
    };

    namespace 
    {
        Net* connect(Netlist* nl, Gate* src, std::string src_pin, Gate* dst, std::string dst_pin) 
        {
            Net* n;
            if (n = src->get_fan_out_net(src_pin); n != nullptr)
            {
                n->add_destination(dst, dst_pin);
            }
            else if (n = dst->get_fan_in_net(dst_pin); n != nullptr)
            {
                n->add_source(src, src_pin);
            }
            else
            {
                n = nl->create_net("net_" + std::to_string(src->get_id()) + "_" + std::to_string(dst->get_id()));
                n->add_source(src, src_pin);
                n->add_destination(dst, dst_pin);
            }
            return n;
        }

        std::unique_ptr<GateLibrary> create_buffer_lib() 
        {
            std::unique_ptr<GateLibrary> lib = std::unique_ptr<GateLibrary>(new GateLibrary("dummy_path", "dummy_name"));
            GateType* gnd = lib->create_gate_type("GND", {GateTypeProperty::combinational, GateTypeProperty::ground});
            gnd->add_pin("O", PinDirection::output);
            gnd->add_boolean_function("O", BooleanFunction::from_string("0"));
            lib->mark_gnd_gate_type(gnd);

            GateType* vcc = lib->create_gate_type("VCC", {GateTypeProperty::combinational, GateTypeProperty::power});
            vcc->add_pin("O", PinDirection::output);
            vcc->add_boolean_function("O", BooleanFunction::from_string("1"));
            lib->mark_vcc_gate_type(vcc);

            GateType* and2 = lib->create_gate_type("AND2", {GateTypeProperty::combinational});
            and2->add_pins({"I0", "I1"}, PinDirection::input);
            and2->add_pin("O", PinDirection::output);
            and2->add_boolean_function("O", BooleanFunction::from_string("I0 & I1"));

            GateType* lut2 = lib->create_gate_type("LUT2", {GateTypeProperty::combinational, GateTypeProperty::lut});
            lut2->add_pins({"I0", "I1"}, PinDirection::input);
            lut2->add_pin("O", PinDirection::output, PinType::lut);
            lut2->set_config_data_category("generic");
            lut2->set_config_data_identifier("INIT");

            GateType* lut4 = lib->create_gate_type("LUT4", {GateTypeProperty::combinational, GateTypeProperty::lut});
            lut4->add_pins({"I0", "I1", "I2", "I3"}, PinDirection::input);
            lut4->add_pin("O", PinDirection::output, PinType::lut);
            lut4->set_config_data_category("generic");
            lut4->set_config_data_identifier("INIT");

            GateType* buf = lib->create_gate_type("BUF", {GateTypeProperty::combinational});
            buf->add_pin("I", PinDirection::input);
            buf->add_pin("O", PinDirection::output);
            buf->add_boolean_function("O", BooleanFunction::from_string("I"));

            return std::move(lib);
        }
    }

    /**
     * Testing the get_subgraph_function
     *
     * Functions: get_subgraph_function
     */
    TEST_F(NetlistUtilsTest, check_get_subgraph_function)
    {
        TEST_START
        // Use the example netlist, that is filled with boolean functions for this test
        // (3) becomes a NOT gate, (0) an AND gate, (7) a NOT gate
        std::unique_ptr<Netlist> test_nl = test_utils::create_example_netlist();

        // -- Set the boolean functions for these gates
        Gate* gate_3 = test_nl->get_gate_by_id(MIN_GATE_ID + 3);
        Gate* gate_0 = test_nl->get_gate_by_id(MIN_GATE_ID + 0);
        Gate* gate_7 = test_nl->get_gate_by_id(MIN_GATE_ID + 7);
        gate_3->add_boolean_function("O", BooleanFunction::from_string("!I", std::vector<std::string>({"I"})));
        gate_0->add_boolean_function("O", BooleanFunction::from_string("I0 & I1", std::vector<std::string>({"I0", "I1"})));
        gate_7->add_boolean_function("O", BooleanFunction::from_string("!I", std::vector<std::string>({"I"})));
        // -- Get the names of the connected nets
        std::string net_13_name = std::to_string(MIN_NET_ID + 13);
        std::string net_20_name = std::to_string(MIN_NET_ID + 20);
        std::string net_78_name = std::to_string(MIN_NET_ID + 78);
        std::string net_30_name = std::to_string(MIN_NET_ID + 30);

        {
            // Get the boolean function of a normal sub-graph
            const std::vector<const Gate*> subgraph_gates({gate_0, gate_3});
            const Net* output_net        = test_nl->get_net_by_id(MIN_NET_ID + 045);
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            BooleanFunction expected_bf = BooleanFunction::from_string(("!" + net_13_name + " & " + net_20_name), std::vector<std::string>({net_13_name, net_20_name}));

            EXPECT_EQ(sub_graph_bf, expected_bf);
        }
        // NEGATIVE
        {
            NO_COUT_BLOCK;
            // No subgraph gates are passed
            const std::vector<const Gate*> subgraph_gates({});
            const Net* output_net        = test_nl->get_net_by_id(MIN_NET_ID + 045);
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_TRUE(sub_graph_bf.is_empty());
        }
        {
            NO_COUT_BLOCK;
            // One of the gates is a nullptr
            const std::vector<const Gate*> subgraph_gates({gate_0, nullptr, gate_3});
            const Net* output_net        = test_nl->get_net_by_id(MIN_NET_ID + 045);
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_TRUE(sub_graph_bf.is_empty());
        }
        {
            NO_COUT_BLOCK;
            // The output net is a nullptr
            const std::vector<const Gate*> subgraph_gates({gate_0, gate_3});
            const Net* output_net        = nullptr;
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_TRUE(sub_graph_bf.is_empty());
        }
        {
            NO_COUT_BLOCK;
            // The output net has multiple sources
            // -- create such a net
            Net* multi_src_net = test_nl->create_net("muli_src_net");
            multi_src_net->add_source(test_nl->get_gate_by_id(MIN_GATE_ID + 4), "O");
            multi_src_net->add_source(test_nl->get_gate_by_id(MIN_GATE_ID + 5), "O");

            const std::vector<const Gate*> subgraph_gates({gate_0, gate_3});
            const Net* output_net        = multi_src_net;
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_TRUE(sub_graph_bf.is_empty());
            // -- remove the net
            test_nl->delete_net(multi_src_net);
        }
        {
            NO_COUT_BLOCK;
            // The output net has no source
            // -- create such a net
            Net* no_src_net = test_nl->create_net("muli_src_net");

            const std::vector<const Gate*> subgraph_gates({gate_0, gate_3});
            const Net* output_net        = no_src_net;
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_TRUE(sub_graph_bf.is_empty());
            // -- remove the net
            test_nl->delete_net(no_src_net);
        }
        {
            NO_COUT_BLOCK;
            // A net in between has multiple sources (expansion should stop in this direction)
            // -- add a source to net 30 temporarily
            test_nl->get_net_by_id(MIN_NET_ID + 30)->add_source(test_nl->get_gate_by_id(MIN_GATE_ID + 8), "O");

            const std::vector<const Gate*> subgraph_gates({gate_0, gate_3});
            const Net* output_net        = test_nl->get_net_by_id(MIN_NET_ID + 045);
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            BooleanFunction expected_bf = BooleanFunction::from_string((net_30_name + " & " + net_20_name), std::vector<std::string>({net_30_name, net_20_name}));

            EXPECT_EQ(sub_graph_bf, expected_bf);
        }
        {
            NO_COUT_BLOCK;
            // The netlist contains a cycle
            // -- create such a netlist:
            /*   .-=|gate_0|=----.
                 *    '------------.  |
                 *    .-=|gate_1|=-'  |
                 *    '---------------'
                 */
            std::unique_ptr<Netlist> cyclic_nl = test_utils::create_empty_netlist();
            Gate* cy_gate_0                    = cyclic_nl->create_gate(test_utils::get_gate_type_by_name("gate_1_to_1"), "gate_0");
            Gate* cy_gate_1                    = cyclic_nl->create_gate(test_utils::get_gate_type_by_name("gate_1_to_1"), "gate_1");
            cy_gate_0->add_boolean_function("O", BooleanFunction::from_string("O", {"I"}));
            cy_gate_1->add_boolean_function("O", BooleanFunction::from_string("O", {"I"}));

            Net* cy_net_0 = cyclic_nl->create_net("net_0");
            Net* cy_net_1 = cyclic_nl->create_net("net_1");
            cy_net_0->add_source(cy_gate_0, "O");
            cy_net_0->add_destination(cy_gate_1, "I");
            cy_net_1->add_source(cy_gate_1, "O");
            cy_net_1->add_destination(cy_gate_0, "I");

            const std::vector<const Gate*> subgraph_gates({cy_gate_0, cy_gate_1});
            const Net* output_net        = cy_net_0;
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_TRUE(sub_graph_bf.is_empty());
        }
        {
            NO_COUT_BLOCK;
            // A gate of the subgraph has unconnected input pins
            const std::vector<const Gate*> subgraph_gates({gate_7});
            const Net* output_net        = test_nl->get_net_by_id(MIN_NET_ID + 78);
            BooleanFunction sub_graph_bf = netlist_utils::get_subgraph_function(output_net, subgraph_gates);

            EXPECT_EQ(sub_graph_bf, gate_7->get_boolean_function("O"));
        }
        TEST_END
    }

    /**
     * Testing the deep copying of netlists
     *
     * Functions: copy_netlist
     */
    TEST_F(NetlistUtilsTest, check_copy_netlist)
    {
        TEST_START
        // Create an example netlist that should be copied
        std::unique_ptr<Netlist> test_nl = test_utils::create_example_netlist();

        // -- Add some modules to the example netlist
        Module* mod_0 = test_nl->create_module(test_utils::MIN_MODULE_ID + 0,
                                               "mod_0",
                                               test_nl->get_top_module(),
                                               std::vector<Gate*>{test_nl->get_gate_by_id(MIN_GATE_ID + 0),
                                                                  test_nl->get_gate_by_id(MIN_GATE_ID + 1),
                                                                  test_nl->get_gate_by_id(MIN_GATE_ID + 2),
                                                                  test_nl->get_gate_by_id(MIN_GATE_ID + 3),
                                                                  test_nl->get_gate_by_id(MIN_GATE_ID + 4),
                                                                  test_nl->get_gate_by_id(MIN_GATE_ID + 5)});
        Module* mod_1 = test_nl->create_module(test_utils::MIN_MODULE_ID + 1,
                                               "mod_1",
                                               mod_0,
                                               std::vector<Gate*>{
                                                   test_nl->get_gate_by_id(MIN_GATE_ID + 0),
                                                   test_nl->get_gate_by_id(MIN_GATE_ID + 4),
                                                   test_nl->get_gate_by_id(MIN_GATE_ID + 5),
                                               });

        // -- Add some groupings to the netlist
        Grouping* prime_grouping = test_nl->create_grouping("prime_gates");
        prime_grouping->assign_gate_by_id(MIN_GATE_ID + 2);
        prime_grouping->assign_gate_by_id(MIN_GATE_ID + 3);
        prime_grouping->assign_gate_by_id(MIN_GATE_ID + 5);
        prime_grouping->assign_gate_by_id(MIN_GATE_ID + 7);
        prime_grouping->assign_net_by_id(MIN_NET_ID + 13);
        prime_grouping->assign_module_by_id(MIN_MODULE_ID + 1);    // (I know 1 is not prime :P)
        Grouping* empty_grouping = test_nl->create_grouping("empty_grouping");

        // -- Mark Gates as GND/VCC
        test_nl->mark_gnd_gate(test_nl->get_gate_by_id(MIN_GATE_ID + 1));
        test_nl->mark_vcc_gate(test_nl->get_gate_by_id(MIN_GATE_ID + 2));

        // -- Mark nets as global inputs/outputs
        test_nl->mark_global_input_net(test_nl->get_net_by_id(MIN_NET_ID + 20));
        test_nl->mark_global_output_net(test_nl->get_net_by_id(MIN_NET_ID + 78));

        // -- Add some boolean functions to the gates
        test_nl->get_gate_by_id(MIN_GATE_ID + 0)->add_boolean_function("O", BooleanFunction::from_string("I0 & I1"));
        test_nl->get_gate_by_id(MIN_GATE_ID + 4)->add_boolean_function("O", BooleanFunction::from_string("!I"));

        // Copy and compare the netlist
        std::unique_ptr<Netlist> test_nl_copy = netlist_utils::copy_netlist(test_nl.get());

        EXPECT_TRUE(test_utils::netlists_are_equal(test_nl.get(), test_nl_copy.get()));
        TEST_END
    }

    /**
     * Testing the get_next_sequential_gates variants.
     * Testing only the gate functions for now as they call the net functions.
     * TODO: test both variants
     *
     * Functions: get_next_sequential_gates
     */
    TEST_F(NetlistUtilsTest, check_get_next_sequential_gates)
    {
        TEST_START
        // Create an example netlist that should be copied
        std::unique_ptr<Netlist> nl = test_utils::create_empty_netlist(1);
        GateLibrary* gl             = test_utils::get_testing_gate_library();

        Gate* gate_0     = nl->create_gate(MIN_GATE_ID + 0, gl->get_gate_types().at("gnd"), "gate_0");
        Gate* gate_1     = nl->create_gate(MIN_GATE_ID + 1, gl->get_gate_types().at("vcc"), "gate_1");
        Gate* gate_2     = nl->create_gate(MIN_GATE_ID + 2, gl->get_gate_types().at("gate_1_to_1"), "gate_2");
        Gate* gate_3     = nl->create_gate(MIN_GATE_ID + 3, gl->get_gate_types().at("gate_2_to_1"), "gate_3");
        Gate* gate_4_seq = nl->create_gate(MIN_GATE_ID + 4, gl->get_gate_types().at("gate_2_to_1_sequential"), "gate_4_seq");
        Gate* gate_5_seq = nl->create_gate(MIN_GATE_ID + 5, gl->get_gate_types().at("gate_2_to_1_sequential"), "gate_5_seq");
        Gate* gate_6     = nl->create_gate(MIN_GATE_ID + 6, gl->get_gate_types().at("gate_2_to_1"), "gate_6");

        connect(nl.get(), gate_0, "O", gate_2, "I");
        connect(nl.get(), gate_0, "O", gate_3, "I1");
        connect(nl.get(), gate_1, "O", gate_3, "I0");
        connect(nl.get(), gate_3, "O", gate_4_seq, "I0");
        connect(nl.get(), gate_4_seq, "O", gate_4_seq, "I1");
        connect(nl.get(), gate_4_seq, "O", gate_5_seq, "I0");
        connect(nl.get(), gate_0, "O", gate_5_seq, "I1");
        connect(nl.get(), gate_4_seq, "O", gate_6, "I0");
        connect(nl.get(), gate_5_seq, "O", gate_6, "I1");

        std::map<Gate*, std::vector<Gate*>> test_successors = {
            {gate_0, {gate_4_seq, gate_5_seq}},
            {gate_1, {gate_4_seq}},
            {gate_2, {}},
            {gate_3, {gate_4_seq}},
            {gate_4_seq, {gate_4_seq, gate_5_seq}},
            {gate_5_seq, {}},
            {gate_6, {}},
        };

        std::map<Gate*, std::vector<Gate*>> test_predecessors = {
            {gate_0, {}},
            {gate_1, {}},
            {gate_2, {}},
            {gate_3, {}},
            {gate_4_seq, {gate_4_seq}},
            {gate_5_seq, {gate_4_seq}},
            {gate_6, {gate_5_seq, gate_4_seq}},
        };

        for (auto [start, expected] : test_successors)
        {
            auto successors = netlist_utils::get_next_sequential_gates(start, true);
            std::unordered_map<u32, std::vector<Gate*>> cache;
            auto successors_cached = netlist_utils::get_next_sequential_gates(start, true, cache);
            EXPECT_EQ(successors, successors_cached);
            EXPECT_TRUE(test_utils::vectors_have_same_content(successors, expected));

            // std::cout << "successors of: " << start->get_name() << std::endl;
            // std::cout << "computed: " << utils::join(", ", successors, [](auto x) { return x->get_name(); }) << std::endl;
            // std::cout << "expected: " << utils::join(", ", expected, [](auto x) { return x->get_name(); }) << std::endl;
        }

        for (auto [start, expected] : test_predecessors)
        {
            auto predecessors = netlist_utils::get_next_sequential_gates(start, false);
            std::unordered_map<u32, std::vector<Gate*>> cache;
            auto predecessors_cached = netlist_utils::get_next_sequential_gates(start, false, cache);
            EXPECT_EQ(predecessors, predecessors_cached);
            EXPECT_TRUE(test_utils::vectors_have_same_content(predecessors, expected));

            // std::cout << "predecessors of: " << start->get_name() << std::endl;
            // std::cout << "computed: " << utils::join(", ", predecessors, [](auto x) { return x->get_name(); }) << std::endl;
            // std::cout << "expected: " << utils::join(", ", expected, [](auto x) { return x->get_name(); }) << std::endl;
        }

        TEST_END
    }

    /**
     * Testing getting the nets connected to a set of pins.
     *
     * Functions: get_nets_at_pins
     */
    TEST_F(NetlistUtilsTest, check_get_nets_at_pins)
    {
        TEST_START

        std::unique_ptr<GateLibrary> lib = create_buffer_lib();

        std::unique_ptr<Netlist> nl = std::make_unique<Netlist>(lib.get());
        ASSERT_NE(nl, nullptr);

        Gate* l0 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l0");
        Gate* l1 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l1");
        Gate* l2 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l2");
        Gate* l3 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l3");
        Gate* l4 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l4");

        Net* n0 = connect(nl.get(), l0, "O", l4, "I0");
        Net* n1 = connect(nl.get(), l1, "O", l4, "I1");
        Net* n2 = connect(nl.get(), l2, "O", l4, "I2");
        Net* n3 = connect(nl.get(), l3, "O", l4, "I3");

        EXPECT_EQ(netlist_utils::get_nets_at_pins(l4, {"I0", "I2"}, true), std::unordered_set<Net*>({n0, n2}));
        EXPECT_EQ(netlist_utils::get_nets_at_pins(l4, {"I1", "I2", "I4"}, true), std::unordered_set<Net*>({n1, n2}));
        EXPECT_EQ(netlist_utils::get_nets_at_pins(l4, {"I1", "I2", "I3"}, true), std::unordered_set<Net*>({n1, n2, n3}));
        EXPECT_EQ(netlist_utils::get_nets_at_pins(l0, {"O"}, false), std::unordered_set<Net*>({n0}));
        EXPECT_EQ(netlist_utils::get_nets_at_pins(l0, {"A", "B", "C"}, true), std::unordered_set<Net*>());

        TEST_END
    }

    /**
     * Testing removal of buffer gates.
     *
     * Functions: remove_buffers
     */
    TEST_F(NetlistUtilsTest, check_remove_buffers)
    {
        TEST_START

        std::unique_ptr<GateLibrary> lib = create_buffer_lib();

        {
            std::unique_ptr<Netlist> nl = std::make_unique<Netlist>(lib.get());
            ASSERT_NE(nl, nullptr);

            Gate* gnd_gate = nl->create_gate(lib->get_gate_type_by_name("GND"), "gnd");
            nl->mark_gnd_gate(gnd_gate);
            Net* gnd_net = nl->create_net("gnd");
            gnd_net->add_source(gnd_gate, "O");
            Gate* vcc_gate = nl->create_gate(lib->get_gate_type_by_name("VCC"), "vcc");
            nl->mark_vcc_gate(vcc_gate);
            Net* vcc_net = nl->create_net("vcc");
            vcc_net->add_source(vcc_gate, "O");

            Gate* g0 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g0");
            Gate* g1 = nl->create_gate(lib->get_gate_type_by_name("BUF"), "g1");
            Gate* g2 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g2");

            Net* n0 = nl->create_net("n0");
            n0->add_destination(g0, "I0");
            n0->mark_global_input_net();

            Net* n1 = nl->create_net("n1");
            n1->add_destination(g0, "I1");
            n1->mark_global_input_net();

            Net* n2 = nl->create_net("n2");
            n2->add_destination(g2, "I1");
            n2->mark_global_input_net();

            Net* n3 = connect(nl.get(), g0, "O", g1, "I");
            Net* n4 = connect(nl.get(), g1, "O", g2, "I0");

            netlist_utils::remove_buffers(nl.get());

            ASSERT_EQ(nl->get_gates().size(), 4);
            ASSERT_EQ(nl->get_nets().size(), 6);

            EXPECT_EQ(g0->get_successor("O")->get_gate(), g2);
        }

        {
            std::unique_ptr<Netlist> nl = std::make_unique<Netlist>(lib.get());
            ASSERT_NE(nl, nullptr);

            Gate* gnd_gate = nl->create_gate(lib->get_gate_type_by_name("GND"), "gnd");
            nl->mark_gnd_gate(gnd_gate);
            Net* gnd_net = nl->create_net("gnd");
            gnd_net->add_source(gnd_gate, "O");
            Gate* vcc_gate = nl->create_gate(lib->get_gate_type_by_name("VCC"), "vcc");
            nl->mark_vcc_gate(vcc_gate);
            Net* vcc_net = nl->create_net("vcc");
            vcc_net->add_source(vcc_gate, "O");

            Gate* g0 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g0");
            Gate* g1 = nl->create_gate(lib->get_gate_type_by_name("LUT2"), "g1");
            g1->add_boolean_function("O", BooleanFunction::from_string("I1"));
            Gate* g2 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g2");

            Net* n0 = nl->create_net("n0");
            n0->add_destination(g0, "I0");
            n0->mark_global_input_net();

            Net* n1 = nl->create_net("n1");
            n1->add_destination(g0, "I1");
            n1->mark_global_input_net();

            Net* n2 = nl->create_net("n2");
            n2->add_destination(g2, "I1");
            n2->mark_global_input_net();

            gnd_net->add_destination(g1, "I0");

            Net* n3 = connect(nl.get(), g0, "O", g1, "I1");
            Net* n4 = connect(nl.get(), g1, "O", g2, "I0");

            netlist_utils::remove_buffers(nl.get());

            ASSERT_EQ(nl->get_gates().size(), 4);
            ASSERT_EQ(nl->get_nets().size(), 6);

            EXPECT_EQ(g0->get_successor("O")->get_gate(), g2);
        }

        {
            std::unique_ptr<Netlist> nl = std::make_unique<Netlist>(lib.get());
            ASSERT_NE(nl, nullptr);

            Gate* gnd_gate = nl->create_gate(lib->get_gate_type_by_name("GND"), "gnd");
            nl->mark_gnd_gate(gnd_gate);
            Net* gnd_net = nl->create_net("gnd");
            gnd_net->add_source(gnd_gate, "O");
            Gate* vcc_gate = nl->create_gate(lib->get_gate_type_by_name("VCC"), "vcc");
            nl->mark_vcc_gate(vcc_gate);
            Net* vcc_net = nl->create_net("vcc");
            vcc_net->add_source(vcc_gate, "O");

            Gate* g0 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g0");
            Gate* g1 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g1");
            Gate* g2 = nl->create_gate(lib->get_gate_type_by_name("AND2"), "g2");

            Net* n0 = nl->create_net("n0");
            n0->add_destination(g0, "I0");
            n0->mark_global_input_net();

            Net* n1 = nl->create_net("n1");
            n1->add_destination(g0, "I1");
            n1->mark_global_input_net();

            Net* n2 = nl->create_net("n2");
            n2->add_destination(g2, "I1");
            n2->mark_global_input_net();

            vcc_net->add_destination(g1, "I0");

            Net* n3 = connect(nl.get(), g0, "O", g1, "I1");
            Net* n4 = connect(nl.get(), g1, "O", g2, "I0");

            netlist_utils::remove_buffers(nl.get());

            ASSERT_EQ(nl->get_gates().size(), 4);
            ASSERT_EQ(nl->get_nets().size(), 6);

            EXPECT_EQ(g0->get_successor("O")->get_gate(), g2);
        }
        
        TEST_END
    }

    /**
     * Testing removal of LUT fan-in endpoints that are not present within the LUT's Boolean function.
     *
     * Functions: remove_buffers
     */
    TEST_F(NetlistUtilsTest, check_remove_unused_lut_endpoints)
    {
        TEST_START

        std::unique_ptr<GateLibrary> lib = create_buffer_lib();

        std::unique_ptr<Netlist> nl = std::make_unique<Netlist>(lib.get());
        ASSERT_NE(nl, nullptr);

        Gate* gnd_gate = nl->create_gate(lib->get_gate_type_by_name("GND"), "gnd");
        nl->mark_gnd_gate(gnd_gate);
        Net* gnd_net = nl->create_net("gnd");
        gnd_net->add_source(gnd_gate, "O");

        Gate* l0 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l0");
        Gate* l1 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l1");
        Gate* l2 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l2");
        Gate* l3 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l3");
        Gate* l4 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l4");
        Gate* l5 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l5");
        l4->add_boolean_function("O", BooleanFunction::from_string("I0 & I1 & I2 & I3"));
        l5->add_boolean_function("O", BooleanFunction::from_string("I2"));

        connect(nl.get(), l0, "O", l4, "I0");
        connect(nl.get(), l1, "O", l4, "I1");
        connect(nl.get(), l2, "O", l4, "I2");
        connect(nl.get(), l3, "O", l4, "I3");

        connect(nl.get(), l0, "O", l5, "I0");
        connect(nl.get(), l1, "O", l5, "I1");
        connect(nl.get(), l2, "O", l5, "I2");
        connect(nl.get(), l3, "O", l5, "I3");

        EXPECT_EQ(l4->get_predecessor("I0")->get_gate(), l0);
        EXPECT_EQ(l4->get_predecessor("I1")->get_gate(), l1);
        EXPECT_EQ(l4->get_predecessor("I2")->get_gate(), l2);
        EXPECT_EQ(l4->get_predecessor("I3")->get_gate(), l3);

        EXPECT_EQ(l5->get_predecessor("I0")->get_gate(), l0);
        EXPECT_EQ(l5->get_predecessor("I1")->get_gate(), l1);
        EXPECT_EQ(l5->get_predecessor("I2")->get_gate(), l2);
        EXPECT_EQ(l5->get_predecessor("I3")->get_gate(), l3);

        netlist_utils::remove_unused_lut_endpoints(nl.get());

        EXPECT_EQ(l4->get_predecessor("I0")->get_gate(), l0);
        EXPECT_EQ(l4->get_predecessor("I1")->get_gate(), l1);
        EXPECT_EQ(l4->get_predecessor("I2")->get_gate(), l2);
        EXPECT_EQ(l4->get_predecessor("I3")->get_gate(), l3);

        EXPECT_EQ(l5->get_predecessor("I0")->get_gate(), gnd_gate);
        EXPECT_EQ(l5->get_predecessor("I1")->get_gate(), gnd_gate);
        EXPECT_EQ(l5->get_predecessor("I2")->get_gate(), l2);
        EXPECT_EQ(l5->get_predecessor("I3")->get_gate(), gnd_gate);

        TEST_END
    } 

    /**
     * Testing detection of common inputs of gates.
     *
     * Functions: get_common_inputs
     */
    TEST_F(NetlistUtilsTest, check_get_common_inputs)
    {
        TEST_START

        std::unique_ptr<GateLibrary> lib = create_buffer_lib();

        std::unique_ptr<Netlist> nl = std::make_unique<Netlist>(lib.get());
        ASSERT_NE(nl, nullptr);

        Gate* gnd_gate = nl->create_gate(lib->get_gate_type_by_name("GND"), "gnd");
        nl->mark_gnd_gate(gnd_gate);

        Gate* l0 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l0");
        Gate* l1 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l1");
        Gate* l2 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l2");
        Gate* l3 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l3");
        Gate* l4 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l4");
        Gate* l5 = nl->create_gate(lib->get_gate_type_by_name("LUT4"), "l4");

        Net* gnd_net = connect(nl.get(), gnd_gate, "O", l2, "I0");
        connect(nl.get(), gnd_gate, "O", l3, "I0");
        connect(nl.get(), gnd_gate, "O", l4, "I0");
        connect(nl.get(), gnd_gate, "O", l5, "I0");
        Net* common_net4 = connect(nl.get(), l0, "O", l2, "I2");
        connect(nl.get(), l0, "O", l3, "I2");
        connect(nl.get(), l0, "O", l4, "I2");
        connect(nl.get(), l0, "O", l5, "I2");
        Net* common_net2 = connect(nl.get(), l1, "O", l2, "I1");
        connect(nl.get(), l1, "O", l3, "I1");

        std::vector<Gate*> gates = {l2, l3, l4, l5};
        std::vector<Net*> common_nets4 = netlist_utils::get_common_inputs(gates);
        std::vector<Net*> common_nets2 = netlist_utils::get_common_inputs(gates, 2);

        ASSERT_EQ(common_nets4.size(), 1);
        EXPECT_TRUE(std::find(common_nets4.begin(), common_nets4.end(), common_net4) != common_nets4.end());

        ASSERT_EQ(common_nets2.size(), 2);
        EXPECT_TRUE(std::find(common_nets2.begin(), common_nets2.end(), common_net2) != common_nets2.end());
        EXPECT_TRUE(std::find(common_nets2.begin(), common_nets2.end(), common_net4) != common_nets2.end());

        TEST_END
    } 

}    //namespace hal
