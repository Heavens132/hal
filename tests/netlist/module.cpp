#include "netlist/module.h"
#include "netlist/event_system/gate_event_handler.h"
#include "netlist/gate_library/gate_library_manager.h"
#include "netlist/netlist.h"
#include "netlist/netlist_factory.h"
#include "test_def.h"
#include "gtest/gtest.h"
#include <core/log.h>
#include <iostream>
#include <netlist/gate.h>
#include <netlist/net.h>

class module_test : public ::testing::Test
{
protected:
    const std::string g_lib_name = "EXAMPLE_GATE_LIBRARY";
    const u32 MIN_MODULE_ID = 2;
    //const u32 MIN_NL_ID = 1;
    const u32 MIN_GATE_ID = 1;
    const u32 MIN_NET_ID = 1;

    virtual void SetUp()
    {
        NO_COUT_BLOCK;
        gate_library_manager::load_all();
    }

    virtual void TearDown()
    {
    }

    // Creates an empty netlist with a certain id if passed
    std::shared_ptr<netlist> create_empty_netlist(const int id = -1)
    {
        NO_COUT_BLOCK;
        std::shared_ptr<gate_library> gl = gate_library_manager::get_gate_library(g_lib_name);
        std::shared_ptr<netlist> nl(new netlist(gl));

        if (id >= 0)
        {
            nl->set_id(id);
        }
        return nl;
    }

    /*
     *      Example netlist circuit diagram (Id in brackets). Used for get fan in and
     *      out nets.
     *
     *
     *      GND (1) =-= INV (3) =--=             .------= INV (4) =
     *                                 AND2 (0) =-
     *      VCC (2) =--------------=             '------=
     *                                                     AND2 (5) =
     *                                                  =
     *
     *     =                       =           =----------=           =
     *       BUF (6)              ... OR2 (7)             ... OR2 (8)
     *     =                       =           =          =           =
     */

    // Creates a simple netlist shown in the diagram above
    std::shared_ptr<netlist> create_example_netlist(int id = -1)
    {
        NO_COUT_BLOCK;
        std::shared_ptr<gate_library> gl = gate_library_manager::get_gate_library(g_lib_name);
        std::shared_ptr<netlist> nl      = std::make_shared<netlist>(gl);
        if (id >= 0)
        {
            nl->set_id(id);
        }

        // Create the gates
        std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "AND2", "gate_0");
        std::shared_ptr<gate> gate_1 = nl->create_gate(MIN_GATE_ID+1, "GND", "gate_1");
        std::shared_ptr<gate> gate_2 = nl->create_gate(MIN_GATE_ID+2, "VCC", "gate_2");
        std::shared_ptr<gate> gate_3 = nl->create_gate(MIN_GATE_ID+3, "INV", "gate_3");
        std::shared_ptr<gate> gate_4 = nl->create_gate(MIN_GATE_ID+4, "INV", "gate_4");
        std::shared_ptr<gate> gate_5 = nl->create_gate(MIN_GATE_ID+5, "AND2", "gate_5");
        std::shared_ptr<gate> gate_6 = nl->create_gate(MIN_GATE_ID+6, "BUF", "gate_6");
        std::shared_ptr<gate> gate_7 = nl->create_gate(MIN_GATE_ID+7, "OR2", "gate_7");
        std::shared_ptr<gate> gate_8 = nl->create_gate(MIN_GATE_ID+8, "OR2", "gate_8");

        // Add the nets (net_x_y1_y2... := net between the gate with id x and the gates y1,y2,...)
        std::shared_ptr<net> net_1_3 = nl->create_net(MIN_NET_ID+13, "net_1_3");
        net_1_3->set_src(gate_1, "O");
        net_1_3->add_dst(gate_3, "I");

        std::shared_ptr<net> net_3_0 = nl->create_net(MIN_NET_ID+30, "net_3_0");
        net_3_0->set_src(gate_3, "O");
        net_3_0->add_dst(gate_0, "I0");

        std::shared_ptr<net> net_2_0 = nl->create_net(MIN_NET_ID+20, "net_2_0");
        net_2_0->set_src(gate_2, "O");
        net_2_0->add_dst(gate_0, "I1");

        std::shared_ptr<net> net_0_4_5 = nl->create_net(MIN_NET_ID+045, "net_0_4_5");
        net_0_4_5->set_src(gate_0, "O");
        net_0_4_5->add_dst(gate_4, "I");
        net_0_4_5->add_dst(gate_5, "I0");

        std::shared_ptr<net> net_7_8 = nl->create_net(MIN_NET_ID+78, "net_7_8");
        net_7_8->set_src(gate_7, "O");
        net_7_8->add_dst(gate_8, "I0");

        return nl;
    }
};

/**
 * Testing the access on the id, the type and the stored netlist after calling the constructor
 *
 * Functions: constructor, get_id, get_name
 */
TEST_F(module_test, check_constructor){
    TEST_START
        {
            // Creating a module of id 123 and type "test module"
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+123, "test module", nl->get_top_module());

            EXPECT_EQ(test_module->get_id(), (u32)(MIN_MODULE_ID+123));
            EXPECT_EQ(test_module->get_name(), "test module");
            EXPECT_EQ(test_module->get_netlist(), nl);
            }
    TEST_END
}

/**
 * Testing the set_name function of module
 *
 * Functions: set_name
 */
TEST_F(module_test, check_set_id){
    TEST_START
        {
            // Set a new name for module
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());

            test_module->set_name("new_name");
            EXPECT_EQ(test_module->get_name(), "new_name");
        }
        {
            // Set an already set name
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());

            test_module->set_name("test_module");
            EXPECT_EQ(test_module->get_name(), "test_module");
        }
        { //TODO: Fails
            // Set an empty name
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());

            test_module->set_name("");
            EXPECT_EQ(test_module->get_name(), "test_module");
        }
    TEST_END
}

/**
 * Testing the contains_gate function
 *
 * Functions: contains_gate
 */
TEST_F(module_test, check_contains_gate){
    TEST_START
        // POSITIVE
        {
            // Check a gate, that is part of the module (not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            m_0->insert_gate(gate_0);

            EXPECT_TRUE(m_0->contains_gate(gate_0));
        }
        {
            // Check a gate, that isn't part of the module (not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");

            EXPECT_FALSE(m_0->contains_gate(gate_0));
        }
        {
            // Check a gate, that isn't part of the module, but of a submodule (not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "test_module", m_0);
            ASSERT_NE(submodule, nullptr);
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            submodule->insert_gate(gate_0);

            EXPECT_FALSE(m_0->contains_gate(gate_0));
        }
        {
            // Check a gate, that isn't part of the module, but of a submodule (recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "test_module", m_0);
            ASSERT_NE(submodule, nullptr);
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            submodule->insert_gate(gate_0);

            EXPECT_TRUE(m_0->contains_gate(gate_0, true));
        }
    TEST_END
}

/**
 * Testing the addition of gates to the module. Verify the addition by call the
 * get_gates function and the contains_gate function
 *
 * Functions: insert_gate
 */
TEST_F(module_test, check_insert_gate){
    TEST_START
        {
            // Add some gates to the module
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            std::shared_ptr<gate> gate_1 = nl->create_gate(MIN_GATE_ID+1, "INV", "gate_1");
            // this gate is not part of the module
            std::shared_ptr<gate> gate_not_in_m = nl->create_gate(MIN_GATE_ID+2, "INV", "gate_not_in_m");

            // Add gate_0 and gate_1 to a module
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            test_module->insert_gate(gate_0);
            test_module->insert_gate(gate_1);

            std::set<std::shared_ptr<gate>> expRes = {gate_0, gate_1};

            EXPECT_EQ(test_module->get_gates(), expRes);
            EXPECT_TRUE(test_module->contains_gate(gate_0));
            EXPECT_TRUE(test_module->contains_gate(gate_1));
            EXPECT_FALSE(test_module->contains_gate(gate_not_in_m));
        }
        {
            // Add the same gate twice to the module
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl  = create_empty_netlist();
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");

            // Add gate_0 twice
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            test_module->insert_gate(gate_0);
            test_module->insert_gate(gate_0);

            std::set<std::shared_ptr<gate>> expRes = {
                gate_0,
            };

            EXPECT_EQ(test_module->get_gates(), expRes);
            EXPECT_TRUE(test_module->contains_gate(gate_0));
        }
        {
            // Insert a gate owned by a submodule
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl  = create_empty_netlist();
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");

            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "submodule", test_module);
            submodule->insert_gate(gate_0);
            ASSERT_TRUE(submodule->contains_gate(gate_0));
            ASSERT_FALSE(test_module->contains_gate(gate_0));

            test_module->insert_gate(gate_0);

            std::set<std::shared_ptr<gate>> expRes = {
                    gate_0
            };

            EXPECT_EQ(test_module->get_gates(), expRes);
            EXPECT_FALSE(submodule->contains_gate(gate_0));
        }

        // NEGATIVE
        {
            // Gate is a nullptr
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            test_module->insert_gate(nullptr);
            EXPECT_TRUE(test_module->get_gates().empty());
        }
    TEST_END
}

/**
 * Testing the deletion of gates from modules
 *
 * Functions: remove_gate
 */
TEST_F(module_test, check_remove_gate){
    TEST_START
        {
            // Delete a gate from a module (gate owned by the modules)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            m_0->insert_gate(gate_0);

            ASSERT_TRUE(m_0->contains_gate(gate_0));
            m_0->remove_gate(gate_0);
            EXPECT_FALSE(m_0->contains_gate(gate_0));
        }
        {
            // Try to delete a gate from a module (gate owned by another module)
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> m_other = nl->create_module(MIN_MODULE_ID+1, "other_test_module", nl->get_top_module());
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            m_other->insert_gate(gate_0);

            m_0->remove_gate(gate_0);
            EXPECT_FALSE(m_0->contains_gate(gate_0));
            EXPECT_TRUE(m_other->contains_gate(gate_0));
        }
        // NEGATIVE
        {
            // Try to delete a gate from the top-module (should change nothing)
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV", "gate_0");
            std::shared_ptr<module> tm =  nl->get_top_module();

            ASSERT_TRUE(tm->contains_gate(gate_0));
            tm->remove_gate(gate_0);
            EXPECT_TRUE(tm->contains_gate(gate_0));
        }
        {
            // Try to delete a nullptr (should not crash)
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());

            m_0->remove_gate(nullptr);
        }
    TEST_END
}

/**
 * Testing the get_gate_by_id function
 *
 * Functions: get_gate_by_id
 */
TEST_F(module_test, check_get_gate_by_id){
    TEST_START
        // POSITIVE
        {
            // get a gate by its id (gate owned by module)(not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<gate> gate_123 = nl->create_gate(MIN_GATE_ID+123, "INV", "gate_123");
            m_0->insert_gate(gate_123);

            ASSERT_TRUE(m_0->contains_gate(gate_123));
            EXPECT_EQ(m_0->get_gate_by_id(MIN_GATE_ID+123), gate_123);
        }
        {
            // get a gate by its id (not owned by a submodule)(not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "other_module", m_0);
            std::shared_ptr<gate> gate_123 = nl->create_gate(MIN_GATE_ID+123, "INV", "gate_123");
            submodule->insert_gate(gate_123);

            EXPECT_EQ(m_0->get_gate_by_id(MIN_GATE_ID+123), nullptr);
        }
        {
            // get a gate by its id (not owned by a submodule)(recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "other_module", m_0);
            std::shared_ptr<gate> gate_123 = nl->create_gate(MIN_GATE_ID+123, "INV", "gate_123");
            submodule->insert_gate(gate_123);

            EXPECT_EQ(m_0->get_gate_by_id(MIN_GATE_ID+123, true), gate_123);
        }
    TEST_END
}

/**
 * Testing the contains_net function
 *
 * Functions: contains_net
 */
TEST_F(module_test, check_contains_net){
    TEST_START
        // POSITIVE
        {
            // Check a net, that is part of the module (not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            m_0->insert_net(net_0);

            EXPECT_TRUE(m_0->contains_net(net_0));
        }
        {
            // Check a net, that isn't part of the module (not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");

            EXPECT_FALSE(m_0->contains_net(net_0));
        }
        {
            // Check a net, that isn't part of the module, but of a submodule (not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "test_module", m_0);
            ASSERT_NE(submodule, nullptr);
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            submodule->insert_net(net_0);

            EXPECT_FALSE(m_0->contains_net(net_0));
        }
        {
            // Check a net, that isn't part of the module, but of a submodule (recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "test_module", m_0);
            ASSERT_NE(submodule, nullptr);
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            submodule->insert_net(net_0);

            EXPECT_TRUE(m_0->contains_net(net_0, true));
        }
    TEST_END
}

/**
 * Testing the addition of nets to the module. Verify the addition by call the
 * get_nets function and the contains_net and get_nets function
 *
 * Functions: insert_net
 */
TEST_F(module_test, check_insert_net){
    TEST_START
        {
            // Add some nets to the module
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            std::shared_ptr<net> net_1 = nl->create_net(MIN_NET_ID+1, "net_1");
            // this net is not part of the module
            std::shared_ptr<net> net_not_in_m = nl->create_net(MIN_NET_ID+2, "net_not_in_m");

            // Add net_0 and net_1 to a module
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+1, "test module", nl->get_top_module());
            test_module->insert_net(net_0);
            test_module->insert_net(net_1);

            std::set<std::shared_ptr<net>> expRes = {net_0, net_1};

            EXPECT_EQ(test_module->get_nets(), expRes);
        }
        {
            NO_COUT_TEST_BLOCK;
            // Add the same net twice to the module
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<net> net_0  = nl->create_net(MIN_NET_ID+0, "net_0");

            // Add net_0 twice
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            test_module->insert_net(net_0);
            test_module->insert_net(net_0);

            std::set<std::shared_ptr<net>> expRes = {
                net_0,
            };

            EXPECT_EQ(test_module->get_nets(), expRes);
            EXPECT_TRUE(test_module->contains_net(net_0));
        }
        {
            // Insert a net owned by a submodule
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl  = create_empty_netlist();
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");

            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "submodule", test_module);
            submodule->insert_net(net_0);
            ASSERT_TRUE(submodule->contains_net(net_0));
            ASSERT_FALSE(test_module->contains_net(net_0));

            test_module->insert_net(net_0);

            std::set<std::shared_ptr<net>> expRes = {
                    net_0
            };

            EXPECT_EQ(test_module->get_nets(), expRes);
            EXPECT_FALSE(submodule->contains_net(net_0));
        }

        // NEGATIVE
        {
            // Net is a nullptr
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl   = create_empty_netlist();
            std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test module", nl->get_top_module());
            test_module->insert_net(nullptr);
            EXPECT_TRUE(test_module->get_nets().empty());
        }

        TEST_END
}

/**
 * Testing the deletion of nets from modules
 *
 * Functions: remove_net
 */
TEST_F(module_test, check_remove_net){
    TEST_START
        {
            // Delete a net from a module (net owned by the modules)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            m_0->insert_net(net_0);

            ASSERT_TRUE(m_0->contains_net(net_0));
            m_0->remove_net(net_0);
            EXPECT_FALSE(m_0->contains_net(net_0));
        }
        {
            // Try to delete a net from a module (net owned by another module)
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> m_other = nl->create_module(MIN_MODULE_ID+1, "other_test_module", nl->get_top_module());
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            m_other->insert_net(net_0);

            m_0->remove_net(net_0);
            EXPECT_FALSE(m_0->contains_net(net_0));
            EXPECT_TRUE(m_other->contains_net(net_0));
        }
        // NEGATIVE
        {
            // Try to delete a net from the top-module (should change nothing)
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<net> net_0 = nl->create_net(MIN_NET_ID+0, "net_0");
            std::shared_ptr<module> tm =  nl->get_top_module();

            ASSERT_TRUE(tm->contains_net(net_0));
            tm->remove_net(net_0);
            EXPECT_TRUE(tm->contains_net(net_0));
        }
        {
            // Try to delete a nullptr (should not crash)
            NO_COUT_TEST_BLOCK;
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());

            m_0->remove_net(nullptr);
        }
    TEST_END
}

/**
 * Testing the get_net_by_id function
 *
 * Functions: get_net_by_id
 */
TEST_F(module_test, check_get_net_by_id){
    TEST_START
        // POSITIVE
        {
            // get a net by its id (net owned by module)(not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<net> net_123 = nl->create_net(MIN_NET_ID+123, "net_123");
            m_0->insert_net(net_123);

            ASSERT_TRUE(m_0->contains_net(net_123));
            EXPECT_EQ(m_0->get_net_by_id(MIN_NET_ID+123), net_123);
        }
        {
            // get a net by its id (not owned by a submodule)(not recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "other_module", m_0);
            std::shared_ptr<net> net_123 = nl->create_net(MIN_NET_ID+123, "net_123");
            submodule->insert_net(net_123);

            EXPECT_EQ(m_0->get_net_by_id(MIN_NET_ID+123), nullptr);
        }
        {
            // get a net by its id (not owned by a submodule)(recursive)
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
            std::shared_ptr<module> submodule = nl->create_module(MIN_MODULE_ID+1, "other_module", m_0);
            std::shared_ptr<net> net_123 = nl->create_net(MIN_NET_ID+123, "net_123");
            submodule->insert_net(net_123);

            EXPECT_EQ(m_0->get_net_by_id(MIN_NET_ID+123, true), net_123);
        }
    TEST_END
}


/**
 * Testing the access on submodules. Therefore we build up a module tree like this:
 *
 *               .----> MODULE_0
 *               |
 * TOP_MODULE ---+                .--> MODULE_2
 *               |                |
 *               '----> MODULE_1 -+
 *                                |
 *                                '--> MODULE_3
 *
 *   (Remark: MODULE_0 and MODULE_2 are both named "even_module", while MODULE_1 and MODULE_3 are named "odd_module")
 *
 * Functions: get_submodules
 */
TEST_F(module_test, check_get_submodules){
    TEST_START
        // Set up the module tree
        std::shared_ptr<netlist> nl = create_empty_netlist();
        std::shared_ptr<module> tm = nl->get_top_module();
        ASSERT_NE(tm, nullptr);
        std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "even_module", tm);
        ASSERT_NE(m_0, nullptr);
        std::shared_ptr<module> m_1 = nl->create_module(MIN_MODULE_ID+1, "odd_module", tm);
        ASSERT_NE(m_1, nullptr);
        std::shared_ptr<module> m_2 = nl->create_module(MIN_MODULE_ID+2, "even_module", m_1);
        ASSERT_NE(m_2, nullptr);
        std::shared_ptr<module> m_3 = nl->create_module(MIN_MODULE_ID+3, "odd_module", m_1);
        ASSERT_NE(m_3, nullptr);
        {
            // Testing the access on submodules (no name_filter, not recursive)
            {
                // Submodules of TOP_MODULE;
                std::set<std::shared_ptr<module>> exp_result = {m_0, m_1};
                EXPECT_EQ(tm->get_submodules(DONT_CARE, false), exp_result);
            }
            {
                // Submodules of MODULE_1;
                std::set<std::shared_ptr<module>> exp_result = {m_2, m_3};
                EXPECT_EQ(m_1->get_submodules(DONT_CARE, false), exp_result);
            }
            {
                // Submodules of MODULE_0;
                std::set<std::shared_ptr<module>> exp_result = {};
                EXPECT_EQ(m_0->get_submodules(DONT_CARE, false), exp_result);
            }
        }
        {
            // Testing the access on submodules (name_filter set, not recursive)
            {
                // Submodules of TOP_MODULE;
                std::set<std::shared_ptr<module>> exp_result = {m_0};
                EXPECT_EQ(tm->get_submodules("even_module", false), exp_result);
            }
            {
                // Submodules of MODULE_1;
                std::set<std::shared_ptr<module>> exp_result = {m_2};
                EXPECT_EQ(m_1->get_submodules("even_module", false), exp_result);
            }
            {
                // Submodules of TOP_MODULE (name does not exists);
                std::set<std::shared_ptr<module>> exp_result = {};
                EXPECT_EQ(tm->get_submodules("non_existing_name", false), exp_result);
            }
        }
        {
            // Testing the access on submodules (recursive)
            {
                // Submodules of TOP_MODULE;
                std::set<std::shared_ptr<module>> exp_result = {m_0,m_1,m_2,m_3};
                EXPECT_EQ(tm->get_submodules(DONT_CARE, true), exp_result);
            }
            {
                // Submodules of TOP_MODULE (with name_filter);
                std::set<std::shared_ptr<module>> exp_result = {m_0,m_2};
                EXPECT_EQ(tm->get_submodules("even_module", true), exp_result);
            }
            {
                // Submodules of MODULE_0
                std::set<std::shared_ptr<module>> exp_result = {};
                EXPECT_EQ(m_0->get_submodules(DONT_CARE, true), exp_result);
            }
        }
    TEST_END
}

/*
 *      Testing the get_input_nets, get_output_nets by using the following example netlist with a module
 *
 *                     ################################################
 *                     # TEST_MODULE                                  #
 *                     #                                              #
 *      global_in -----§---------------=  INV (0)  = -----------------§----- global_out
 *                     #                                              #
 *                 .---§--= INV (1) =--=                              #
 *                 |   #                  AND2 (2) =--+---------------§----= INV (5)
 *    = INV (4) =--+---§---------------=              |               #
 *                     #                              '--= INV (3) =  #
 *                     #                                              #
 *                     ################################################
 *
 *
 */

/**
 * Testing the get_input_nets and get_output_nets function
 *
 * Functions: get_input_nets, get_output_nets
 */
TEST_F(module_test, check_get_input_nets){
    TEST_START
        // +++ Create the example netlist (see above)

        std::shared_ptr<netlist> nl = create_empty_netlist();

        // Add the gates
        std::shared_ptr<gate> gate_0 = nl->create_gate(MIN_GATE_ID+0, "INV" , "gate_0");
        std::shared_ptr<gate> gate_1 = nl->create_gate(MIN_GATE_ID+1, "INV" , "gate_1");
        std::shared_ptr<gate> gate_2 = nl->create_gate(MIN_GATE_ID+2, "AND2", "gate_2");
        std::shared_ptr<gate> gate_3 = nl->create_gate(MIN_GATE_ID+3, "INV" , "gate_3");
        std::shared_ptr<gate> gate_4 = nl->create_gate(MIN_GATE_ID+4, "INV" , "gate_4");
        std::shared_ptr<gate> gate_5 = nl->create_gate(MIN_GATE_ID+5, "INV" , "gate_5");

        // Add the nets (net_x_y1_y2_... is net from x to y1,y2,... (g = global input/output))
        std::shared_ptr<net> net_g_0   = nl->create_net(MIN_NET_ID+0, "name_0");
        std::shared_ptr<net> net_0_g   = nl->create_net(MIN_NET_ID+1, "name_0");
        std::shared_ptr<net> net_1_2   = nl->create_net(MIN_NET_ID+3, "name_0");
        std::shared_ptr<net> net_4_1_2 = nl->create_net(MIN_NET_ID+4, "name_1");
        std::shared_ptr<net> net_2_3_5 = nl->create_net(MIN_NET_ID+5, "name_1");

        // Connect the nets
        net_g_0->add_dst(gate_0, "I");

        net_0_g->set_src(gate_0, "O");

        net_4_1_2->set_src(gate_4, "O");
        net_4_1_2->add_dst(gate_1, "I");
        net_4_1_2->add_dst(gate_2, "I1");

        net_1_2->set_src(gate_1, "O");
        net_1_2->add_dst(gate_2, "I0");

        net_2_3_5->set_src(gate_2, "O");
        net_2_3_5->add_dst(gate_3, "I");
        net_2_3_5->add_dst(gate_5, "I");

        // Mark global nets
        nl->mark_global_input_net(net_g_0);
        nl->mark_global_output_net(net_0_g);

        // Create the module
        std::shared_ptr<module> test_module = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());
        for(auto g : std::set<std::shared_ptr<gate>>({gate_0, gate_1, gate_2, gate_3})){
            test_module->insert_gate(g);
        }
        for(auto n : std::set<std::shared_ptr<net>>({net_g_0, net_0_g, net_1_2, net_4_1_2, net_2_3_5})){
            test_module->insert_net(n);
        }

        {
            // Get input nets of the test module (no name filter)
            std::set<std::shared_ptr<net>> exp_result = {net_g_0, net_4_1_2};
            EXPECT_EQ(test_module->get_input_nets(), exp_result);
        }
        {
            // Get input nets of the test module with name_1
            std::set<std::shared_ptr<net>> exp_result = {net_4_1_2};
            EXPECT_EQ(test_module->get_input_nets("name_1"), exp_result);
        }
        {
            // Get output nets of the test module (no name filter)
            std::set<std::shared_ptr<net>> exp_result = {net_0_g, net_2_3_5};
            EXPECT_EQ(test_module->get_output_nets(), exp_result);
        }
        {
            // Get output nets of the test module with name_1
            std::set<std::shared_ptr<net>> exp_result = {net_2_3_5};
            EXPECT_EQ(test_module->get_output_nets("name_1"), exp_result);
        }

    TEST_END
}

/**
 * Testing <stuff>
 *
 * Functions: <functions>
 */
TEST_F(module_test, check_s){
    TEST_START
        {
            // <do shit>
            std::shared_ptr<netlist> nl = create_empty_netlist();
            std::shared_ptr<module> m_0 = nl->create_module(MIN_MODULE_ID+0, "test_module", nl->get_top_module());

        }
    TEST_END
}

