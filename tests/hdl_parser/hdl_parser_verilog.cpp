#include "netlist/gate.h"
#include "netlist/gate_library/gate_library_manager.h"
#include "netlist/netlist.h"
#include "netlist/netlist_factory.h"
#include "netlist/persistent/netlist_serializer.h"
#include "test_def.h"
#include "gtest/gtest.h"
#include <core/log.h>
#include <hdl_parser/hdl_parser_verilog.h>
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>

class hdl_parser_verilog_test : public ::testing::Test
{
protected:
    const std::string g_lib_name = "EXAMPLE_GATE_LIBRARY";
    const std::string temp_lib_name = "TEMP_GATE_LIBRARY";
    // Minimum id for netlists, gates, nets and modules
    const u32 INVALID_GATE_ID = 0;
    const u32 INVALID_NET_ID = 0;
    const u32 INVALID_MODULE_ID = 0;
    const u32 MIN_MODULE_ID = 2;
    const u32 MIN_GATE_ID = 1;
    const u32 MIN_NET_ID = 1;
    const u32 MIN_NETLIST_ID = 1;
    const u32 TOP_MODULE_ID = 1;
    // Path used, to create a custom gate library (used to test certain behaviour of input and output vectors)
    hal::path temp_lib_path;

    virtual void SetUp()
    {
        NO_COUT_BLOCK;
        temp_lib_path = core_utils::get_gate_library_directories()[0] / "temp_lib.json";
        gate_library_manager::load_all();
    }

    virtual void TearDown()
    {
        boost::filesystem::remove(temp_lib_path);
    }

    // Creates an endpoint from a gate and a pin_type
    endpoint get_endpoint(std::shared_ptr<gate> g, std::string pin_type)
    {
        endpoint ep;
        ep.gate     = g;
        ep.pin_type = pin_type;
        return ep;
    }

    // NOTE: temporary
    std::shared_ptr<netlist> create_example_parse_netlist(int id = -1)
    {
        //NO_COUT_BLOCK;
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
        //std::shared_ptr<gate> gate_6 = nl->create_gate(MIN_GATE_ID+6, "BUF", "gate_6");
        std::shared_ptr<gate> gate_6 = nl->create_gate(MIN_GATE_ID+6, "OR2", "gate_6");
        std::shared_ptr<gate> gate_7 = nl->create_gate(MIN_GATE_ID+7, "OR2", "gate_7");

        // Add the nets (net_x_y1_y2... := net between the gate with id x and the gates y1,y2,...)
        std::shared_ptr<net> net_1_3 = nl->create_net(MIN_NET_ID+13, "0_net");
        net_1_3->set_src(gate_1, "O");
        net_1_3->add_dst(gate_3, "I");

        std::shared_ptr<net> net_3_0 = nl->create_net(MIN_NET_ID+30, "net_3_0");
        net_3_0->set_src(gate_3, "O");
        net_3_0->add_dst(gate_0, "I0");

        std::shared_ptr<net> net_2_0 = nl->create_net(MIN_NET_ID+20, "1_net");
        net_2_0->set_src(gate_2, "O");
        net_2_0->add_dst(gate_0, "I1");

        std::shared_ptr<net> net_0_4_5 = nl->create_net(MIN_NET_ID+045, "net_0_4_5");
        net_0_4_5->set_src(gate_0, "O");
        net_0_4_5->add_dst(gate_4, "I");
        net_0_4_5->add_dst(gate_5, "I0");

        std::shared_ptr<net> net_6_7 = nl->create_net(MIN_NET_ID+67, "net_6_7");
        net_6_7->set_src(gate_6, "O");
        net_6_7->add_dst(gate_7, "I0");

        std::shared_ptr<net> net_4_out = nl->create_net(MIN_NET_ID+400, "net_4_out");
        net_4_out->set_src(gate_4, "O");

        std::shared_ptr<net> net_5_out = nl->create_net(MIN_NET_ID+500, "net_5_out");
        net_5_out->set_src(gate_5, "O");

        std::shared_ptr<net> net_7_out = nl->create_net(MIN_NET_ID+700, "net_7_out");
        net_7_out->set_src(gate_7, "O");

        return nl;
    }

    // Create and load temporarily a custom gate library, which contains gates with input and output vectors up to dimension 3
    void create_temp_gate_lib()
    {
        NO_COUT_BLOCK;
        std::ofstream test_lib(temp_lib_path.string());
        test_lib << "{\n"
                    "    \"library\": {\n"
                    "        \"library_name\": \"TEMP_GATE_LIBRARY\",\n"
                    "        \"elements\": {\n"
                    "\t    \"GATE0\" : [[\"I\"], [], [\"O\"]],\n"
                    "            \"GATE1\" : [[\"I(0)\",\"I(1)\",\"I(2)\",\"I(3)\",\"I(4)\"], [], [\"O(0)\",\"O(1)\",\"O(2)\",\"O(3)\", \"O(4)\"]],\n"
                    "            \"GATE2\" : [[\"I(0, 0)\",\"I(0, 1)\",\"I(1, 0)\",\"I(1, 1)\"], [], [\"O(0, 0)\",\"O(0, 1)\",\"O(1, 0)\",\"O(1, 1)\"]],\n"
                    "            \"GATE3\" : [[\"I(0, 0, 0)\",\"I(0, 0, 1)\",\"I(0, 1, 0)\",\"I(0, 1, 1)\",\"I(1, 0, 0)\",\"I(1, 0, 1)\",\"I(1, 1, 0)\",\"I(1, 1, 1)\"], [], [\"O(0, 0, 0)\",\"O(0, 0, 1)\",\"O(0, 1, 0)\",\"O(0, 1, 1)\",\"O(1, 0, 0)\",\"O(1, 0, 1)\",\"O(1, 1, 0)\",\"O(1, 1, 1)\"]],\n"
                    "\n"
                    "            \"GND\" : [[], [], [\"O\"]],\n"
                    "            \"VCC\" : [[], [], [\"O\"]]\n"
                    "        },\n"
                    "        \"vhdl_includes\": [],\n"
                    "        \"global_gnd_nodes\": [\"GND\"],\n"
                    "        \"global_vcc_nodes\": [\"VCC\"]\n"
                    "    }\n"
                    "}";
        test_lib.close();

        gate_library_manager::load_all();
    }

    // Checks if two vectors have the same content regardless of their order
    template<typename T>
    bool vectors_have_same_content(std::vector<T> vec_1, std::vector<T> vec_2)
    {
        if (vec_1.size() != vec_2.size())
        {
            return false;
        }

        // Each element of vec_1 must be found in vec_2
        while (vec_1.size() > 0)
        {
            auto it_1       = vec_1.begin();
            bool found_elem = false;
            for (auto it_2 = vec_2.begin(); it_2 != vec_2.end(); it_2++)
            {
                if (*it_1 == *it_2)
                {
                    found_elem = true;
                    vec_2.erase(it_2);
                    break;
                }
            }
            if (!found_elem)
            {
                return false;
            }
            vec_1.erase(it_1);
        }

        return true;
    }
};


/**
 * Used by clueless testers to understand the verilog syntax used by the writer (will be removed later)
 */
//#include "hdl_writer/hdl_writer_verilog.h"
/*TEST_F(hdl_parser_verilog_test, check_temporary)
{
    TEST_START
        {
            // Write and parse the example netlist (with some additions) and compare the result with the original netlist
            std::shared_ptr<netlist> nl = create_example_parse_netlist(0);


            // Mark the global gates as such
            nl->mark_global_gnd_gate(nl->get_gate_by_id(MIN_GATE_ID+1));
            nl->mark_global_vcc_gate(nl->get_gate_by_id(MIN_GATE_ID+2));

            // Mark global output nets
            nl->mark_global_output_net(nl->get_net_by_id(MIN_NET_ID+400));
            nl->mark_global_input_net(nl->get_net_by_id(MIN_NET_ID+500));
            nl->mark_global_inout_net(nl->get_net_by_id(MIN_NET_ID+700));

            nl->set_device_name("test_device_name");


            // Write and parse the netlist now
            test_def::capture_stdout();
            std::stringstream parser_input;
            hdl_writer_verilog verilog_writer(parser_input);


            // Writes the netlist in the sstream
            bool writer_suc = verilog_writer.write(nl);

            if(!writer_suc){
                std::cout << test_def::get_captured_stdout() << std::endl;
            }
            ASSERT_TRUE(writer_suc);

            hdl_parser_verilog verilog_parser(parser_input);
            // Parse the .v file
            std::shared_ptr<netlist> parsed_nl = verilog_parser.parse(g_lib_name);

            if(parsed_nl == nullptr){
                std::cout << test_def::get_captured_stdout() << std::endl;
            }
            //ASSERT_NE(parsed_nl, nullptr);
            test_def::get_captured_stdout();

            std::cout << "\n---OUTPUT---\n" << parser_input.str() << "\n---OUTPUT---\n";
        }
    TEST_END
}*/



/*                                    net_0
 *                  .--= INV (0) =----.
 *  global_in       |                   '-=                     global_out
 *      ------------|                   .-= AND3 (2) = ----------
 *                  |                   | =
 *                  '--=                |
 *                       AND2 (1) =---'
 *                     =              net_1
 *                                                              global_inout
 *      -----------------------------------------------------------
 *
 */

/**
 * Testing the correct usage of the verilog parser by parse a small verilog-format string, which describes the netlist
 * shown above.
 *
 * Functions: parse
 */
TEST_F(hdl_parser_verilog_test, check_main_example)
{
    TEST_START
        { // NOTE: inout nets can't be handled
            std::stringstream input("module  (\n"
                                    "  global_in,\n"
                                    "  global_out \n"
                                    /*"  global_inout\n"*/
                                    " ) ;\n"
                                    "  input global_in ;\n"
                                    "  output global_out ;\n"
                                    "  wire global_inout ;\n"
                                    "  wire net_0 ;\n"
                                    "  wire net_1 ;\n"
                                    "INV gate_0 (\n"
                                    "  .\\I (global_in ),\n"
                                    "  .\\O (net_0 )\n"
                                    " ) ;\n"
                                    "AND2 gate_1 (\n"
                                    "  .\\I0 (global_in ),\n"
                                    "  .\\O (net_1 )\n"
                                    " ) ;\n"
                                    "AND3 gate_2 (\n"
                                    "  .\\I0 (net_0 ),\n"
                                    "  .\\I1 (net_1 ),\n"
                                    "  .\\O (global_out )\n"
                                    " ) ;\n"
                                    "endmodule");
            test_def::capture_stdout();
            hdl_parser_verilog verilog_parser(input);
            std::shared_ptr<netlist> nl = verilog_parser.parse(g_lib_name);
            if (nl == nullptr)
            {
                std::cout << test_def::get_captured_stdout();
            }
            else
            {
                test_def::get_captured_stdout();
            }

            ASSERT_NE(nl, nullptr);


            // Check if the gates are parsed correctly
            ASSERT_EQ(nl->get_gates("INV").size(), 1);
            std::shared_ptr<gate> gate_0 = *(nl->get_gates("INV").begin());
            ASSERT_EQ(nl->get_gates("AND2").size(), 1);
            std::shared_ptr<gate> gate_1 = *(nl->get_gates("AND2").begin());
            ASSERT_EQ(nl->get_gates("AND3").size(), 1);
            std::shared_ptr<gate> gate_2 = *(nl->get_gates("AND3").begin());

            ASSERT_NE(gate_0, nullptr);
            EXPECT_EQ(gate_0->get_name(), "gate_0");

            ASSERT_NE(gate_1, nullptr);
            EXPECT_EQ(gate_1->get_name(), "gate_1");

            ASSERT_NE(gate_2, nullptr);
            EXPECT_EQ(gate_2->get_name(), "gate_2");

            // Check if the nets are parsed correctly
            ASSERT_FALSE(nl->get_nets("net_0").empty());
            std::shared_ptr<net> net_0            = *(nl->get_nets("net_0").begin());
            ASSERT_FALSE(nl->get_nets("net_1").empty());
            std::shared_ptr<net> net_1            = *(nl->get_nets("net_1").begin());
            ASSERT_FALSE(nl->get_nets("global_in").empty());
            std::shared_ptr<net> net_global_in    = *(nl->get_nets("global_in").begin());
            ASSERT_FALSE(nl->get_nets("global_out").empty());
            std::shared_ptr<net> net_global_out   = *(nl->get_nets("global_out").begin());
            ASSERT_FALSE(nl->get_nets("global_inout").empty());
            std::shared_ptr<net> net_global_inout = *(nl->get_nets("global_inout").begin());

            ASSERT_NE(net_0, nullptr);
            EXPECT_EQ(net_0->get_name(), "net_0");
            EXPECT_EQ(net_0->get_src(), get_endpoint(gate_0, "O"));
            std::vector<endpoint> exp_net_0_dsts = {get_endpoint(gate_2, "I0")};
            EXPECT_TRUE(vectors_have_same_content(net_0->get_dsts(), std::vector<endpoint>({get_endpoint(gate_2, "I0")})));

            ASSERT_NE(net_1, nullptr);
            EXPECT_EQ(net_1->get_name(), "net_1");
            EXPECT_EQ(net_1->get_src(), get_endpoint(gate_1, "O"));
            EXPECT_TRUE(vectors_have_same_content(net_1->get_dsts(), std::vector<endpoint>({get_endpoint(gate_2, "I1")})));

            ASSERT_NE(net_global_in, nullptr);
            EXPECT_EQ(net_global_in->get_name(), "global_in");
            EXPECT_EQ(net_global_in->get_src(), get_endpoint(nullptr, ""));
            EXPECT_TRUE(vectors_have_same_content(net_global_in->get_dsts(), std::vector<endpoint>({get_endpoint(gate_0, "I"), get_endpoint(gate_1, "I0")})));
            EXPECT_TRUE(nl->is_global_input_net(net_global_in));

            ASSERT_NE(net_global_out, nullptr);
            EXPECT_EQ(net_global_out->get_name(), "global_out");
            EXPECT_EQ(net_global_out->get_src(), get_endpoint(gate_2, "O"));
            EXPECT_TRUE(net_global_out->get_dsts().empty());
            EXPECT_TRUE(nl->is_global_output_net(net_global_out));

            ASSERT_NE(net_global_inout, nullptr);
            EXPECT_EQ(net_global_inout->get_name(), "global_inout");
            EXPECT_EQ(net_global_inout->get_src(), get_endpoint(nullptr, ""));
            EXPECT_TRUE(net_global_inout->get_dsts().empty());
            //EXPECT_TRUE(nl->is_global_inout_net(net_global_inout));

            EXPECT_EQ(nl->get_global_input_nets().size(), 1);
            EXPECT_EQ(nl->get_global_output_nets().size(), 1);
            //EXPECT_EQ(nl->get_global_inout_nets().size(), 1);
        }
    TEST_END
}

/**
 * Testing the correct detection of single line comments (with '//') and comment blocks(with '/ *' and '* /')
 *
 * Functions: parse
 */
TEST_F(hdl_parser_verilog_test, check_comment_detection){
    TEST_START
        {
            // IN PROGRESS
        }
    TEST_END
}

/**
 * Testing the correct storage of data
 *
 * Functions: parse
 */
TEST_F(hdl_parser_verilog_test, check_generic_map){
    TEST_START
        {
            // IN PROGRESS
        }
    TEST_END
}

/**
 * Testing the correct usage of vector bounds
 *
 * Functions: parse
 */
TEST_F(hdl_parser_verilog_test, check_vector_bounds){
    TEST_START
        {
            // IN PROGRESS
        }
    TEST_END
}


/**
 * Testing the correct handling of invalid input
 *
 * Functions: parse
 */
TEST_F(hdl_parser_verilog_test, check_invalid_input)
{
    TEST_START
        {
            // IN PROGRESS
        }
    TEST_END
}




