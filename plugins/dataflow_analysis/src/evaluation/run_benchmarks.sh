#!/bin/bash 

OUTPUT_PATH=/home/osboxes/Desktop/dataflow_out
OUT_LOG_PATH=/home/osboxes/hal/plugins/dataflow_analysis/src/evaluation/hal_output_log.txt
BENCHMARKS_DIR=/home/osboxes/Downloads/

echo "################################################################" >> $OUT_LOG_PATH
echo $(date -u) >> $OUT_LOG_PATH

# bencharks with sizes:
echo "with sizes:" >> $OUT_LOG_PATH
echo "present results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/present_lsi_10k_synopsys.v  --dataflow --sizes "80,64"  --path $OUTPUT_PATH

echo "open8 results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/open8_lsi_10k_synopsys.v  --dataflow --sizes "16,8"  --path $OUTPUT_PATH

echo "DES results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/des_lsi_10k_synopsys.v  --dataflow --sizes "64,56"  --path $OUTPUT_PATH

echo "SHA-3 results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/sha-3_lsi_10k_synopsys.v  --dataflow --sizes "1600,1088,512"  --path $OUTPUT_PATH

# echo "ibex results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/ibex_lsi_10k_synopsys.v  --dataflow --sizes "33,32,31"  --path $OUTPUT_PATH

# echo "edge results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/edge_lsi_10k_synopsys.v  --dataflow --sizes "33,32,31"  --path $OUTPUT_PATH

# echo "tiny AES results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/tiny_aes_hsing_10k_synopsys.v  --dataflow --sizes "128"  --path $OUTPUT_PATH

# echo "RSA results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/rsa_lsi_10k_synopsys.v  --dataflow --sizes "512"  --path $OUTPUT_PATH


# bencharks without sizes:
echo "without sizes:" >> $OUT_LOG_PATH
echo "present results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/present_lsi_10k_synopsys.v  --dataflow --path $OUTPUT_PATH

echo "open8 results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/open8_lsi_10k_synopsys.v  --dataflow   --path $OUTPUT_PATH

echo "DES results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/des_lsi_10k_synopsys.v  --dataflow  --path $OUTPUT_PATH

echo "SHA-3 results:" >> $OUT_LOG_PATH
hal -i $BENCHMARKS_DIR/sha-3_lsi_10k_synopsys.v  --dataflow  --path $OUTPUT_PATH

# echo "ibex results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/ibex_lsi_10k_synopsys.v  --dataflow   --path $OUTPUT_PATH

# echo "edge results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/edge_lsi_10k_synopsys.v  --dataflow  --path $OUTPUT_PATH

# echo "tiny AES results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/tiny_aes_hsing_lsi_10k_synopsys.v  --dataflow  --path $OUTPUT_PATH

# echo "RSA results:" >> $OUT_LOG_PATH
# hal -i $BENCHMARKS_DIR/rsa_lsi_10k_synopsys.v  --dataflow --path $OUTPUT_PATH

echo "################################################################" >> $OUT_LOG_PATH