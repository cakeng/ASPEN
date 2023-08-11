bandwidth=$1
echo "//////////    Set TC in EDGE    //////////"
tc_reset_cmd_wo_ssh="echo bestnxcl | sudo -S tc qdisc del dev wlan0 root"
tc_reset_cmd="$tc_reset_cmd_wo_ssh"
echo "     $tc_reset_cmd"
eval $tc_reset_cmd
tc_set_cmd_wo_ssh="echo bestnxcl | sudo -S tc qdisc add dev wlan0 root handle 1: htb default 6"
tc_set_cmd="$tc_set_cmd_wo_ssh"
echo "     $tc_set_cmd"
eval $tc_set_cmd
tc_set_bw_cmd_wo_ssh="echo bestnxcl | sudo -S tc class add dev wlan0 parent 1: classid 1:6 htb rate ${bandwidth}mbit"
tc_set_bw_cmd="$tc_set_bw_cmd_wo_ssh"
echo "     $tc_set_bw_cmd"
eval $tc_set_bw_cmd


