#Take the first argument as the output list name, and append the remaining arguments to the list
if [ "$#" -lt 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./make_list.sh <output_list_name> <output1> <output2> ..."
    exit 1
fi
output_list_name=$1
shift
output_list=()
while [ "$#" -gt 0 ]; do
    output_list+=("$1")
    shift
done
echo -n "" > $output_list_name
for i in "${output_list[@]}"; do
    echo "$i" >> $output_list_name
done
