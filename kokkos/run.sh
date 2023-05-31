#!/bin/bash

# Set the number of iterations
ITERATIONS=10

M=$2
N=$3
K=$4

# first compile
make build KOKKOS_DEVICES=$1
EXE=""
if [ "$1" = "OpenMP" ]; then
   EXE="./main.host"
elif [ "$1" = "Cuda" ]; then
   EXE="./main.cuda"
fi

kernel_times=()
flops=()
times=()

# Loop for the specified number of iterations
for ((i = 1; i <= $ITERATIONS; i++)); do
   echo "Running iteration $i"
   output=$(${EXE} ${M} ${N} ${K})
   kernel_time=$(echo "$output" | grep "Kernel" | awk '{print $2}')
   f=$(echo "$output" | grep "FLOPS" | awk '{print $2}')
   time=$(echo "$output" | grep "Time" | awk '{print $2}')
   echo "$kernel_time $f $time"
   kernel_times+=($kernel_time)
   flops+=($f)
   times+=($time)
done

kernel_time_sum=0
flops_sum=0
time_sum=0

for kernel_time in "${kernel_times[@]}"; do
   kernel_time_sum=$(awk "BEGIN {printf \"%.10f\", $kernel_time_sum + $kernel_time; exit}")
done

for f in "${flops[@]}"; do
   flops_sum=$(awk "BEGIN {printf \"%.10f\", $flops_sum + $f; exit}")
done

for time in "${times[@]}"; do
   time_sum=$(awk "BEGIN {printf \"%.10f\", $time_sum + $time; exit}")
done

kernel_time_avg=$(awk "BEGIN {printf \"%.10f\", $kernel_time_sum / ${#kernel_times[@]}; exit}")
flops_avg=$(awk "BEGIN {printf \"%e\", $flops_sum / ${#flops[@]}; exit}")
time_avg=$(awk "BEGIN {printf \"%.10f\", $time_sum / ${#times[@]}; exit}")

echo "Average Kernel Time: ${kernel_time_avg}s"
echo "Average FLOPS: ${flops_avg}"
echo "Average Time: ${time_avg}s"
