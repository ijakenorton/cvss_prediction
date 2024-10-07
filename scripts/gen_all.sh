
# if [ $# -gt 0 ]; then
#     metric=$1
# fi

    
for metric in integrityImpact availabilityImpact; do
    sed -i -E 's/(current_metric: Metrics_t =)(.*$)/\1 MetricNames.'"$metric"'/' config.py
    echo current metric $metric
    for i in 20 40 60 80 100; do 
        ./generate.sh $i 
    done
done
