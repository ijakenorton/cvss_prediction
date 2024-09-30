
# if [ $# -gt 0 ]; then
#     metric=$1
# fi

    
for metric in privilegesRequired userInteraction confidentialityImpact integrityImpact availabilityImpact; do
    sed -i -E 's/(current_metric: Metrics_t =)(.*$)/\1 MetricNames.'"$metric"'/' config.py
    echo current metric $metric
    for i in 14 16 18 20 50 75 100; do 
        ./generate.sh $i 
    done
done
