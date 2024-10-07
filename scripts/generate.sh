num_topics=20
metric="integrityImpact"
if [ $# -gt 0 ]; then
    num_topics=$1
    echo $num_topics
    sed -i -E 's/(num_topics = )([0-9]+)/\1'"$num_topics"'/' config.py
fi


python grid_search_lda_compare.py && \
python ./parse_topics.py && \
python ./aggregate_classes.py 
