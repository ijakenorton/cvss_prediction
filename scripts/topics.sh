for num_topics in 2 4 6 8 10
do
    cat ./lda_word2vec_desc_compare_output/lda_*.txt | grep "Num Topics: $num_topics" | head -n 1
done
