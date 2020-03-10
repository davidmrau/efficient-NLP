cat 2019qrels-pass_query_ids.txt | xargs -I{} awk '$1 == "{}"' msmarco-passagetest2019-top1000.tsv >> msmarco-passagetest2019-top1000_43.tsv
awk '{print $1}' 2019qrels-pass.txt >> 2019qrels-pass_query_ids.txt
