
## getting only docs for judged queries

# get query ids
awk '{print $1}' 2019qrels-pass.txt >> 2019qrels-pass_query_ids.txt
# get only judged docs 
cat 2019qrels-pass_query_ids.txt | xargs -I{} awk '$1 == "{}"' msmarco-passagetest2019-top1000.tsv >> msmarco-passagetest2019-top1000_43.tsv

# sort docs and make unique
sort -r -k 1 msmarco-passagetest2019-top1000_43.tsv.d_id_doc.tokenized.tsv | uniq >> msmarco-passagetest2019-top1000_43.tsv.d_id_doc.tokenized_uniq.tsv




## qrels in ms_marco_eval.py format 
awk '$4 > 1 {print $1"\t0\t"$3"\t"$4}' 2019qrels-pass.txt >> 2019qrels-pass_ms_marco.txt
# filter qrels for docs that are within the top1000

awk '{print $1}' msmarco-passagetest2019-top1000_43.tsv.d_id_doc.tokenized_uniq.tsv  | xargs -I{} awk '"{}" == $3 {print $0}' 2019qrels-pass.txt >> 2019qrels-pass_filtered.txt

