DATA_PATH=../data/msmarco/
## getting only docs for judged queries

# get only judged docs 
awk '{print $1}' ${DATA_PATH}/2019qrels-pass.txt | uniq| xargs -I{} awk '$1 == "{}"' ${DATA_PATH}/msmarco-passagetest2019-top1000.tsv  > ${DATA_PATH}/msmarco-passagetest2019-top1000_43.tsv
#  extract doc_id and doc text, sort docs and make unique

awk -F'\t' '{print $2"\t"$4}' ${DATA_PATH}/msmarco-passagetest2019-top1000_43.tsv | sort -r -k 1 | uniq > /tmp/tmp.tsv
mv /tmp/tmp.tsv ${DATA_PATH}/msmarco-passagetest2019-top1000_43.tsv 
echo "Tokenizing"
# tokenize 
python3 ../tokenizer.py --folder ${DATA_PATH} --fname msmarco-passagetest2019-top1000_43.tsv --max_len 150
echo "Filter qrels for docs within top1000"
# filter qrels for docs that are within the top1000

awk '{print $1}' ${DATA_PATH}/msmarco-passagetest2019-top1000_43.tokenized.tsv  | xargs -I{} awk '"{}" == $3 {print $0}' ${DATA_PATH}/2019qrels-pass.txt > ${DATA_PATH}/2019qrels-pass_filtered.txt

## qrels in ms_marco_eval.py format 
echo "Convert qrel to MS-Marco"
awk '$4 > 1 {print $1"\t0\t"$3"\t"$4}' ${DATA_PATH}/2019qrels-pass_filtered.txt > ${DATA_PATH}/2019qrels-pass_filtered_ms_marco.txt

# get juged queries
awk '{print $1}' ${DATA_PATH}/2019qrels-pass.txt | uniq | xargs -I{} awk '$1 == "{}"' ${DATA_PATH}/msmarco-test2019-queries.tsv >> ${DATA_PATH}/msmarco-test2019-queries_43.tsv


python3 ../tokenizer.py --folder ${DATA_PATH} --fname msmarco-test2019-queries_43.tsv
