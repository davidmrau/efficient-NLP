chmod u+x telegram.sh
wget https://rijsbergen.hum.uva.nl/david/hosted/robust04_AOL_anserini_top_100_qld_no_stem.gz

gunzip robust04_AOL_anserini_top_100_qld_no_stem.gz

mkdir -p data/robust04/

mv robust04_AOL_anserini_top_100_qld_no_stem data/robust04/

pip3 install -r requirements.txt

python3 util_scripts/offset_dict_anserini.py --fname data/robust04/robust04_AOL_anserini_top_100_qld_no_stem
 
python3 generate_triplets_from_weak.py --sampler uniform --min_hits 10 --weak_results_file data/robust04/robust04_AOL_anserini_top_100_qld_no_stem --target binary --top_k_per_query 100


bash telegram.sh -c -462467791 "generated triples"

git clone https://github.com/alexandres/terashuf.git

cd terashuf/
make
cd ..
./terashuf/terashuf <  data/robust04/robust04_AOL_anserini_top_100_qld_no_stem_TRIPLETS_100_train >  data/robust04/robust04_AOL_anserini_top_100_qld_no_stem_TRIPLETS_100_train_shuf 


./terashuf/terashuf <  data/robust04/robust04_AOL_anserini_top_100_qld_no_stem_TRIPLETS_100_val >  data/robust04/robust04_AOL_anserini_top_100_qld_no_stem_TRIPLETS_100_val_shuf 


bash telegram.sh -c -462467791 "triples shuffled"
