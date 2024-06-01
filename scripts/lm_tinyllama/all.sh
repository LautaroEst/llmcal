
echo "Starting TinyLlama"
date

bash scripts/lm_tinyllama/sst2.sh
bash scripts/lm_tinyllama/agnews.sh
bash scripts/lm_tinyllama/dbpedia.sh
bash scripts/lm_tinyllama/20newsgroups.sh
bash scripts/lm_tinyllama/banking77.sh

echo "TinyLlama complete"
date