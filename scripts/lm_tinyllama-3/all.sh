
echo "Starting llama3"
date

bash scripts/lm_tinyllama-3/sst2.sh
# bash scripts/lm_tinyllama-3/agnews.sh
bash scripts/lm_tinyllama-3/dbpedia.sh
bash scripts/lm_tinyllama-3/20newsgroups.sh
bash scripts/lm_tinyllama-3/banking77.sh

echo "llama3 complete"
date