
echo "Starting llama3"
date

bash scripts/lm_tinyllama-5/sst2.sh
# bash scripts/lm_tinyllama-5/agnews.sh
bash scripts/lm_tinyllama-5/dbpedia.sh
bash scripts/lm_tinyllama-5/20newsgroups.sh
bash scripts/lm_tinyllama-5/banking77.sh

echo "llama3 complete"
date