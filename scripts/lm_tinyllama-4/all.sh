
echo "Starting llama3"
date

bash scripts/lm_tinyllama-4/sst2.sh
# bash scripts/lm_tinyllama-4/agnews.sh
bash scripts/lm_tinyllama-4/dbpedia.sh
bash scripts/lm_tinyllama-4/20newsgroups.sh
bash scripts/lm_tinyllama-4/banking77.sh

echo "llama3 complete"
date