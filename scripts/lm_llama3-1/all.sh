
echo "Starting llama3"
date

# bash scripts/lm_llama3-1/sst2.sh
# bash scripts/lm_llama3-1/agnews.sh
# bash scripts/lm_llama3-1/dbpedia.sh
bash scripts/lm_llama3-1/20newsgroups.sh
bash scripts/lm_llama3-1/banking77.sh

echo "llama3 complete"
date