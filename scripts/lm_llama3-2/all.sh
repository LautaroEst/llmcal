
echo "Starting llama3"
date

# bash scripts/lm_llama3-2/sst2.sh
# bash scripts/lm_llama3-2/agnews.sh
# bash scripts/lm_llama3-2/dbpedia.sh
bash scripts/lm_llama3-2/20newsgroups.sh
bash scripts/lm_llama3-2/banking77.sh

echo "llama3 complete"
date