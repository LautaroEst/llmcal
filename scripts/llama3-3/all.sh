
echo "Starting TinyLlama"
date

bash scripts/llama3-3/sst2.sh
bash scripts/llama3-3/agnews.sh
bash scripts/llama3-3/dbpedia.sh
bash scripts/llama3-3/20newsgroups.sh
bash scripts/llama3-3/banking77.sh

echo "TinyLlama complete"
date