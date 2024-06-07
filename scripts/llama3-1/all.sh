
echo "Starting TinyLlama"
date

bash scripts/llama3-1/sst2.sh
bash scripts/llama3-1/agnews.sh
bash scripts/llama3-1/dbpedia.sh
bash scripts/llama3-1/20newsgroups.sh
bash scripts/llama3-1/banking77.sh

echo "TinyLlama complete"
date