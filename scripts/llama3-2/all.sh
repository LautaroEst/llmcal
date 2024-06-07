
echo "Starting TinyLlama"
date

bash scripts/llama3-2/sst2.sh
bash scripts/llama3-2/agnews.sh
bash scripts/llama3-2/dbpedia.sh
bash scripts/llama3-2/20newsgroups.sh
bash scripts/llama3-2/banking77.sh

echo "TinyLlama complete"
date