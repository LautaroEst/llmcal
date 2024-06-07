
echo "Starting TinyLlama"
date

bash scripts/llama3-4/sst2.sh
bash scripts/llama3-4/agnews.sh
bash scripts/llama3-4/dbpedia.sh
bash scripts/llama3-4/20newsgroups.sh
bash scripts/llama3-4/banking77.sh

echo "TinyLlama complete"
date