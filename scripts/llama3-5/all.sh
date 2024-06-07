
echo "Starting TinyLlama"
date

bash scripts/llama3-5/sst2.sh
bash scripts/llama3-5/agnews.sh
bash scripts/llama3-5/dbpedia.sh
bash scripts/llama3-5/20newsgroups.sh
bash scripts/llama3-5/banking77.sh

echo "TinyLlama complete"
date