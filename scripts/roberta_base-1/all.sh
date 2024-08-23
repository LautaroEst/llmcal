
echo "Starting roberta_base"
date

bash scripts/roberta_base-1/sst2.sh
bash scripts/roberta_base-1/agnews.sh
bash scripts/roberta_base-1/dbpedia.sh
bash scripts/roberta_base-1/20newsgroups.sh
bash scripts/roberta_base-1/banking77.sh

echo "roberta_base complete"
date