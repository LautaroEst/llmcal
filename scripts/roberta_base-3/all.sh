
echo "Starting roberta_base"
date

bash scripts/roberta_base-3/sst2.sh
bash scripts/roberta_base-3/agnews.sh
bash scripts/roberta_base-3/dbpedia.sh
bash scripts/roberta_base-3/20newsgroups.sh
bash scripts/roberta_base-3/banking77.sh

echo "roberta_base complete"
date