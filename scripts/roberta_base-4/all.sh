
echo "Starting roberta_base"
date

bash scripts/roberta_base-4/sst2.sh
bash scripts/roberta_base-4/agnews.sh
bash scripts/roberta_base-4/dbpedia.sh
bash scripts/roberta_base-4/20newsgroups.sh
bash scripts/roberta_base-4/banking77.sh

echo "roberta_base complete"
date