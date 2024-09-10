
echo "Starting roberta_base"
date

bash scripts/roberta_base-2/sst2.sh
bash scripts/roberta_base-2/agnews.sh
bash scripts/roberta_base-2/dbpedia.sh
bash scripts/roberta_base-2/20newsgroups.sh
bash scripts/roberta_base-2/banking77.sh

echo "roberta_base complete"
date