
echo "Starting roberta_base"
date

bash scripts/roberta_base-5/sst2.sh
bash scripts/roberta_base-5/agnews.sh
bash scripts/roberta_base-5/dbpedia.sh
bash scripts/roberta_base-5/20newsgroups.sh
bash scripts/roberta_base-5/banking77.sh

echo "roberta_base complete"
date