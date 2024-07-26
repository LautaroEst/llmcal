
echo "Starting phi3"
date

bash scripts/lm_phi3-4/sst2.sh
bash scripts/lm_phi3-4/agnews.sh
bash scripts/lm_phi3-4/dbpedia.sh
bash scripts/lm_phi3-4/20newsgroups.sh
bash scripts/lm_phi3-4/banking77.sh

echo "phi3 complete"
date