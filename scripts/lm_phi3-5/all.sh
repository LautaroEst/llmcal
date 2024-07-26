
echo "Starting phi3"
date

bash scripts/lm_phi3-5/sst2.sh
bash scripts/lm_phi3-5/agnews.sh
bash scripts/lm_phi3-5/dbpedia.sh
bash scripts/lm_phi3-5/20newsgroups.sh
bash scripts/lm_phi3-5/banking77.sh

echo "phi3 complete"
date