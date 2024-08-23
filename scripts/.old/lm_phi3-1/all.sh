
echo "Starting phi3"
date

bash scripts/lm_phi3-1/sst2.sh
bash scripts/lm_phi3-1/agnews.sh
bash scripts/lm_phi3-1/dbpedia.sh
# bash scripts/lm_phi3-1/20newsgroups.sh
# bash scripts/lm_phi3-1/banking77.sh

echo "phi3 complete"
date