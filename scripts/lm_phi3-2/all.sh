
echo "Starting phi3"
date

bash scripts/lm_phi3-2/sst2.sh
bash scripts/lm_phi3-2/agnews.sh
bash scripts/lm_phi3-2/dbpedia.sh
# bash scripts/lm_phi3-2/20newsgroups.sh
# bash scripts/lm_phi3-2/banking77.sh

echo "phi3 complete"
date