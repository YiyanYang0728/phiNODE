N_train=$1
phage_ct=$2
prok_ct=${3:-2000} # the maximal number of prokaryotic features. This number can be larger than the real number.
res_dir=${4:-results} # output folder

seq 0 $(($phage_ct-1)) |parallel -k "/bin/bash scripts/ft_col_pred_inter.sh {} perturb data ${N_train} ${prok_ct}"
cat perturb/*_S_ij.norm.tsv > ${res_dir}/predicted_interactions.tsv
rm -rf perturb/*_S_ij.tmp
rm -rf perturb/*_S_ij.norm.tsv
