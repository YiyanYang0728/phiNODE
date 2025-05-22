feature_col=$1

best_model=results/phiNODE_best_model.pth
res_dir=results
data_dir=data

outdir=perturb
mkdir -p ${outdir}

if [ -f "${outdir}/ft_${feature_col}_interact.tsv" ]; then
    exit 0
fi

awk -F"\t" -v feature_col=${feature_col} 'NR==(feature_col+1){print }' ${data_dir}/Phage_feature_names.txt

# Step.1 get feature 0 changed profile across samples (remaining samples are rescaled)
python scripts/perturb_feature.py ${data_dir}/Phage_train.tsv ${feature_col} mean ${outdir}

# Step.2 use phiNODE to predict new profile and stored as ${outdir}/ft_i_pred.tsv
if [ ! -f "${outdir}/bact_names.txt" ]; then
    cat ${data_dir}/Bact_arc_feature_names.txt > ${outdir}/bact_names.txt
fi
outdim=`awk 'END{print NR}' ${outdir}/bact_names.txt`

sed "s|#INFILE#|${outdir}/ft_${feature_col}_profile.tsv|g;s|#PREFIX#|${outdir}/ft_${feature_col}|g;s|#BESTMODEL#|${best_model}|g;s|#OUTDIM#|${outdim}|g" config/interaction.tmp.yaml > ${outdir}/config.${feature_col}.yaml
# add best parameters to config file
awk '{print "  "$0}' ${res_dir}/phiNODE_best_params.yaml >> ${outdir}/config.${feature_col}.yaml
python model/predict.py -c ${outdir}/config.${feature_col}.yaml

# Step.3 get diff/delta between predicted profile and original profile for each target feature
# 1) original profile; 2) perturbed profile
python scripts/profile_diff.py ${res_dir}/phiNODE_predict_train_pred.tsv ${outdir}/ft_${feature_col}_pred.tsv ${outdir}/ft_${feature_col}_pred_diff.tsv

# Step.5 calculate Sji btw feature i and all predicted features (all j)
python scripts/infer_interaction.py ${outdir}/ft_${feature_col}_pred_diff.tsv ${outdir}/ft_${feature_col}_delta.tsv ${outdir}/ft_${feature_col}_interact.tsv

# Step.6 clean outdir
cd ${outdir}
rm -f config.${feature_col}.yaml ft_${feature_col}_pred.tsv ft_${feature_col}_profile.tsv ft_${feature_col}_no_clr_pred.tsv ft_${feature_col}_pred_diff.tsv ft_${feature_col}_delta.tsv
cd ..
