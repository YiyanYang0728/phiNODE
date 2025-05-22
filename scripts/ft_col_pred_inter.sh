ft_ct=$1
fld=$2
data_dir=$3
N_train=$4
rank_perc=${5:-1}

outdim=`awk 'END{print NR}' ${data_dir}/Bact_arc_feature_names.txt`
d=$(($ft_ct+1))
phage=`sed -n ${d}p ${data_dir}/Phage_feature_names.txt`

# get original data
paste ${data_dir}/Bact_arc_feature_names.txt <(datamash sum 1-${outdim} < ${fld}/ft_${ft_ct}_interact.tsv|datamash transpose) | awk -F"\t" 'BEGIN{OFS="\t"}{print $1,$2,$2}' | sort -t$'\t' -k2,2gr > ${fld}/${ft_ct}_bact_names_interact.tsv

if (( $(echo "$rank_perc < 1" | bc -l) ))
then
    # NR represents the rank
    awk -F"\t" -v phage="${phage}" -v th="${rank_perc}" -v ct="${outdim}" 'BEGIN{cutoff=int(th*ct+0.5)} NR<=cutoff {print phage"\t"$1"\t"$2"\t"$3"\t"NR"\t"cutoff"\t"ct}' ${fld}/${ft_ct}_bact_names_interact.tsv | awk -v N_train=${N_train} '{print $1"\t"$2"\t"$3/N_train"\t"$3"\t"$4}' > ${fld}/${ft_ct}_S_ij.tmp 
else
    awk -F"\t" -v phage="${phage}" -v th="${rank_perc}" -v ct="${outdim}" 'BEGIN{cutoff=th} NR<=cutoff {print phage"\t"$1"\t"$2"\t"$3"\t"NR"\t"cutoff"\t"ct}' ${fld}/${ft_ct}_bact_names_interact.tsv | awk -v N_train=${N_train} '{print $1"\t"$2"\t"$3/N_train"\t"$3"\t"$4}' > ${fld}/${ft_ct}_S_ij.tmp

# normalize
m=`datamash mean 3 < ${fld}/${ft_ct}_S_ij.tmp`
s=`datamash sstdev 3 < ${fld}/${ft_ct}_S_ij.tmp`
awk -F"\t" -v m="${m}" -v s="${s}" '{print $1"\t"$2"\t"(($3-m)/s)}' ${fld}/${ft_ct}_S_ij.tmp > ${fld}/${ft_ct}_S_ij.norm.tsv

rm -f ${fld}/${ft_ct}_bact_names_interact.tsv
fi