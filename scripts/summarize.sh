prefix=$1

pcc_ft_file=${prefix}_pcc_ft.tsv
cos_sim_ft_file=${prefix}_cos_sim_ft.tsv
r2_ft_file=${prefix}_r2_ft.tsv

pcc_file=${prefix}_pcc.tsv
cos_sim_file=${prefix}_cos_sim.tsv
r2_file=${prefix}_r2.tsv

echo "#############feature-wise metrics#############"

avg_pcc=`datamash --headers mean 1 < <(grep -v "\"\"" $pcc_ft_file) | tail -n+2`
echo -e "Mean PCC: ${avg_pcc}"

ct=`awk 'NR>1 && $1>0.8' <(grep -v "\"\"" $pcc_ft_file) | wc -l`
echo -e "No. of pcc > 0.8: $ct"

all_ct=`tail -n+2 <(grep -v "\"\"" $pcc_ft_file) | wc -l`
perc=`echo "($ct/$all_ct)*100"|bc -l`
echo -e "Perc of pcc > 0.8: $perc"

top50_pcc=`tail -n+2 <(grep -v "\"\"" $pcc_ft_file)|sort -t$'\t' -k1,1gr|head -n50|datamash mean 1`
echo -e "Top 50 pcc mean: $top50_pcc"

top20_pcc=`tail -n+2 <(grep -v "\"\"" $pcc_ft_file)|sort -t$'\t' -k1,1gr|head -n20|datamash mean 1`
echo -e "Top 20 pcc mean: $top20_pcc"

top10_pcc=`tail -n+2 <(grep -v "\"\"" $pcc_ft_file)|sort -t$'\t' -k1,1gr|head -n10|datamash mean 1`
echo -e "Top 10 pcc mean: $top10_pcc"

echo "-----------------------------------"
avg_cos_sim=`datamash --headers mean 1 < <(grep -v "\"\"" $cos_sim_ft_file) | tail -n+2`
echo -e "Mean Cos Sim: ${avg_cos_sim}"

ct=`awk 'NR>1 && $1>0.8' <(grep -v "\"\"" $cos_sim_ft_file) | wc -l`
echo -e "No. of cos_sim > 0.8: $ct"

all_ct=`tail -n+2 <(grep -v "\"\"" $cos_sim_ft_file) | wc -l`
perc=`echo "($ct/$all_ct)*100"|bc -l`
echo -e "Perc of cos_sim > 0.8: $perc"

top50=`tail -n+2 <(grep -v "\"\"" $cos_sim_ft_file)|sort -t$'\t' -k1,1gr|head -n50|datamash mean 1`
echo -e "Top 50 cos_sim mean: $top50"

top20=`tail -n+2 <(grep -v "\"\"" $cos_sim_ft_file)|sort -t$'\t' -k1,1gr|head -n20|datamash mean 1`
echo -e "Top 20 cos_sim mean: $top20"

top10=`tail -n+2 <(grep -v "\"\"" $cos_sim_ft_file)|sort -t$'\t' -k1,1gr|head -n10|datamash mean 1`
echo -e "Top 10 cos_sim mean: $top10"

echo "-----------------------------------"
avg_r2=`datamash --headers mean 1 < <(grep -v "\"\"" $r2_ft_file) | tail -n+2`
echo -e "Mean R2: ${avg_r2}"

ct=`awk 'NR>1 && $1>0.6' <(grep -v "\"\"" $r2_ft_file) | wc -l`
echo -e "No. of R2 > 0.6: $ct"

all_ct=`tail -n+2 <(grep -v "\"\"" $r2_ft_file) | wc -l`
perc=`echo "($ct/$all_ct)*100"|bc -l`
echo -e "Perc of R2 > 0.6: $perc"

top50=`tail -n+2 <(grep -v "\"\"" $r2_ft_file)|sort -t$'\t' -k1,1gr|head -n50|datamash mean 1`
echo -e "Top 50 R2 mean: $top50"

top20=`tail -n+2 <(grep -v "\"\"" $r2_ft_file)|sort -t$'\t' -k1,1gr|head -n20|datamash mean 1`
echo -e "Top 20 R2 mean: $top20"

top10=`tail -n+2 <(grep -v "\"\"" $r2_ft_file)|sort -t$'\t' -k1,1gr|head -n10|datamash mean 1`
echo -e "Top 10 R2 mean: $top10"

echo ""
echo "#############sample-wise metrics#############"

avg_pcc=`datamash --headers mean 1 < <(grep -v "\"\"" $pcc_file) | tail -n+2`
echo -e "Mean PCC: ${avg_pcc}"

ct=`awk 'NR>1 && $1>0.8' <(grep -v "\"\"" $pcc_file) | wc -l`
echo -e "No. of pcc > 0.8: $ct"

all_ct=`tail -n+2 <(grep -v "\"\"" $pcc_file) | wc -l`
perc=`echo "($ct/$all_ct)*100"|bc -l`
echo -e "Perc of pcc > 0.8: $perc"

top50_pcc=`tail -n+2 <(grep -v "\"\"" $pcc_file)|sort -t$'\t' -k1,1gr|head -n50|datamash mean 1`
echo -e "Top 50 pcc mean: $top50_pcc"

top20_pcc=`tail -n+2 <(grep -v "\"\"" $pcc_file)|sort -t$'\t' -k1,1gr|head -n20|datamash mean 1`
echo -e "Top 20 pcc mean: $top20_pcc"

top10_pcc=`tail -n+2 <(grep -v "\"\"" $pcc_file)|sort -t$'\t' -k1,1gr|head -n10|datamash mean 1`
echo -e "Top 10 pcc mean: $top10_pcc"

echo "-----------------------------------"
avg_cos_sim=`datamash --headers mean 1 < <(grep -v "\"\"" $cos_sim_file) | tail -n+2`
echo -e "Mean Cos Sim: ${avg_cos_sim}"

ct=`awk 'NR>1 && $1>0.8' <(grep -v "\"\"" $cos_sim_file) | wc -l`
echo -e "No. of cos_sim > 0.8: $ct"

all_ct=`tail -n+2 <(grep -v "\"\"" $cos_sim_file) | wc -l`
perc=`echo "($ct/$all_ct)*100"|bc -l`
echo -e "Perc of cos_sim > 0.8: $perc"

top50=`tail -n+2 <(grep -v "\"\"" $cos_sim_file)|sort -t$'\t' -k1,1gr|head -n50|datamash mean 1`
echo -e "Top 50 cos_sim mean: $top50"

top20=`tail -n+2 <(grep -v "\"\"" $cos_sim_file)|sort -t$'\t' -k1,1gr|head -n20|datamash mean 1`
echo -e "Top 20 cos_sim mean: $top20"

top10=`tail -n+2 <(grep -v "\"\"" $cos_sim_file)|sort -t$'\t' -k1,1gr|head -n10|datamash mean 1`
echo -e "Top 10 cos_sim mean: $top10"

echo "-----------------------------------"
avg_r2=`datamash --headers mean 1 < <(grep -v "\"\"" $r2_file) | tail -n+2`
echo -e "Mean R2: ${avg_r2}"

ct=`awk 'NR>1 && $1>0.6' <(grep -v "\"\"" $r2_file) | wc -l`
echo -e "No. of R2 > 0.6: $ct"

all_ct=`tail -n+2 <(grep -v "\"\"" $r2_file) | wc -l`
perc=`echo "($ct/$all_ct)*100"|bc -l`
echo -e "Perc of R2 > 0.6: $perc"

top50=`tail -n+2 <(grep -v "\"\"" $r2_file)|sort -t$'\t' -k1,1gr|head -n50|datamash mean 1`
echo -e "Top 50 R2 mean: $top50"

top20=`tail -n+2 <(grep -v "\"\"" $r2_file)|sort -t$'\t' -k1,1gr|head -n20|datamash mean 1`
echo -e "Top 20 R2 mean: $top20"

top10=`tail -n+2 <(grep -v "\"\"" $r2_file)|sort -t$'\t' -k1,1gr|head -n10|datamash mean 1`
echo -e "Top 10 R2 mean: $top10"
