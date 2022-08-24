#Workflow for Admixture Ancestry Scoring

#Testing admixture locally with Hapmap data in this path
export PATH=$PATH:/Applications/GenomicsTools
admixture hapmap3.bed 3

#common snps only for PCA using downloaded bystro file (maf>.05)
cut -f1,2,5,65,66 Annotation/SamplePopcommonsnps.annotation.tsv > Annotation/PartialCommon.txt
python ConvertChrPosToRsCommonsnps.py Annotation/PartialCommon.txt Annotation/CommonWchrRs.txt
cut -f4 Annotation/CommonWchrRs.txt > $SnpPath/commonSnpListforgwas.txt
#need to clean plink file before running pca, then extract common snps
plink2 --bfile $SnpPath/CleanPlinkFile --extract $SnpPath/commonSnpListforgwas.txt --allow-extra-chr --make-bed --out $SnpPath/FileForpca
# Identify LD between SNPs
plink2 --bfile $SnpPath/FileForpca --indep-pairwise 50 5 0.2 --allow-extra-chr --out $SnpPath/allcleanLD
# LD pruning
plink2 --bfile $SnpPath/FileForpca --extract $SnpPath/allcleanLD.prune.out --allow-extra-chr --make-bed -out $SnpPath/pruned
###PCA 
plink2 --bfile $SnpPath/pruned --allow-extra-chr --pca 10
