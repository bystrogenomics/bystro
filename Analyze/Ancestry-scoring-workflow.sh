#Workflow for Admixture Ancestry Scoring

#Testing admixture locally with Hapmap data in this path
export PATH=$PATH:/Applications/GenomicsTools
admixture hapmap3.bed 3

# Verify the two datasets have the same set of SNPs
diff -s 1kGPreference.bim study.bim
# Run unsupervised ADMIXTURE with K=2
admixture 1kGPreference.bed 2
# Use learned allele frequencies as (fixed) input to next step
cp 1kGPreference.2.P study.2.P.in
# Run projection ADMIXTURE with K=2
admixture -P study.bed 2

#Using 1kgenomes as ref - http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
#Before using ref, converted vcf to plink2 files and merge

#common snps only for PCA using downloaded bystro file (maf>.05)
#cut -f1,2,5,65,66 Annotation/SamplePopcommonsnps.annotation.tsv > Annotation/PartialCommon.txt
#python ConvertChrPosToRsCommonsnps.py Annotation/PartialCommon.txt Annotation/CommonWchrRs.txt
#cut -f4 Annotation/CommonWchrRs.txt > $SnpPath/commonSnpListforgwas.txt
#need to clean plink file before running pca, then extract common snps
#plink2 --bfile $SnpPath/CleanPlinkFile --extract $SnpPath/commonSnpListforgwas.txt --allow-extra-chr --make-bed --out $SnpPath/FileForpca

# Identify LD between SNPs
plink2 --bfile 1kGPallcommsnps --indep-pairwise 50 5 0.9 --out allcleanLD
# LD pruning
plink2 --bfile 1kGPallcommsnps --extract allcleanLD.prune.out --make-bed -out pruned
#There's still too many snps so let's try to adjust the parameters - snp window to 500
./plink2 --bfile 1kGPallcommsnps --indep-pairwise 2 2 0.95 --out allcleanLD
# LD pruning
./plink2 --bfile 1kGPallcommsnps --extract allcleanLD.prune.out --make-bed -out pruned500ksnps

#Admixture with 1kGP
#Figure out how many pops and then only keep IDs w/o multiple pops
python AddAncstTo1kGPfam.py igsr_IDwAncst.txt pruned500ksnps.fam prunedsetwBothAnct.txt
cut -f7 prunedsetwBothAnct.txt > PopsOnly.txt
sort -u PopsOnly.txt > AllPops.txt
cut -f8 prunedsetwBothAnct.txt > BothPopsOnly.txt
sort -u BothPopsOnly.txt > All2ryPops.txt

plink2 --bfile pruned 
#For Superpop code and not counting admixed samples, it's 5
#Without LD pruning (now running on linux version)
./admixture  1kGPallcommsnps.bed 5 -j4 
#It ran out of memory, so we're pruning down to 500k snvs
#With LD pruning
./admixture pruned500ksnps.bed 5 -j4
#PCA 
plink2 --bfile pruned --pca 5
python AddAncstTo1kGPfam.py igsr_IDwAncst.txt plink2.eigenvec 1kGPprunedwPCs.txt
