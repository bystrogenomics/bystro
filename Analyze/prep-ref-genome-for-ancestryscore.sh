#Convert vcf to plink for reference 1000genomes data

#Starting with not the biggest one in case this isn't the result we're looking for
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr13.filtered.SNV_INDEL_SV_phased_panel.vcf --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPchr13
#This made new plink file types: psam, pgen, pvar but we need bed/bim/fam for admixture
plink2 --pfile 1kGPchr13 --make-bed --out 1kGPchr13oldplink
#Only need common variants
plink2 --bfile 1kGPchr13oldplink --maf 0.05 --make-bed --out 1kGPoldplinkchr13

#For the rest of the chromosomes:
#I ran this one first to test that this worked, the rest are below
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr15.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr15
#Chr 1-12,14,16-22
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr1.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr2.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr2
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr3.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr3
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr4.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr4
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr5.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr5
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr6.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr6
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr7.filtered.SNV_INDEL_SV_phased_panel.vcf --snps-only --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr7snps
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr8.filtered.SNV_INDEL_SV_phased_panel.vcf --snps-only --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr8snps
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr9.filtered.SNV_INDEL_SV_phased_panel.vcf --snps-only --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr9snps
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr10.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr10
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr11.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr11
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr12.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr12

plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr14.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr14
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr16.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr16
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr17.filtered.SNV_INDEL_SV_phased_panel.vcf --snps-only --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr17snps
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr18.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr18
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr19.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr19
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr20.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr20
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf --snps-only --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr21
plink2 --vcf Volumes/Seagate/1kGenomesFullVCF/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf --maf 0.05 --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr22

#Only need snps - started doing this for 7,8,9,21 while re-running it
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps
plink2 --bfile Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1 --snps-only --make-bed --out Volumes/Main/Users/cristinat/Desktop/repos/1kreference-genome/1kGPoldplinkchr1snps

