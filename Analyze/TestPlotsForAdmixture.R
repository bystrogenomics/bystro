setwd("/Applications/GenomicsTools")

tbl=read.table("hapmap3.3.Q")
barplot(t(as.matrix(tbl)), col=rainbow(3),
          xlab="Individual #", ylab="Ancestry", border=NA)