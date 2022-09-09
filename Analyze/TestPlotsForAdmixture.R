setwd("/Applications/GenomicsTools")
setwd("~/Desktop/repos/1kreference-genome/")

tbl=read.table("pruned500ksnps.5.Q")
barplot(t(as.matrix(tbl)), col=rainbow(5),
          xlab="Individual #", ylab="Ancestry", border=NA)

dev.copy(png,'plots/All1kGP500ksnps5pops.png')
dev.off()

library(ggplot2)
library(plotly)
#read.sas7bdat('adspallbirths.sas7bdat', debug=FALSE)

#PCA plotting - hapmap
PCAtbl=read.table("IDsWqscore.txt")
Neatpcatbl=data.frame(PCAtbl[1],PCAtbl[2],PCAtbl[7],PCAtbl[8],PCAtbl[13])
colnames(Neatpcatbl) <- c("ID","ID2","PC1", "PC2","ClrCode")

#PCA plotting - eigenvec file combined with fam file
PCAtbl=read.table('IDsWqscore.txt')
Neatpcatbl=data.frame(PCAtbl[7],PCAtbl[8],PCAtbl[1],PCAtbl[2])
#Neatpcatbl2=data.frame(PCAtbl[9],PCAtbl[10],PCAtbl[1],PCAtbl[6])
colnames(Neatpcatbl) <- c("Q1", "Q2","ID","ID2")
#colnames(Neatpcatbl2) <- c("PC3", "PC4","ID","ID2")
attach(Neatpcatbl)
plot(Q1, Q2, main = 'PC analysis', xlab='Q1', ylab ='Q2')

#plot without caco 
#attach(Neatpcatbl)
#plot(PC1, PC2, main = 'PC analysis', xlab='PC1', ylab = 'PC2')
attach(Neatpcatbl)
plot(PC1, PC2, col=c('red','blue')[ID2], main = 'PC analysis', xlab='PC1', ylab ='PC2')
attach(Neatpcatbl2)
plot(PC3, PC4, col=c('red','blue')[ID2], main = 'PC analysis', xlab='PC3', ylab = 'PC4')
library(plotly)
p <- plot_ly(Neatpcatbl, x = PC1 , y = PC2, text = ID, type = "scatter",
             mode = "markers", color = ID2, marker = list(size = 2)) 
p <-  add_trace(p, x = PC1, y = PC2, type = "scatter",  mode = "markers", color = ID2, marker = list(size = 2)) 
p <-  layout(p, title = "PCA Clusters", 
             xaxis = list(title = "PC 1"),
             yaxis = list(title = "PC 2"))
p

p <- plot_ly(Neatpcatbl2, x = PC3 , y = PC4, text = ID, type = "scatter",
             mode = "markers", color = ID2, marker = list(size = 2)) 
p <-  add_trace(p, x = PC3, y = PC4, type = "scatter",  mode = "markers", color = ID2, marker = list(size = 2)) 
p <-  layout(p, title = "PCA Clusters", 
             xaxis = list(title = "PC 3"),
             yaxis = list(title = "PC 4"))
p

#Hapmap
attach(Neatpcatbl)
plot(PC1, PC2, col=c('red','black','blue','magenta','cyan','green')[ClrCode], main = 'PC analysis', xlab='PC1', ylab = 'PC2')
attach(Neatpcatbl2)
plot(PC3, PC4, col=c('red','black','blue','magenta','cyan','green')[ClrCode], main = 'PC analysis', xlab='PC3', ylab = 'PC4')


p <- plot_ly(Neatpcatbl, x = PC1 , y = PC2, text = ID, type = "scatter",
             mode = "markers", color = ClrCode, marker = list(size = 6)) 
p <-  add_trace(p, x = PC1, y = PC2, type = "scatter",  mode = "markers", color = ClrCode, marker = list(size = 6)) 
p <-  layout(p, title = "PCA Clusters", 
             xaxis = list(title = "PC 1"),
             yaxis = list(title = "PC 2"))
p

p <- plot_ly(Neatpcatbl, x = PC3 , y = PC4, text = ID, type = "scatter",
             mode = "markers", color = ClrCode, marker = list(size = 6)) 
p <-  add_trace(p, x = PC1, y = PC2, type = "scatter",  mode = "markers", color = ClrCode, marker = list(size = 6)) 
p <-  layout(p, title = "PCA Clusters", 
             xaxis = list(title = "PC 3"),
             yaxis = list(title = "PC 4"))
p