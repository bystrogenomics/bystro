#!/usr/bin/env bash

echo -e "\n\nInstalling liftover into /usr/local/bin/"

# LiftOver is used for the LiftOverCadd.pm package, to liftOver cadd to hg38
# and cadd's GRCh37.p13 MT to hg19

wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/liftOver && chmod +x liftOver && sudo mv $_ /usr/local/bin/;