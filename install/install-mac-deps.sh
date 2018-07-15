#!/usr/bin/env bash

echo -e "\n\nInstalling homebrew, and MacOS dependencies using the same"

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
echo 'export PATH="/usr/local/opt/gettext/bin:$PATH"' >> $HOME/.bash_profile

# Run install and upgrade for key packages in case we have older versions
# installed at runtime
brew install mysql
brew upgrade mysql
brew install gcc
brew upgrade gcc
brew install openssl
brew upgrade openssl
brew install wget
brew install pigz

brew install nodejs
sudo npm install -g pm2