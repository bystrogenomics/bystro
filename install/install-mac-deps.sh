#!/usr/bin/env bash

echo "Installing homebrew, and MacOS dependencies using the same"

/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
echo 'export PATH="/usr/local/opt/gettext/bin:$PATH"' >> ~/.bash_profile

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
