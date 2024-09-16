#!/usr/bin/env bash
# detect-shell-profile.sh

# Detect the shell profile file
USER_SHELL=$(getent passwd "$USER" | cut -d: -f7)

if [[ -n "$SUDO_USER" ]]; then
  HOME_DIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  HOME_DIR="$HOME"
fi

# Use ~/.profile unless ~/.bash_profile already exists
PROFILE_FILE="$HOME_DIR/.profile"

if [ -f "$HOME_DIR/.bash_profile" ]; then
    PROFILE_FILE="$HOME_DIR/.bash_profile"
fi

# Ensure the profile file exists
if [ ! -f "$PROFILE_FILE" ]; then
    touch "$PROFILE_FILE"
    echo "# Created $PROFILE_FILE" >> "$PROFILE_FILE"
    echo "$PROFILE_FILE has been created." >&2
fi

echo "$PROFILE_FILE"