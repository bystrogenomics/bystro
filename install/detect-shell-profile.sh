#!/usr/bin/env bash
# detect-shell-profile.sh

# Detect the shell profile file
USER_SHELL=$(getent passwd "$USER" | cut -d: -f7)

if [[ -n "$SUDO_USER" ]]; then
  HOME_DIR="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  HOME_DIR="$HOME"
fi

case "$USER_SHELL" in
    */bash)
        # Use .bash_profile if it exists; otherwise, use .bashrc
        if [ -f "$HOME_DIR/.bash_profile" ]; then
            PROFILE_FILE="$HOME_DIR/.bash_profile"
        elif [ -f "$HOME_DIR/.bashrc" ]; then
            PROFILE_FILE="$HOME_DIR/.bashrc"
        else
            PROFILE_FILE="$HOME_DIR/.bash_profile"
        fi
        ;;
    */zsh)
        PROFILE_FILE="$HOME_DIR/.zshrc"
        ;;
    */ksh)
        PROFILE_FILE="$HOME_DIR/.kshrc"
        ;;
    *)
        # Default to .profile for other shells
        PROFILE_FILE="$HOME_DIR/.profile"
        ;;
esac

# Ensure the profile file exists
if [ ! -f "$PROFILE_FILE" ]; then
    touch "$PROFILE_FILE"
    echo "# Created $PROFILE_FILE" >> "$PROFILE_FILE"
    echo "$PROFILE_FILE has been created."
fi

echo "$PROFILE_FILE"