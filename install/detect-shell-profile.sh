#!/usr/bin/env bash
# detect-shell-profile.sh

# Detect the shell profile file
USER_SHELL=$(getent passwd "$USER" | cut -d: -f7)

case "$USER_SHELL" in
    */bash)
        # Use .bash_profile if it exists; otherwise, use .bashrc
        if [ -f "$HOME/.bash_profile" ]; then
            PROFILE_FILE="$HOME/.bash_profile"
        elif [ -f "$HOME/.bashrc" ]; then
            PROFILE_FILE="$HOME/.bashrc"
        else
            PROFILE_FILE="$HOME/.bash_profile"
        fi
        ;;
    */zsh)
        PROFILE_FILE="$HOME/.zshrc"
        ;;
    */ksh)
        PROFILE_FILE="$HOME/.kshrc"
        ;;
    *)
        # Default to .profile for other shells
        PROFILE_FILE="$HOME/.profile"
        ;;
esac

# Ensure the profile file exists
if [ ! -f "$PROFILE_FILE" ]; then
    touch "$PROFILE_FILE"
    echo "# Created $PROFILE_FILE" >> "$PROFILE_FILE"
    echo "$PROFILE_FILE has been created."
fi

echo "$PROFILE_FILE"