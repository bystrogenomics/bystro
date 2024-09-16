# utils.sh

# Function to append a line to PROFILE_FILE if it doesn't already exist
append_if_missing() {
  local line="$1"
  local profile_file="$2"
  local found=0

  # Read the profile file line by line and check if the line exists
  while IFS= read -r current_line; do
    if [ "$current_line" = "$line" ]; then
      found=1
      break
    fi
  done < "$profile_file"

  # If the line was not found, append it to the file
  if [ $found -eq 0 ]; then
    echo -e "\n$line" >> "$profile_file"
    echo "Added to $profile_file: $line"
  else
    echo "Already present in $profile_file: $line"
  fi
}
