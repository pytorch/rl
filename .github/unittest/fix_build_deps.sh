#!/usr/bin/env bash

# Fix all install scripts to install build dependencies BEFORE PyTorch

for file in \
    .github/unittest/linux_libs/scripts_brax/install.sh \
    .github/unittest/linux_libs/scripts_chess/install.sh \
    .github/unittest/linux_libs/scripts_d4rl/install.sh \
    .github/unittest/linux_libs/scripts_gen-dgrl/install.sh \
    .github/unittest/linux_libs/scripts_gym/install.sh \
    .github/unittest/linux_libs/scripts_habitat/install.sh \
    .github/unittest/linux_libs/scripts_jumanji/install.sh \
    .github/unittest/linux_libs/scripts_llm/install.sh \
    .github/unittest/linux_libs/scripts_meltingpot/install.sh \
    .github/unittest/linux_libs/scripts_minari/install.sh \
    .github/unittest/linux_libs/scripts_open_spiel/install.sh \
    .github/unittest/linux_libs/scripts_openx/install.sh \
    .github/unittest/linux_libs/scripts_pettingzoo/install.sh \
    .github/unittest/linux_libs/scripts_roboset/install.sh \
    .github/unittest/linux_libs/scripts_robohive/install_and_run_test.sh \
    .github/unittest/linux_libs/scripts_sklearn/install.sh \
    .github/unittest/linux_libs/scripts_smacv2/install.sh \
    .github/unittest/linux_libs/scripts_unity_mlagents/install.sh \
    .github/unittest/linux_libs/scripts_vd4rl/install.sh \
    .github/unittest/linux_libs/scripts_vmas/install.sh \
    .github/unittest/linux_libs/scripts_isaaclab/isaac.sh \
    .github/unittest/linux_libs/scripts_d4rl/run_test.sh
do
    if [ -f "$file" ]; then
        echo "Fixing $file"
        
        # Remove the late build deps installation (right before torchrl install)
        sed -i.bak '/^# Install build dependencies (required for --no-build-isolation)/,/^uv pip install setuptools wheel numpy ninja "pybind11\[global\]" cmake$/d' "$file"
        sed -i.bak '/^printf "\* Installing build dependencies\\n"$/,/^uv pip install setuptools wheel numpy ninja "pybind11\[global\]" cmake$/d' "$file"
        
        # Add early build deps installation after venv activation
        # Find the line with "source.*activate" and add build deps after it
        awk '
        /source.*\.venv.*activate/ {
            print
            print ""
            print "# Install build dependencies EARLY (required for --no-build-isolation)"
            print "printf \"* Installing build dependencies\\n\""
            print "uv pip install setuptools ninja \"pybind11[global]\""
            print ""
            next
        }
        { print }
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        
        rm -f "$file.bak"
    fi
done

echo "Done!"

