set -e -x

# gdown "https://drive.google.com/uc?id=1_vd5bK_MhtlfisA6BkK1IgiJNfDbIntq"

# mkdir -p record3d_dance
# shellcheck disable=SC2035
unzip *.r3d -d record3d_dance && rm *.r3d
