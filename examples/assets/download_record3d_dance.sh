set -e -x

gdown "https://drive.google.com/uc?id=1_vd5bK_MhtlfisA6BkK1IgiJNfDbIntq"

mkdir record3d_dance
unzip "*.r3d" -d record3d_dance
rm "*.r3d"
