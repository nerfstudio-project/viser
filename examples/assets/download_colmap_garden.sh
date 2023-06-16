# This downloads the COLMAP model for the MIP-NeRF garden dataset
# with the images that are downscaled by a factor of 8.
# The full dataset is available at https://jonbarron.info/mipnerf360/.

set -e -x

gdown "https://drive.google.com/uc?id=1wYHdrgwXPHtREdCjItvt4gqRQGISMade"

mkdir -p colmap_garden
# shellcheck disable=SC2035
unzip *.zip && rm *.zip