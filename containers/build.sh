#!/usr/bin/env bash
# Build Apptainer images in dependency order, from the repo root (so %files paths resolve).
# Needs apptainer + --fakeroot (WSL/CI). M3 cannot build images (no root): copy the .sif there.
#   containers/build.sh                       # all
#   containers/build.sh warpx                 # builds mpi-base first if missing
#   OMPI_VERSION=4.1.6 containers/build.sh mpi-base
# Output: sif/<component>-<version>.sif (+ .sha256) and sif/<component>.sif -> that file
# (the latter is what dependent .def files use as their base).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sif
GIT_DESC="$(git describe --always --dirty --tags 2>/dev/null || echo unknown)"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' pyproject.toml | head -1)"

build() {   # build <component> [extra apptainer build args]
  local c="$1"; shift
  local out="sif/${c}-${VERSION}.sif"
  echo "==> $c -> $out (git $GIT_DESC)"
  apptainer build --fakeroot --force --build-arg "FUSORSIM_GIT=$GIT_DESC" "$@" "$out" "containers/${c}.def"
  ln -sf "${c}-${VERSION}.sif" "sif/${c}.sif"
  sha256sum "$out" | tee "$out.sha256"
}

need() { [ -e "sif/$1.sif" ] || build "$1"; }

[ $# -eq 0 ] && set -- mpi-base warpx openmc analysis viz ansys
for c in "$@"; do
  case "$c" in
    warpx|openmc) need mpi-base ;;
    viz)          need analysis ;;
  esac
  if [ "$c" = mpi-base ] && [ -n "${OMPI_VERSION:-}" ]; then
    build mpi-base --build-arg "OMPI_VERSION=$OMPI_VERSION"
  else
    build "$c"
  fi
done
