#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$project_root/ros_ws"
catkin_cmd="$project_root/.venvs/ros/bin/catkin"

if [[ ! -x "$catkin_cmd" ]]; then
  echo "Catkin is missing at $catkin_cmd" >&2
  exit 1
fi

cd "$workspace"
"$catkin_cmd" config --init --extend "$project_root/.venvs/ros/melodic" --cmake-args \
  "-DPYTHON_EXECUTABLE=$project_root/.venvs/ros/bin/python" \
  -DSETUPTOOLS_DEB_LAYOUT=OFF \
  -DCATKIN_ENABLE_TESTING=OFF \
  -DBoost_NO_BOOST_CMAKE=ON
