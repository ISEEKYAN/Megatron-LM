#!/usr/bin/env bash
# Output ownership is checked before any receipts or service logs are written.
recipe_claim_output() {
 local output=${1:?output} job=${2:?job}
 mkdir -p "$(dirname "$output")"
 if ! mkdir "$output"; then
  echo "RECIPE_OUTPUT_REUSED: $output already exists; prior markers or receipts must not be reused" >&2
  return 1
 fi
 printf '%s\n' "$job" > "$output/job-owner-$job"
}
recipe_check_output() {
 local output=${1:?output} job=${2:?job} marker
 if [[ ! -f "$output/job-owner-$job" ]] || [[ "$(cat "$output/job-owner-$job")" != "$job" ]]; then
  echo "RECIPE_OUTPUT_REUSED: $output is not owned by job $job" >&2
  return 1
 fi
 for marker in "$output"/job-owner-* "$output"/stop-workers*; do
  [[ -e "$marker" ]] || continue
  case "${marker##*/}" in "job-owner-$job"|"stop-workers-$job") ;; *)
   echo "RECIPE_OUTPUT_REUSED: foreign job marker $marker" >&2
   return 1 ;;
  esac
 done
}
