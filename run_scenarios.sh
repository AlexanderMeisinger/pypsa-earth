#!/bin/bash
cfgs_countries=(
  "/home/alex-charly/SSD/H2GMA/Github/AP10/analyse-h2g-a-ap10/config/supply-scenarios/config.KE_2030.yaml"
  "/home/alex-charly/SSD/H2GMA/Github/AP10/analyse-h2g-a-ap10/config/supply-scenarios/config.KE_2050.yaml"
) 

for cfg in "${cfgs_countries[@]}"; do
  echo "Running $cfg"
  if snakemake \
    --cores 8 \
    --rerun-incomplete \
    solve_sector_networks \
    --configfile "$cfg"; then
    echo "Run for $cfg finished successfully."
  fi
done

