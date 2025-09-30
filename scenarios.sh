#!/bin/bash
cfgs=("/home/alex-charly/SSD/Kenya/analyse-pypsa-kenya/config/config_KE.yaml"
      "/home/alex-charly/SSD/Kenya/analyse-pypsa-kenya/config/config_KE_geo.yaml"
      "/home/alex-charly/SSD/Kenya/analyse-pypsa-kenya/config/config_KE_geo_pot.yaml")

for cfg in "${cfgs[@]}"; do
    echo "Running $cfg"
    snakemake --cores 32 solve_sector_networks --configfile $cfg
    echo "Run for $cfg finished successfully."

done