# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Proposed code structure:
X read network (.nc-file)
X add export bus
X connect hydrogen buses (advanced: only ports, not all) to export bus
X add store and connect to export bus
X (add load and connect to export bus) only required if the "store" option fails

Possible improvements:
- Select port buses automatically (with both voronoi and gadm clustering). Use data/ports.csv?
"""


import logging
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import locate_bus, override_component_attrs, prepare_costs

logger = logging.getLogger(__name__)

def parse_eopts(eopts_string):
    """
    Parse eopts string to extract marginal cost factors and volume factors.
    Parameters
    ----------
    eopts_string : str
        String like "H2m1.0+NH3m1.0+FTm1.0+H2v1.0+NH3v1.0+FTv1.0" or subsets
    Returns
    -------
    tuple
        (price_factor, volume_factor) dictionaries
    Examples
    --------
    >>> parse_eopts("H2m1.0+NH3m1.0+FTm1.0+H2v1.0+NH3v1.0+FTv1.0")
    ({'H2': 1.0, 'NH3': 1.0, 'FT': 1.0}, {'H2': 1.0, 'NH3': 1.0, 'FT': 1.0})
    >>> parse_eopts("H2+NH3+H2m1.5+NH3m2.0")
    ({'H2': 1.5, 'NH3': 2.0}, {'H2': 1.0, 'NH3': 1.0})
    >>> parse_eopts("H2+NH3")
    ({'H2': 1.0, 'NH3': 1.0}, {'H2': 1.0, 'NH3': 1.0})
    """
    price_factor = {}
    volume_factor = {}

    # Pattern to match carrier with optional marginal cost (m) or volume (v) factor
    # Matches: H2m1.0, NH3v2.5, FT, etc.
    pattern = r"([A-Z0-9]+)([mv]?)([\d.]*)"

    matches = re.findall(pattern, eopts_string)

    for carrier, factor_type, value in matches:

        if carrier not in ["H2", "NH3", "FT"]:
            raise NotImplementedError(f"'{carrier}' is not yet implemented.")

        # Default factor is 1.0 if no value specified
        factor_value = float(value) if value else 1.0


        if factor_type == "m":
            price_factor[carrier] = factor_value
        elif factor_type == "v":
            volume_factor[carrier] = factor_value
        elif factor_type != "": 
            raise ValueError(
                f"Invalid factor type '{factor_type}' for carrier '{carrier}'. "
                "Expected 'm' for marginal cost factor or 'v' for volume factor "
                "or '' for factor=1.0."
            )
        # set default 1.0, if not already set
        if carrier not in price_factor.keys():
            price_factor[carrier] = 1.0
        if carrier not in volume_factor.keys():
            volume_factor[carrier] = 1.0

    return pd.Series(price_factor), pd.Series(volume_factor)


def create_ship_profile(export_volume, ship_opts):
    """
    Create a ship profile with absolute values for hydrogen export.
    Parameters
    ----------
    export_volume : float
        The annual export volume in MWh.
    ship_opts : dict
        Dictionary containing ship options with keys:
        - ship_capacity: float
        - travel_time: float
        - fill_time: float
        - unload_time: float
    Returns
    -------
    pd.Series
        The ship profile with absolute values as a pandas Series.
    """
    ship_capacity = ship_opts["ship_capacity"] * 1e6  # convert TWh/ship to MWh/ship
    travel_time = ship_opts["travel_time"]
    fill_time = ship_opts["fill_time"]
    unload_time = ship_opts["unload_time"]

    landing = export_volume / ship_capacity  # fraction of max delivery
    pause_time = 8760 / landing - (fill_time + travel_time)
    # full_cycle = fill_time + travel_time + unload_time + pause_time # not used

    max_transport = ship_capacity * 8760 / (fill_time + travel_time + unload_time)
    print(f"The maximum transport capacity per ship is {max_transport/1e6:.2f} TWh/year")

    # throw error if max_transport < export_volume
    if max_transport < export_volume:
        ships = np.ceil(export_volume / max_transport)
        print(f"Number of ships needed to export {export_volume} TWh/year is {ships}")
        logger.info(
            "Not enough ship capacity to export all hydrogen in one ship. " \
            "Extending the number of ships to {}".format(ships)
        )

    # Set fill_time ->  1 and travel_time, unload_time, pause_time -> 0
    ship = pd.Series(
        [1.0] * fill_time + [0.0] * int(travel_time + unload_time + pause_time)
    )  # , index)
    ship.name = "profile"
    ship = pd.concat(
        [ship] * 1000, ignore_index=True
    )  # extend ship series to above 8760 hours

    # Add index, cut profile after length of snapshots
    snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
    ship = ship[: len(snapshots)]
    ship.index = snapshots

    # Scale ship profile to export_volume
    export_profile = ship / ship.sum() * export_volume

    if np.abs(export_profile.sum() - export_volume) > 1:  # Threshold of 1 MWh
        raise ValueError(
            f"Sum of ship profile ({export_profile.sum()/1e6} TWh) "
            f"does not match export demand ({export_volume} TWh)"
        )

    return export_profile


def create_export_profile(export_volume, export_type="constant"):
    """
    This function creates an export profile time series 
    based on the specified annual export volume and export profile type.
    Supported profile types include:
    - "constant": Distributes the annual export volume evenly across all hours of the year.
    - "ship": Uses a custom ship-based profile via `create_ship_profile`.
    The resulting profile is resampled to the temporal resolution defined by the "sopts" wildcard in Snakemake,
    which should specify the desired frequency (e.g., "1h", "3h", etc.).
    Parameters
    ----------
    export_volume : float
        The total annual export demand (in MWh).
    Returns
    -------
    pd.Series
        Time-indexed export profile resampled to the specified temporal resolution.
    
    """

    if export_type == "constant":
        export_profile = export_volume / 8760
        snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
        export_profile = pd.Series(export_profile, index=snapshots)

    elif export_type == "ship":

        export_profile = create_ship_profile(export_volume, snakemake.params.export_ship)

    # Resample to temporal resolution defined in wildcard "sopts" with pandas resample
    sopts = snakemake.wildcards.sopts.split("-")
    for o in sopts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            freq = m.group(0).casefold()
            export_profile = export_profile.resample(freq).mean()

    logger.info(
        f"The yearly export demand is {export_volume/1e6} TWh, "
        f"profile generated based on {export_type} method and resampled to {freq}"
    )

    return export_profile

def select_ports(n):
    """
    This function selects the buses where ports are located.
    """

    ports = pd.read_csv(
        snakemake.input.export_ports,
        index_col=None,
        keep_default_na=False,
    ).squeeze()

    gadm_layer_id = snakemake.params.gadm_layer_id

    ports = locate_bus(
        ports,
        countries,
        gadm_layer_id,
        snakemake.input.shapes_path,
        snakemake.params.alternative_clustering,
    )

    # TODO: revise if ports quantity and property by shape become relevant
    # drop duplicated entries
    gcol = "gadm_{}".format(gadm_layer_id)
    ports_sel = ports.loc[~ports[gcol].duplicated(keep="first")].set_index(gcol)

    # Select and define the nodes with ports
    nodes_with_port = n.buses.loc[ports_sel.index].index
    nodes_with_port.name = "Bus"

    return nodes_with_port


def add_export(n, exp_carrier, volume, price, profile, nodes_with_port, costs, snakemake):
    """    
    This function creates a centralized export system by adding:
    1. A central export bus
    2. Export links connecting port buses to the central export bus
    3. Export demand (as load or negative generator) at the central bus
    4. Optional hydrogen storage for buffering export demand
    
    The function supports both direct export (H2, NH3) and carbon-neutral 
    hydrocarbon export (FT, MeOH) with CO2 accounting.
    
    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify
    exp_carrier : str
        Export carrier type. Supported: 'H2', 'NH3', 'FT'
    volume : float
        Annual export volume in MWh. Can be np.inf for unlimited endogenous export
    price : float
        Export price in currency per MWh. Used as negative marginal cost for export links
    profile : pd.Series
        Time series profile for export demand (used only for exogenous export)
    nodes_with_port : pd.Index
        List of bus names where ports are located
    costs : pd.DataFrame
        Cost parameters DataFrame for CO2 intensity and storage costs
    snakemake : snakemake.Snakemake
        Snakemake workflow object containing configuration parameters
        
    Notes
    -----
    
    For H2 and NH3, direct export links are created with 100% efficiency.
    For FT and MeOH, links include CO2 accounting to ensure carbon neutrality,
    connecting to 'co2 atmosphere' bus with appropriate CO2 intensity factors.
    
    Export implementation depends on configuration:
    - Endogenous export: Adds negative generator with optional volume constraint
    - Exogenous export: Adds load with specified profile
    
    Optional H2 storage can be added for export buffering when configured.
    
    Raises
    ------
    KeyError
        If required carrier buses are not found in the network
    NotImplementedError
        If export carrier is not supported
    ValueError
        If configuration parameters are invalid
        
    Examples
    --------
    >>> add_export(network, 'H2', 1000000, 50.0, h2_profile, ports, cost_df, snakemake_obj)
    # Adds H2 export with 1 TWh annual volume at 50 €/MWh
    
    """
    country_shape = gpd.read_file(snakemake.input["shapes_path"])
    # Find most northwestern point in country shape and get x and y coordinates
    country_shape = country_shape.to_crs(
        "EPSG:3395"
    )  # Project to Mercator projection (Projected)

    # Get coordinates of the most western and northern point of the country and add a buffer of 2 degrees (equiv. to approx 220 km)
    x_export = country_shape.geometry.centroid.x.min() - 2
    y_export = country_shape.geometry.centroid.y.max() + 2

     # add one central export bus
    n.add(
        "Bus",
        exp_carrier + " export bus",
        carrier=exp_carrier + " export bus",
        location="Earth",
        x=x_export,
        y=y_export,
    )

    # add links for H2 and NH3 export
    if exp_carrier in ["H2", "NH3"]:

        try: 
            buses_ports = n.buses.loc[nodes_with_port + " " + exp_carrier].index
        except KeyError:
            raise KeyError(
                f"No buses found for {exp_carrier}. "
                "Ensure that the carrier (e.g. ammonia) is activated in the config."
            )
        logger.info(f"Adding green export links from {buses_ports} to central {exp_carrier} export bus, "
                    f"with price {price}")

        # TODO: decide if we want add liquefaction as a intermediate step. Easy to implement, but complicates result analysis.

        n.madd(
            "Link",
            buses_ports + " export",
            bus0=buses_ports,
            bus1=exp_carrier + " export bus",
            carrier=exp_carrier + " export",
            p_nom=1e7, #volume * 0.01,  # TODO: check if setting p_nom to 1% of annual export volume is interesting
            efficiency=1,
            marginal_cost=-price,
        )

    # add links for FT and MeOH with accounting for CO2
    elif exp_carrier in ["FT"]: # TODO: add "MeOH" once implemented in all steps
        # For the green hydrocarbon export, the reference bus carrier are oil, methanol or gas.
        # An extra constraints will be added in solve_network to ensure that the green liquid fuel conversion
        # technologies from hydrogen to X will be used ('>='). 
        # Feeding the co2 fraction of the exported products back to the co2 atmosphere of the system ensures, 
        # that the exports are green and carbon neutral on a system level.
        ref_bus_carrier = {
            "FT": "oil",
            "MeOH": "methanol",
            }
        ref_bus_carrier = ref_bus_carrier[exp_carrier]

        co2_i = {
            "FT": ("oil", "CO2 intensity"),
            "MeOH": ("methanolisation", "carbondioxide-input"),
        }
        co2_intensity = costs.at[co2_i[exp_carrier][0], co2_i[exp_carrier][1]]

        try: 
            buses_ports = n.buses.loc[nodes_with_port + " " + ref_bus_carrier].index
        except KeyError:
            raise KeyError(
                f"No buses found for {exp_carrier}. "
                "Ensure that the carrier (e.g. methanol) is activated in the config."
            )

        logger.info(f"Adding green export links from {buses_ports} to central {exp_carrier} export bus, "
                    f"with price {price} and CO2 intensity {co2_intensity}")
        n.madd(
            "Link",
            buses_ports + " export",
            bus0=buses_ports,
            bus1=exp_carrier + " export bus",
            bus2="co2 atmosphere",
            carrier=exp_carrier + " export",
            p_nom_extendable=True,
            p_nom=1e7, #volume * 0.01,  # TODO: check if setting p_nom to 1% of annual export volume is interesting
            efficiency=1,
            efficiency2=co2_intensity,
            marginal_cost=-price,
        )
    else:
        raise NotImplementedError(f"Export carrier {exp_carrier} not implemented")

    export_links = n.links[n.links.index.str.contains("export")]
    logger.info(f"Added export links: {export_links.index}")

    # Add export load
    # If endogenous export is true, 
    #   1. a negative generator is added, 
    #   2. a global constraint is added if volume is not ".inf"
    # If endogenous export is false, an exogenous load with a profile is added.  
    if snakemake.params.export_endogenous:
        # add endogenous export by implementing a negative generation
        n.add(
            "Generator",
            exp_carrier + " export",
            bus=exp_carrier + " export bus",
            carrier=exp_carrier + " export",
            p_nom=1e7, #volume * 0.01,  # TODO: check if setting p_nom to 1% of annual export volume is interesting
            p_max_pu=0, 
            p_min_pu=-1,
        )

        if volume < np.inf:
            logger.info(f"Adding global constraint to limit export of {exp_carrier} to {volume/1e6} TWh/a")
            n.add(
                "GlobalConstraint",
                exp_carrier + "_max_export_limit",
                type="operational_limit",
                carrier_attribute=exp_carrier + " export",
                sense=">=", # ">=" because the export generators p is negative.
                constant=-volume,
            )

    elif snakemake.params.export_endogenous is False:
        if volume == np.inf or volume == 0:
            raise ValueError(
                f"Value {volume} for ['export']['volume'] is not valid. "
                "It must be a finite number and greater than zero."
            )

        profile_type = snakemake.params.export_profile
        logger.info(f"Adding exogenous export of {exp_carrier} by implementing a load {volume/1e6} TWh/a "
                    f"with a {profile_type} profile type.")
        n.add(
            "Load",
            exp_carrier + " export",
            bus=exp_carrier + " export bus",
            carrier=exp_carrier + " export",
            p_set=profile,
        )
    else: 
        raise ValueError(
            f"Value {snakemake.params.export_endogenous} for ['export']['endogenous'] must be true or false."
        )

    # add store depending on config settings
    if snakemake.params.export_store == True and exp_carrier == "H2":
        if snakemake.params.export_store_capital_costs == "no_costs":
            capital_cost = 0
        elif snakemake.params.export_store_capital_costs == "standard_costs":
            capital_cost = costs.at[
                "hydrogen storage tank type 1 including compressor", "fixed"
            ]
        else:
            raise ValueError(
                f"Value {snakemake.params.export_store_capital_costs} for ['export']['store_capital_costs'] "
                "is not valid"
            )

        n.add(
            "Store",
            "H2 export store",
            bus="H2 export bus",
            e_nom_extendable=True,
            carrier="H2 export store",
            e_initial=0,  # actually not required, since e_cyclic=True
            marginal_cost=0,
            capital_cost=capital_cost,
            e_cyclic=True,
        )

    return


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_export",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate="0.071",
            demand="AB",
            h2export="120",
            # configfile="test/config.test1.yaml",
        )

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    countries = list(n.buses.country.unique())

    # Create export profile
    price_factor, volume_factor = parse_eopts(snakemake.wildcards["eopts"])
    export_carriers = list(set(price_factor.keys()).union(set(volume_factor.keys())))
    #logger.info(f"The following export carriers will be considered: {export_carriers}")

    # Prepare export volumes and prices for selected carriers
    export_volume_params = pd.Series(snakemake.params.export_volume).loc[export_carriers]
    export_price_params = pd.Series(snakemake.params.export_price).loc[export_carriers]

    # Calculate export volumes (in MWh) and prices
    export_volumes = export_volume_params * volume_factor * 1e6  # TWh to MWh
    export_prices = export_price_params * price_factor

    logger.info(f"Calculated export volumes (MWh): {export_volumes}")
    logger.info(f"Calculated export prices (Currency/MWh): {export_prices}")

    Nyears = n.snapshot_weightings.generators.sum() / 8760

    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs["output_currency"],
        snakemake.params.costs["fill_values"],
        Nyears,
        snakemake.params.costs["default_USD_to_EUR"],
    )

    # select and define nodes for export via port and shipping 
    nodes_with_port = select_ports(n)

    # add export values and components to network for each export carrier
    for export_carrier in export_carriers:

        export_volume = export_volumes[export_carrier]
        export_price = export_prices[export_carrier]
        logger.info(
            f"Creating export profile for {export_carrier}: "
            f"{'using specified export profile type' if export_carrier == 'H2' else 'using constant profile'}.")
        if export_carrier == "H2":
            export_profile = create_export_profile(export_volume, snakemake.params.export_profile)
        else:
            export_profile = create_export_profile(export_volume, export_type="constant")

        logger.info(f"Adding export for {export_carrier} with volume {export_volume} MWh "
                    f"and price {export_price} Currency/MWh")



        add_export(n, export_carrier, export_volume, export_price, export_profile, nodes_with_port, costs, snakemake)


    n.export_to_netcdf(snakemake.output[0])

    logger.info("Network successfully exported")
