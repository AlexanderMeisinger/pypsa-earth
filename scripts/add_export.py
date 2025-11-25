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
from _helpers import locate_bus, override_component_attrs, prepare_costs, calculate_annuity
import searoute as sr

logger = logging.getLogger(__name__)

def parse_eopts(eopts_string):
    """
    Parse eopts string to extract marginal cost factors and volume factors.

    Parameters
    ----------
    eopts_string : str
        String like "H2m1.0+NH3m1.0+H2v1.0+NH3v1.0" or subsets

    Returns
    -------
    tuple
        (price_factor, volume_factor) dictionaries

    Examples
    --------
    >>> parse_eopts("H2m1.0+NH3m1.0+H2v1.0+NH3v1.0")
    ({'H2': 1.0, 'NH3': 1.0}, {'H2': 1.0, 'NH3': 1.0})

    >>> parse_eopts("H2+NH3+H2m1.5+NH3m2.0")
    ({'H2': 1.5, 'NH3': 2.0}, {'H2': 1.0, 'NH3': 1.0})

    >>> parse_eopts("H2+NH3")
    ({'H2': 1.0, 'NH3': 1.0}, {'H2': 1.0, 'NH3': 1.0})
    """
    price_factor = {}
    volume_factor = {}

    # Pattern to match carrier with optional marginal cost (m) or volume (v) factor
    # Matches: H2m1.0, NH3v2.5, etc.
    # Note: Only lowercase 'm' and 'v' are supported
    pattern = r"([A-Z0-9]+)([mv]?)([\d.]*)"

    matches = re.findall(pattern, eopts_string)

    for carrier, factor_type, value in matches:

        if carrier not in ["H2", "LH2", "NH3", "MEOH"]:
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

    # Select and define the nodes with ports and ports fraction
    port_fraction = ports_sel["fraction"].rename("port fraction")
    port_fraction.index.name = "Bus"

    # Select and define the nodes with ports
    nodes_with_port = n.buses.loc[ports_sel.index].index
    nodes_with_port.name = "Bus"

    return nodes_with_port, port_fraction


def find_nodes_to_connect_to_export_bus(n, exp_carrier, nodes_with_port, ref_bus_carrier=None):
    """
    Find nodes (buses) for specific export carriers with hierarchical fallback.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to search in
    exp_carrier : str
        The export carrier type (e.g., 'H2', 'NH3', 'MEOH')
    nodes_with_port : pd.Index
        Nodes with ports for regional search
    ref_bus_carrier : str, optional
        Reference bus carrier for hydrocarbon exports
        If None, uses exp_carrier directly
        
    Returns
    -------
    pd.Index
        Found bus indices
        
    Raises
    ------
    KeyError
        If no buses are found for the carrier
    """
    # Determine the actual carrier to search for
    search_carrier = ref_bus_carrier if ref_bus_carrier else exp_carrier
    
    # Search for carrier buses in order of preference: port-specific -> global
    candidate_patterns = [
        nodes_with_port + " " + search_carrier,  # Regional buses at ports
        ["Earth " + search_carrier]              # Global bus as fallback
    ]
    
    for pattern in candidate_patterns:
        try:
            buses = n.buses.loc[pattern].index
            if not buses.empty:
                logger.info(f"Found {search_carrier} buses for {exp_carrier} export: {list(buses)}")
                return buses
        except KeyError:
            continue
    
    # Enhanced error message
    error_msg = (
        f"No buses found for {exp_carrier} export. "
        f"Searched for: {search_carrier} buses. "
        f"Ensure that the carrier is properly configured:"
        f"e.g. ammonia, methanol, liquid h2 is set to true."
    )
    if ref_bus_carrier:
        error_msg += f" (Note: {exp_carrier} exports use {ref_bus_carrier} buses)"
    
    raise KeyError(error_msg)


def get_shipping_route(origin, destination):
    ports_fuel_export = pd.DataFrame(index=nodes_with_port, columns=["x", "y", "distance"])
    ports_fuel_export.x = n.buses.loc[nodes_with_port].lon
    ports_fuel_export.y = n.buses.loc[nodes_with_port].lat
    
    # Source searoutes: 10.1109/EEM64765.2025.11050281
    for i in ports_fuel_export.index:
        ports_fuel_export.loc[i, "distance"] = sr.searoute([ports_fuel_export.loc[i, "x"], ports_fuel_export.loc[i, "y"]], destination)["properties"]["length"] 

    shipping_route = ports_fuel_export

    return shipping_route


def get_ships_required(export_volume, ship_capacity, shipping_hour):   
    travel_time = 2 * shipping_hour
    fill_time = snakemake.params.export_ship_fill_time
    unload_time = snakemake.params.export_ship_unload_time

    max_transport = ship_capacity * 8760 / (fill_time + travel_time + unload_time)

    ships_required = np.ceil(export_volume / max_transport)

    return ships_required


def add_export(n, exp_carrier, volume, price, profile, nodes_with_port, port_fraction, costs, snakemake):
    """    
    This function creates a centralized export system by adding:
    1. A central export bus
    2. Export links connecting port buses to the central export bus
    3. Export demand (as load or negative generator) at the central bus
    4. Optional hydrogen storage for buffering export demand
    
    The function supports both direct export (H2, LH2, NH3) and carbon-neutral 
    hydrocarbon export (MEOH) with CO2 accounting.
    
    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to modify
    exp_carrier : str
        Export carrier type. Supported: 'H2', 'LH2', 'NH3'
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
    
    For H2, LH2 and NH3, direct export links are created with 100% efficiency.
    For MEOH, links include CO2 accounting to ensure carbon neutrality,
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

    # add one central export bus and carrier
    n.add("Carrier", exp_carrier + " export")

    n.add(
        "Bus",
        exp_carrier + " export",
        carrier=exp_carrier + " export",
        location="Earth",
        x=x_export,
        y=y_export,
    )

    # add links for exports without co2 intensity through export carrier
    if exp_carrier in ["H2", "NH3", "LH2"]:
        ref_bus_carrier = {
                "H2": "H2",
                "NH3": "NH3",
                "LH2": "H2 liquefaction"
                }[exp_carrier]

        nodes_to_connect = find_nodes_to_connect_to_export_bus(
            n, exp_carrier, nodes_with_port, ref_bus_carrier)
        
        logger.info(f"Adding green export links from {nodes_to_connect} to central {exp_carrier} export bus, "
                    f"with price {price}")
        
        if snakemake.params.export_crossborder:
            n.madd(
                "Bus",
                nodes_to_connect + " export",
                carrier=exp_carrier + " export",
                #location
            )

            n.madd(
                    "Link",
                    nodes_to_connect + " export",
                    bus0=nodes_to_connect,
                    bus1=nodes_to_connect + " export",
                    carrier=exp_carrier + " export",
                    p_nom=1e7, 
                    efficiency=1,
                )

            add_export_crossborder(exp_carrier, nodes_to_connect, port_fraction, price, profile)
        else:
            n.madd(
                    "Link",
                    nodes_to_connect + " export",
                    bus0=nodes_to_connect,
                    bus1=exp_carrier + " export",
                    carrier=exp_carrier + " export",
                    p_nom=1e7, 
                    efficiency=1,
                    marginal_cost=-price,
                )
        
    # add links for MEOH with accounting for CO2 intensity
    elif exp_carrier in ["MEOH"]:
        # For the green hydrocarbon export, the reference bus carrier are oil, methanol or gas.
        # An extra constraints will be added in solve_network to ensure that the green liquid fuel conversion
        # technologies from hydrogen to X will be used ('>='). 
        # Feeding the co2 fraction of the exported products back to the co2 atmosphere of the system ensures, 
        # that the exports are green and carbon neutral on a system level.
        ref_bus_carrier = {
            "MEOH": "methanol",
            }[exp_carrier]

        co2_i = {
            "MEOH": ("methanol", "CO2 intensity"),
        }
        co2_intensity = costs.at[co2_i[exp_carrier][0], co2_i[exp_carrier][1]]
        
        nodes_to_connect = find_nodes_to_connect_to_export_bus(
            n, exp_carrier, nodes_with_port, ref_bus_carrier
        )
        
        logger.info(f"Adding green export links from {nodes_to_connect} to central {exp_carrier} export bus, "
                    f"with price {price} and CO2 intensity {co2_intensity}")
        

        if snakemake.params.export_crossborder:
            n.madd(
                "Bus",
                nodes_to_connect + " export",
                carrier=exp_carrier + " export",
                #location
            )
            
            n.madd(
                    "Link",
                    nodes_to_connect + " export",
                    bus0=nodes_to_connect,
                    bus1=nodes_to_connect + " export",
                    bus2="co2 atmosphere",
                    carrier=exp_carrier + " export",
                    p_nom=1e7, 
                    efficiency=1,
                    efficiency2=co2_intensity, # ToDo: negativ?
                )

            add_export_crossborder(exp_carrier, nodes_to_connect, port_fraction, price, profile)
        else:
            n.madd(
                    "Link",
                    nodes_to_connect + " export",
                    bus0=nodes_to_connect,
                    bus1=exp_carrier + " export",
                    bus2="co2 atmosphere",
                    carrier=exp_carrier + " export",
                    p_nom=1e7, 
                    efficiency=1,
                    efficiency2=co2_intensity, # ToDo: negativ?
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
    if snakemake.params.export_endogenous == True and snakemake.params.export_crossborder == False:
        # add endogenous export by implementing a negative generation
        n.add(
            "Generator",
            exp_carrier + " export",
            bus=exp_carrier + " export",
            carrier=exp_carrier + " export",
            p_nom=1e7, 
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

    elif snakemake.params.export_endogenous == False and snakemake.params.export_crossborder == False:
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
            bus=exp_carrier + " export",
            carrier=exp_carrier + " export",
            p_set=profile,
        )

    # add store at export bus depending on config settings
    if snakemake.params.export_destination_carrier == False and snakemake.params.export_crossborder == False:
        config_store = snakemake.params.export_store
        config_store_costs = snakemake.params.export_store_capital_costs

        if config_store == True:
            if config_store_costs == "no_costs":
                capital_cost = 0
            elif config_store_costs == "standard_costs" and exp_carrier == "H2":
                capital_cost = costs.at[
                    "hydrogen storage tank type 1 including compressor", "fixed"
                ]
            else:
                raise ValueError(
                    f"Combination of values for export_store and export_store_capital_costs ({config_store_costs}, {exp_carrier}) "
                    "are not valid!"
                )

            n.add("Carrier", exp_carrier + " export Store")

            n.add(
                "Store",
                exp_carrier + " export Store",
                bus=exp_carrier + " export",
                e_nom_extendable=True,
                carrier=exp_carrier + " export Store",
                marginal_cost=0,
                capital_cost=capital_cost,
                e_cyclic=True,
            )

    return


def add_export_crossborder(exp_carrier, nodes_to_connect, port_fraction, price, profile):
    """
    Add cross-border export infrastructure for ship-based energy exports.

    This function builds a complete export chain for an energy carrier by ship 
    (e.g. LH2, NH3, MEOH, oil) including:
    - loading infrastructure,
    - ship transport including boil-off losses,
    - fuel consumption of the ships,
    - emission accounting,
    - unloading infrastructure,
    - import to to foreign markets.

    Parameters
    ----------
    exp_carrier : str
        Exported energy carrier (e.g. "LH2", "NH3", "MEOH").
        Used for naming buses, links, carriers, and selecting ship data.

    nodes_to_connect : pandas.Index or pandas.Series
        Buses representing ports in the exporting region. 
        All infrastructure is created for each of these buses.

    port_fraction : pandas.Series
        Weight of export distribution among ports. 

    price : float
        Export price (€/MWh). Used as negative marginal cost 
        on the final export link.

    Returns
    -------
    None
        Modifies the PyPSA network `n` in place by adding buses, links,
        stores, loads, carriers, and a global constraint.
    """

    fuel_export = snakemake.params.export_ship_fuel
    destination = snakemake.params.export_destination
    discountrate = snakemake.params.costs["discountrate"][0]

    # load shipping data (capacity, losses, speed, etc.)
    # source: 10.1109/EEM64765.2025.11050281
    shipping_data = pd.read_csv(snakemake.input["shipping_data"], index_col=0)

    # map exported carrier to labels in the shipping database
    shipping_index = {
        "LH2": "H2 (l)",
        "NH3": "NH3 (l)",
        "MEOH": "MeOH",
        "oil": "FT"
    }

    # selection of ship transport data
    shipping_data_transport = shipping_data.loc[f"{shipping_index[exp_carrier]} transport ship"]
    shipping_data_transport = shipping_data_transport.set_index("variable")

    if fuel_export == "oil":
        shipping_data_fuel = shipping_data.loc["FT fuel transport ship"]
        shipping_data_fuel = shipping_data_fuel.set_index("variable")
    else: 
        fuel_export = exp_carrier
        shipping_data_fuel=shipping_data_transport

    # determine shipping route
    shipping_route = get_shipping_route(nodes_with_port, destination)
    shipping_distance = shipping_route.distance
    shipping_hour = shipping_route.distance/shipping_data_fuel.loc["average speed", "value"]

    # ship loading
    # add export bus for ship loading
    n.madd(
        "Bus",
        nodes_to_connect + " load ship export",
        carrier=exp_carrier + " export",
        location=nodes_to_connect
    )

    # add export link for ship loading
    n.madd(
        "Link",
        nodes_to_connect + " load ship export",
        bus0=nodes_to_connect + " export",
        bus1=nodes_to_connect + " load ship export",
        carrier=exp_carrier + " export",
        p_nom=1e7,
        efficiency=1-shipping_data_transport.loc["(un-) loading losses", "value"]/100,
    )

    # ship transport
    # add export bus for ship transport
    n.madd(
        "Bus",
        nodes_to_connect + " transport ship export",
        carrier=exp_carrier + " export",
        location=nodes_to_connect
    )

    # add export link for ship transport
    # calculate efficiency due to boil off
    efficiency_boil_off = (1-shipping_data_transport.loc["boil-off", "value"]/100)**shipping_hour
    efficiency_boil_off.index = nodes_to_connect + " transport ship export"
    
    n.madd(
            "Link",
            nodes_to_connect + " transport ship export",
            bus0=nodes_to_connect + " load ship export",
            bus1=nodes_to_connect + " transport ship export",
            carrier=exp_carrier + " export",
            p_nom=1e7, 
            efficiency=efficiency_boil_off,
        )

    # add carrier and bus for fuel ship loading
    n.add("Carrier", fuel_export + " fuel ship export")

    n.madd(
        "Bus",
        nodes_to_connect + " fuel ship export",
        carrier=fuel_export + " fuel ship export",
        location=nodes_to_connect
    )

    # determine required amount of ships
    ship_capacity = shipping_data_transport.loc["capacity", "value"]
    ships_required = get_ships_required(export_volume, ship_capacity, shipping_hour)

    # determine regional fuel amount based on port fraction
    ports_fuel_export_demand = pd.DataFrame(shipping_data_fuel.loc["energy demand", "value"] * 2 * shipping_distance).T # Double check conversion
    ports_fuel_export_demand = ports_fuel_export_demand * ships_required.max() * (port_fraction/port_fraction.sum())

    # adjust port bus in accordance to fuel carrier
    if fuel_export in ["LH2", "NH3", "MEOH"]:
        fuel_nodes_to_connect = nodes_to_connect
    elif fuel_export == "oil":
        fuel_nodes_to_connect = nodes_to_connect.str.replace(exp_carrier, fuel_export)
    elif fuel_nodes_to_connect[fuel_nodes_to_connect.isin(n.buses.index)].empty:
            print(f"Warning: None of the {fuel_export} buses exist!")

    # create region fuel profile    
    ports_fuel_export_profil = ports_fuel_export_demand / 8760
    snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
    
    ports_fuel_export_profil = ports_fuel_export_profil.squeeze()
    ports_fuel_export_profil = pd.DataFrame([ports_fuel_export_profil] * len(snapshots), index=snapshots).astype(float)
    ports_fuel_export_profil.columns = fuel_nodes_to_connect + " fuel ship export"

    # add load for fuel ship
    n.madd(
        "Load",
        fuel_nodes_to_connect + " fuel ship export",
        bus=fuel_nodes_to_connect,
        carrier=fuel_export + " fuel ship export", 
        p_set=ports_fuel_export_profil,
    )

    # consider emissions from fuel ship
    if fuel_export in ["oil", "MEOH"]:
        # match label from costs
        co2_intensity_carrier = {
            "oil": "oil",
            "MEOH": "methanol"
        }

        # identify emissions due to fuel ship
        co2 = (
            ports_fuel_export_profil.sum()
            * costs.at[co2_intensity_carrier[fuel_export], "CO2 intensity"]
        ).sum()
        co2 = co2 / 8760

        # add load for fuel ship emissions
        n.add(
            "Load",
            f"export shipping {fuel_export} emissions",
            bus="co2 atmosphere",
            carrier=fuel_export + " fuel ship export",
            p_set=-co2,
        )

    # add export bus for ship unloading
    n.madd(
        "Bus",
        nodes_to_connect + " unload ship export",
        carrier=exp_carrier + " export",
        location=nodes_to_connect
    )

    # add export link for ship unloading
    n.madd(
        "Link",
        nodes_to_connect + " unload ship export",
        bus0=nodes_to_connect + " transport ship export",
        bus1=nodes_to_connect + " unload ship export",
        carrier=exp_carrier + " export",
        p_nom=1e7, 
        efficiency=1-shipping_data_transport.loc["(un-) loading losses", "value"]/100,
    )

    # consider costs for shipping
    # add bus for ship costs
    n.madd(
        "Bus",
        nodes_to_connect + " costs ship export",
        carrier=exp_carrier + " export",
        location=nodes_to_connect
    )

    # determine ship investment costs based on export volume
    capital_cost = calculate_annuity(costs, discountrate).loc[f"{shipping_index[exp_carrier]} transport ship"] # Unit: €
    capital_cost = capital_cost * ships_required.max() / export_volume  # Unit: €/MWh 

    # add export link for ship costs
    n.madd(
        "Link",
        nodes_to_connect + " costs ship export",
        bus0=nodes_to_connect + " unload ship export",
        bus1=nodes_to_connect + " costs ship export",
        carrier=exp_carrier + " export",
        p_nom=1e7, 
        efficiency=1,
        marginal_cost=capital_cost, 
    )

    # add export bus for import port
    n.add(
        "Bus",
        exp_carrier + " crossborder export",
        carrier=exp_carrier + " export",
        location=nodes_to_connect
    )

    # add export link for import port
    n.madd(
        "Link",
        nodes_to_connect + " crossborder export",
        bus0=nodes_to_connect + " costs ship export",
        bus1=exp_carrier + " crossborder export",
        carrier=exp_carrier + " export",
        p_nom=1e7, 
        efficiency=1,
        marginal_cost=-price, 
    )

    # add export conversion for import port
    # add export conversion carrier and bus for import port
    export_destination_carrier = snakemake.params.export_destination_carrier
    if export_destination_carrier == "H2":
        n.add("Carrier", export_destination_carrier + " destination carrier export")

        n.add(
            "Bus",
            export_destination_carrier + " destination carrier export",
            carrier=export_destination_carrier + " destination carrier export",
        )

        if exp_carrier == "LH2":
            # Source: 10.1109/EEM64765.2025.11050281
            n.add(
                "Link",
                exp_carrier + " evaporation export",
                bus0=exp_carrier + " crossborder export",
                bus1=export_destination_carrier + " destination carrier export",
                p_nom_extendable=True,
                carrier=export_destination_carrier + " destination carrier export",
                efficiency= 1 / export_reconversion_LH2_H2, # Source: "Heuser et al. 2019: https://doi.org/10.1016/j.ijhydene.2018.12.156, table 1; 10.1109/EEM64765.2025.11050281
                # Assumption: Electricity consumption neglected
                capital_cost=costs.at["H2 evaporation", "fixed"] / export_reconversion_LH2_H2, 
                lifetime=costs.at["H2 evaporation", "lifetime"],
            )

            profile = profile / export_reconversion_LH2_H2

        elif exp_carrier == "NH3":
            # Source: pypsa-eur
            n.add(
                "Link",
                exp_carrier + " ammonia cracker export",
                bus0=exp_carrier + " crossborder export",
                bus1=export_destination_carrier + " destination carrier export",
                # Assumption: electricity assumption neglected
                p_nom_extendable=True,
                carrier=export_destination_carrier + " destination carrier export",
                efficiency=1 / export_reconversion_NH3_H2,
                capital_cost=costs.at["Ammonia cracker", "fixed"]
                / export_reconversion_NH3_H2,  # given per MW_H2
                lifetime=costs.at["Ammonia cracker", "lifetime"],
            )

            profile = profile / export_reconversion_NH3_H2

        elif exp_carrier == "MEOH":
            n.add(
                "Link",
                exp_carrier + "Methanol steam reforming",
                bus0=exp_carrier + " crossborder export",
                bus1=export_destination_carrier + " destination carrier export",
                #bus2="co2 atmosphere", # already considered
                p_nom_extendable=True,
                capital_cost=costs.at["Methanol steam reforming", "fixed"] / export_reconversion_MeOH_H2,
                efficiency=1 / export_reconversion_MeOH_H2,
                #efficiency2=costs.at["methanolisation", "carbondioxide-input"], # already considered
                carrier=export_destination_carrier + " destination carrier export",
                lifetime=costs.at["Methanol steam reforming", "lifetime"],
            )

            profile = profile / export_reconversion_MeOH_H2

    else: 
        export_destination_carrier = exp_carrier

        n.add("Carrier", export_destination_carrier + " destination carrier export")

        n.add(
            "Bus",
            export_destination_carrier + " destination carrier export",
            carrier=export_destination_carrier + " destination carrier export",
        )

        n.add(
                "Link",
                exp_carrier + f" destination carrier export",
                bus0=exp_carrier + " crossborder export",
                bus1=export_destination_carrier + " destination carrier export",
                p_nom=1e7, 
                efficiency=1,
                carrier=export_destination_carrier + " destination carrier export",
            )

    n.add(
            "Load",
            export_destination_carrier + " destination carrier export",
            bus=export_destination_carrier + " destination carrier export",
            carrier=export_destination_carrier + " destination carrier export",
            p_set=profile,
        )


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

if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_export",
            simpl="",
            clusters="12",
            ll="vopt",
            opts="Co2L0.27-144H",
            planning_horizons="2050",
            sopts="144H",
            discountrate=0.09,
            demand="NZ",
            eopts="NH3v25",
            configfile="/home/alex-charly/SSD/H2GMA/Github/AP10/analyse-h2g-a-ap10/config/supply-scenarios/config.MA_2050.yaml",
        )

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    countries = list(n.buses.country.unique())

    price_factor, volume_factor = parse_eopts(snakemake.wildcards["eopts"])
    export_carriers = list(set(price_factor.keys()).union(set(volume_factor.keys())))
    logger.info(f"The following export carriers will be considered: {export_carriers}")

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
        snakemake.params.costs,
        snakemake.params.costs["output_currency"],
        snakemake.params.costs["fill_values"],
        Nyears,
    )

    # select and define nodes for export via port and shipping 
    nodes_with_port, port_fraction = select_ports(n)

    # consideration of reconversion
    export_reconversion_LH2_H2 = snakemake.params.export_reconversion_LH2_H2 # MWh_LH2/MWh_H2
    export_reconversion_NH3_H2 = snakemake.params.export_reconversion_NH3_H2 # MWh_NH3/MWh_H2
    export_reconversion_MeOH_H2 = snakemake.params.export_reconversion_MEOH_H2 # MWh_MeOH/MWh_H2

    # add export values and components to network for each export carrier
    for export_carrier in export_carriers:
        if snakemake.params.export_destination_carrier:
            if export_carrier == "LH2":
                export_volume = export_volumes[export_carrier] * export_reconversion_LH2_H2
            elif export_carrier == "NH3":
                export_volume = export_volumes[export_carrier] * export_reconversion_NH3_H2
            elif export_carrier == "MEOH":
                export_volume = export_volumes[export_carrier] * export_reconversion_MeOH_H2
        else:
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

        add_export(n, export_carrier, export_volume, export_price, export_profile, nodes_with_port, port_fraction, costs, snakemake)



    n.export_to_netcdf(snakemake.output[0])

    logger.info("Network successfully exported")