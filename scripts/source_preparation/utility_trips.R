#' Generate EV-focused Origin-Destination (OD) data for trip purposes
#' 
#' This function has been adapted from the original cycling analysis to generate
#' OD data specifically for Electric Vehicle (EV) analysis. Unlike cycling trips
#' which focus on short distances and specific infrastructure, EV analysis considers
#' longer distances, car mode share, and charging infrastructure implications.
#'
#' KEY EV ADAPTATIONS:
#' 1. Extended trip purpose coverage (14 EV-relevant purposes vs 3 cycling purposes)
#' 2. Updated destination mappings for EV-appropriate POI categories
#' 3. Longer distance considerations (up to 100km+ vs cycling 1-20km)
#' 4. Focus on car mode share as baseline for EV adoption potential
#' 5. Social trip destinations use residential zones (people's homes) vs POI locations
#'
#' @param oas Output Area Bounds - residential origins for trips
#' @param os_pois OS Points of Interest - destination locations by category
#' @param grid Spatial grid for aggregating destinations
#' @param purpose Trip purpose string - one of 14 EV-relevant categories
#' @param trip_purposes Scottish Household Survey trip purpose data
#' @param zones Administrative zones (DataZones) with population data  
#' @param parameters Analysis parameters including distance thresholds
#' @return sf object with OD pairs, mode shares, and trip volumes for EV analysis
#'
#' @details
#' EV-RELEVANT TRIP PURPOSES (14 total):
#' Primary (high EV adoption potential):
#'   - commuting: Daily work trips (highest EV use case)
#'   - business: Commercial/work-related travel
#'   - education: School/university trips
#'   - shopping: Retail and errands (frequent car trips)
#'   - visit hospital or other health: Essential medical trips
#'   - other personal business: Errands requiring vehicle capacity
#'   - visiting friends or relatives: Social car trips
#'   - eating/drinking: Dining out trips
#'   - sport/entertainment: Recreation venues
#'   - holiday/daytrip: Tourism destinations
#'   - escort: Driving others (family trips)
#' Secondary (lower EV priority):
#'   - other journey: Miscellaneous car trips
#'
#' DESTINATION MAPPING FOR EV ANALYSIS:
#' - Commuting/Business → Commercial Services (workplaces)
#' - Education/Health → Education and Health facilities
#' - Shopping → Retail locations
#' - Social/Dining → Accommodation, Eating and Drinking
#' - Recreation → Sport and Entertainment / Attractions
#' - Transport → Transport hubs
#' - Miscellaneous → Public Infrastructure
make_od_ev = function(oas, os_pois, grid, purpose, trip_purposes, zones, parameters,
                      distance_frequency_file = "inputdata/distance_frequency_wide.csv") {
  
  # ======================================================================
  # EV-SPECIFIC DESTINATION MAPPING
  # ======================================================================
  # Maps EV trip purposes to appropriate OS POI groupnames.
  # Unlike cycling analysis which only handled 3 purposes (shopping, leisure, visiting),
  # EV analysis requires mappings for 14 trip purposes covering all major car trip types.
  # Each mapping considers where people actually drive cars for that purpose.
  # ======================================================================
  groupname_mappings = list(
    # Original cycling analysis purposes (adapted for EV)
    "shopping" = "Retail",                                    # EV: Shopping centers, retail parks
    "leisure" = "Sport and Entertainment",                    # EV: Recreation venues accessible by car
    
    # NEW EV-specific work/education purposes  
    "commuting" = "Commercial Services",                      # EV: Office buildings, business parks
    "business" = "Commercial Services",                       # EV: Client sites, business meetings
    "education" = "Education and Health",                     # EV: Schools, universities (car access)
    
    # NEW EV-specific essential trip purposes
    "visit hospital or other health" = "Education and Health", # EV: Hospitals, clinics (car essential)
    "other personal business" = "Commercial Services",        # EV: Banks, offices, services
    
    # NEW EV-specific social trip purposes  
    "visiting friends or relatives" = "Accommodation, Eating and Drinking", # EV: Social venues, restaurants
    "eating/drinking" = "Accommodation, Eating and Drinking", # EV: Restaurants, pubs (car for convenience)
    
    # NEW EV-specific recreation purposes
    "sport/entertainment" = "Sport and Entertainment",        # EV: Stadiums, cinemas, gyms
    "holiday/daytrip" = "Attractions",                       # EV: Tourist sites, attractions (car for luggage)
    
    # NEW EV-specific miscellaneous purposes
    "other journey" = "Public Infrastructure",               # EV: General car-accessible destinations
    "escort" = "Transport"                                   # EV: Transport hubs, drop-off points
  )
  
  # Define purpose name mappings for trip_purposes lookup
  purpose_nts_mappings = list(
    "shopping" = "Shopping",
    "leisure" = "Sport/Entertainment",
    "commuting" = "Commuting",
    "business" = "Business",
    "education" = "Education",
    "visit hospital or other health" = "Visit Hospital or other health",
    "other personal business" = "Other personal business",
    "visiting friends or relatives" = "Visiting friends or relatives",
    "eating/drinking" = "Eating/Drinking",
    "sport/entertainment" = "Sport/Entertainment",
    "holiday/daytrip" = "Holiday/daytrip",
    "other journey" = "Other Journey",
    "escort" = "Escort"
  )
  
  # Get the appropriate groupname and purpose_nts for this purpose
  purpose_lower = tolower(purpose)
  groupname = groupname_mappings[[purpose_lower]]
  purpose_nts = purpose_nts_mappings[[purpose_lower]]
  
  # Default fallback if mapping not found
  if (is.null(groupname)) {
    groupname = "Retail"  # Default fallback
    warning(paste("No groupname mapping found for purpose:", purpose, "- using 'Retail' as default"))
  }
  if (is.null(purpose_nts)) {
    purpose_nts = "Shopping"  # Default fallback
    warning(paste("No purpose_nts mapping found for purpose:", purpose, "- using 'Shopping' as default"))
  }
  
  # Filter POIs by groupname
  os_pois = os_pois |>
    dplyr::filter(groupname == !!groupname)
  os_pois = os_pois |>
    sf::st_transform(sf::st_crs(grid))

  # Special handling for certain purposes
  if (purpose_lower == "leisure") {
    # add in park points for leisure
    parks = make_parks(grid)
    os_pois = bind_rows(
      os_pois |> transmute(ref_no = as.character(ref_no)),
      parks |> transmute(ref_no = id)
    )
  }

  os_pois = os_pois |>
    dplyr::mutate(grid_id = sf::st_nearest_feature(os_pois, grid))

  # calculate weighting of each grid point
  p_grid = os_pois |>
    sf::st_drop_geometry() |>
    dplyr::group_by(grid_id) |>
    dplyr::summarise(size = n())
  p_grid = sf::st_as_sf(p_grid, geometry = grid[p_grid$grid_id])
  p_grid = sf::st_transform(p_grid, 4326)

  # ======================================================================
  # EV TRIP GENERATION - ADAPTED FROM CYCLING TO EV ANALYSIS
  # ======================================================================
  # Calculate trip generation rates for EV analysis. Unlike cycling analysis
  # which focuses on bicycle mode share, EV analysis focuses on car trips
  # that could potentially switch to electric vehicles.
  # ======================================================================
  
  # Get trip purpose proportion from Scottish Household Survey data
  proportion_all_distances = trip_purposes |>
    dplyr::filter(tolower(Purpose) == tolower(purpose_nts)) |>
    dplyr::transmute(proportion = adjusted_mean / 100) |>
    dplyr::pull(proportion)
  
  # Zone population data for trip generation
  zones_p = zones |>
    dplyr::select(DataZone, ResPop2011)
  
  # Daily trip rate from National Travel Survey 2019 (England)
  # EV Analysis Note: 953 trips/person/year = 2.61 trips/day (same as cycling analysis)
  # This includes ALL trip purposes; we'll filter to car trips later for EV adoption
  total_trips_per_day = 2.61
  
  # ======================================================================
  # EV DISTANCE FREQUENCY MAPPING 
  # ======================================================================
  # Maps EV trip purposes to existing distance frequency categories for
  # spatial interaction modeling. EV analysis can handle longer distances
  # than cycling (up to 100km+ vs cycling 1-20km), but we use similar
  # patterns for consistency with the existing methodology.
  # ======================================================================
  distance_purpose_mappings = list(
    # Original cycling purposes (same mapping for EV)
    "shopping" = "Shopping",                                  # EV: Retail trip patterns  
    "leisure" = "Leisure",                                    # EV: Recreation trip patterns
    
    # NEW EV work/education purposes - use commuting patterns
    "commuting" = "Commuting",                                # EV: Work trip patterns (direct mapping)
    "business" = "Commuting",                                 # EV: Use work patterns (business = work-related)
    "education" = "Commuting",                                # EV: Use work patterns (regular daily trips)
    
    # NEW EV essential purposes - use shopping patterns (frequent, local trips)
    "visit hospital or other health" = "Shopping",            # EV: Medical trips similar to errands
    "other personal business" = "Shopping",                   # EV: Errands similar to shopping trips
    
    # NEW EV social purposes  
    "visiting friends or relatives" = "Social",               # EV: Social trip patterns
    "eating/drinking" = "Leisure",                           # EV: Use leisure patterns (evening/weekend)
    
    # NEW EV recreation purposes - use leisure patterns
    "sport/entertainment" = "Leisure",                       # EV: Recreation patterns (direct mapping)
    "holiday/daytrip" = "Leisure",                          # EV: Use leisure patterns (longer trips)
    
    # NEW EV miscellaneous purposes
    "other journey" = "Shopping",                            # EV: Default to shopping patterns 
    "escort" = "Commuting"                                   # EV: Use commuting patterns (regular family trips)
  )
  
  purpose2 = distance_purpose_mappings[[purpose_lower]]
  if (is.null(purpose2)) {
    purpose2 = "Shopping"  # Default fallback
    warning(paste("No distance purpose mapping found for purpose:", purpose, "- using 'Shopping' as default"))
  }
  
  distance_frequency = readr::read_csv(distance_frequency_file, show_col_types = FALSE)
  proportion_in_od = distance_frequency |>
    dplyr::filter(tolower(`NPT purpose`) == tolower(purpose2)) |>
    dplyr::select(`0-1 km`, `1-2 km`, `2-5 km`, `5-10 km`, `10-15 km`, `15-20 km`, `20-30 km`, `30-40 km`, `40-50 km`, `50-60 km`, `60-70 km`, `70-80 km`, `80-90 km`, `90-100 km`, `100+ km`) |>
    sum()
  proportion = proportion_all_distances * proportion_in_od
  
  # DIAGNOSTIC: Print values to understand the calculation
  cat("=== DIAGNOSTIC OUTPUT ===\n")
  cat("Purpose: ", purpose, "\n")
  cat("  proportion_all_distances: ", round(proportion_all_distances, 4), "\n")
  cat("  proportion_in_od: ", round(proportion_in_od, 4), "\n")
  cat("  combined proportion: ", round(proportion, 4), "\n")
  cat("========================\n")
  flush.console()
  
  # Trip generation: ALL modes (do not filter to car-only here)
  zones_p = zones_p |>
    dplyr::mutate(p_trips = ResPop2011 * total_trips_per_day * proportion) |>
    dplyr::select(-ResPop2011)
  zones_p = sf::st_transform(zones_p, 4326)
  zones_p = sf::st_make_valid(zones_p)

  # Spatial interaction model of journeys
  max_length_euclidean_km = 40
  
  # Define which purposes use zones as destinations (social/visiting purposes)
  # Note: Both "visiting" and "visiting friends or relatives" now use POI-based destinations
  # (social venues, restaurants) rather than residential zones
  social_purposes = c()
  
  if (purpose_lower %in% social_purposes) {
    # For social/visiting purposes, destinations are residential zones (people's homes)
    od_p_initial = simodels::si_to_od(zones_p, zones_p, max_dist = max_length_euclidean_km * 1000) |>
      dplyr::rename(destination_size = destination_p_trips)
  } else {
    # For all other purposes, destinations are POI-based grid points
    od_p_initial = simodels::si_to_od(zones_p, p_grid, max_dist = max_length_euclidean_km * 1000)
  }
  beta_p = distance_frequency |>
    dplyr::filter(tolower(`NPT purpose`) == tolower(purpose2)) |>
    dplyr::pull(beta)

  od_interaction = od_p_initial |>
    simodels::si_calculate(
      fun = gravity_model,
      m = origin_p_trips,
      n = destination_size,
      d = distance_euclidean,
      beta = beta_p,
      constraint_production = origin_p_trips
    )
  generation_ledger = data.frame(
    stage = c("origin_production", "candidate_normalisation"),
    rows = c(nrow(zones_p), nrow(od_interaction)),
    all_mode_person_trips = c(sum(zones_p$p_trips), sum(od_interaction$interaction))
  )
  # TODO: set min_per_o as a parameter?
  # Keep only max_per_o destinations per origin
  min_per_o = 5
  min_p = min(zones_p$p_trips)
  od_interaction_filtered = purrr::map_dfr(
    unique(od_interaction$O),
    ~ {
      od_o = od_interaction |>
        dplyr::filter(O == .x)
      n_destinations = round(od_o$origin_p_trips[1] / min_p * min_per_o)
      od_o |>
        dplyr::slice_sample(n = n_destinations, weight_by = interaction)
    }
  )
  od_adjusted = od_interaction_filtered |>
    dplyr::group_by(O) |>
    dplyr::mutate(
      proportion = interaction / sum(interaction),
      p_all_modes = origin_p_trips * proportion
    ) |>
    dplyr::ungroup()
  summary(od_adjusted$p_all_modes)
  generation_ledger = rbind(generation_ledger, data.frame(
    stage = "sampled_and_renormalised", rows = nrow(od_adjusted),
    all_mode_person_trips = sum(od_adjusted$p_all_modes)
  ))
  # sum(od_adjusted$p_all_modes) / sum(zones_p$p_trips) # close to 1

  # Jittering
  if (purpose_lower %in% social_purposes) {
    zones_centroids = zones_p |>
      sf::st_centroid()
    oas = c(sf::st_geometry(oas), sf::st_geometry(zones_centroids))
    od_adjusted_jittered = odjitter::jitter(
      od = od_adjusted,
      zones = zones_p,
      subpoints = oas,
      disaggregation_key = "p_all_modes",
      disaggregation_threshold = parameters$disag_threshold,
      deduplicate_pairs = FALSE
    )
  } else {
    using_s2 = sf::sf_use_s2()
    sf::sf_use_s2(TRUE)
    p_polygons = sf::st_buffer(p_grid, dist = 250)
    sf::sf_use_s2(using_s2)
    p_grid_full = dplyr::bind_rows(
      p_grid |> dplyr::mutate(grid_id = as.character(grid_id)),
      os_pois |> dplyr::transmute(grid_id = as.character(ref_no), size = 1)
    )
    od_adjusted_jittered = odjitter::jitter(
      od = od_adjusted,
      zones = zones_p,
      zones_d = p_polygons, # each polygon is a single grid point, so destinations are kept the same
      subpoints_origins = oas,
      subpoints_destinations = p_grid_full,
      disaggregation_key = "p_all_modes",
      disaggregation_threshold = parameters$disag_threshold,
      deduplicate_pairs = FALSE
    )
  }

  # ======================================================================
  # EV MODE SHARE ANALYSIS - KEY ADAPTATION FROM CYCLING
  # ======================================================================
  # Apply mode shares to generate baseline car trips for EV adoption analysis.
  # CRITICAL EV DIFFERENCE: Unlike cycling analysis which focuses on bicycle 
  # mode share growth, EV analysis focuses on the CAR mode share as the baseline
  # for potential EV adoption. The car trips generated here represent the
  # maximum pool of trips that could switch to electric vehicles.
  # ======================================================================

  # Mode shares from Scottish Household Survey Travel Diaries 
  # Source: transport-and-travel-in-scotland-2019-local-authority-tables.xlsx, table 16
  # EV Analysis Focus: The 65.2% car mode share represents our target for EV conversion
  # car = driver + passenger (both could potentially use EVs)
  # public_transport = bus + rail (not directly relevant for EV analysis)
  # taxi = taxi + other (could include ride-sharing EVs in future)
  mode_shares = tibble::tibble(
    bicycle = 0.012,        # 1.2% - cycling baseline (not EV relevant)
    foot = 0.221,           # 22.1% - walking trips (not EV relevant)  
    car = 0.652,            # 65.2% - KEY for EV analysis (potential conversion pool)
    public_transport = 0.093, # 9.3% - public transport (could complement EVs)
    taxi = 0.022            # 2.2% - taxi/other (potential EV taxi/rideshare)
  )

  generation_ledger = rbind(generation_ledger, data.frame(
    stage = "jittered", rows = nrow(od_adjusted_jittered),
    all_mode_person_trips = sum(od_adjusted_jittered$p_all_modes)
  ))

  od_p_jittered = od_adjusted_jittered |>
    dplyr::rename(
      geo_code1 = O,
      geo_code2 = D
    ) |>
    dplyr::mutate(
      # Assign ALL modes using SHS mode shares (car/bicycle/foot/PT/taxi)
      bicycle = p_all_modes * mode_shares$bicycle,
      foot = p_all_modes * mode_shares$foot,
      car = p_all_modes * mode_shares$car,
      public_transport = p_all_modes * mode_shares$public_transport,
      taxi = p_all_modes * mode_shares$taxi
    )

  od_p_subset = od_p_jittered |>
    dplyr::rename(length_euclidean_unjittered = distance_euclidean) |>
    dplyr::mutate(
      length_euclidean_unjittered = length_euclidean_unjittered / 1000,
      length_euclidean_jittered = units::drop_units(st_length(od_p_jittered)) / 1000
    ) |>
    dplyr::filter(
      length_euclidean_jittered > (parameters$min_distance_meters / 1000),
      length_euclidean_jittered < max_length_euclidean_km
    )
  n_short_lines_removed = nrow(od_p_jittered) - nrow(od_p_subset)
  message(n_short_lines_removed, " short or long desire lines removed")

  od_p_subset = od_p_subset |>
    dplyr::rename(
      origin_trips = origin_p_trips,
      all = p_all_modes
    ) |>
    mutate(purpose = purpose)

  generation_ledger = rbind(generation_ledger, data.frame(
    stage = "distance_filtered", rows = nrow(od_p_subset),
    all_mode_person_trips = sum(od_p_subset$all)
  ))
  generation_ledger$car_person_trips = generation_ledger$all_mode_person_trips * mode_shares$car
  attr(od_p_subset, "generation_ledger") = generation_ledger
  attr(od_p_subset, "distance_specification") = list(
    purpose = purpose, pattern = purpose2, beta_km_inverse = beta_p,
    coefficient_file = distance_frequency_file,
    candidate_euclidean_km = max_length_euclidean_km,
    minimum_jittered_km = parameters$min_distance_meters / 1000,
    population_column = "ResPop2011", daily_person_trip_rate = total_trips_per_day,
    car_person_mode_share = mode_shares$car,
    equation = "m * n * exp(beta * distance_m / 1000)"
  )
  od_p_subset
}

make_parks = function(grid) {
  park_points = sf::st_read("inputdata/park_points.gpkg")
  study_area = sf::st_convex_hull(sf::st_union(grid))
  park_points = sf::st_transform(park_points, sf::st_crs(grid))
  park_points[study_area, ]
}
