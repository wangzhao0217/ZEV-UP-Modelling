# Speed profile pipeline extracted from get_speed_profile.qmd.
# The functions below preserve the original route segmentation and speed-profile
# generation logic while exposing it as reusable code for Quarto reports/scripts.

library(sf)
library(dplyr)
library(jsonlite)
library(stringr)
library(lwgeom)
library(osmactive)
library(stplanr)

DEFAULT_SPEED_PROFILE_ROUTE_FILE <- "data/OD_routes/swestrans_routes_tl.geojson"
DEFAULT_SPEED_PROFILE_REGION <- "Scotland"
DEFAULT_URBAN_RURAL_FILE <- "data/SG_UrbanRural_2022/SG_UrbanRural_2022.shp"
MAX_ACCELERATION_MPH_PER_SEC <- 60 / 25
MAX_ACCELERATION_MS2 <- MAX_ACCELERATION_MPH_PER_SEC * 0.44704
MAX_SPEED_FOR_BRAKING <- 60 * 0.44704
BRAKING_DISTANCE_M <- 30
MAX_DECELERATION_MS2 <- (MAX_SPEED_FOR_BRAKING^2) / (2 * BRAKING_DISTANCE_M)
STOPPING_TYPES_REQUIRE_STOP <- c("traffic_light", "sharp_turn", "roundabout")

#' Ensure a route table exposes a stable integer identifier column.
#' @param route_data sf/data.frame with route attributes.
#' @param id_col output identifier column name.
#' @param candidates ordered list of source columns to use before falling back
#'   to row order.
#' @return input data with `id_col` populated as integer values.
ensure_identifier_column <- function(route_data,
                                     id_col = "od_id",
                                     candidates = c("od_id", "route_od_id", "fid")) {
  candidate_cols <- unique(c(id_col, candidates))
  candidate_cols <- candidate_cols[candidate_cols %in% names(route_data)]

  ids <- rep(NA_integer_, nrow(route_data))

  if (length(candidate_cols) > 0) {
    for (candidate_col in candidate_cols) {
      candidate_ids <- suppressWarnings(as.integer(route_data[[candidate_col]]))

      if (length(candidate_ids) == nrow(route_data) && !anyNA(candidate_ids)) {
        ids <- candidate_ids
        break
      }
    }
  }

  if (anyNA(ids)) {
    ids <- seq_len(nrow(route_data))
  }

  route_data[[id_col]] <- ids
  route_data
}

#' Prepare sf features for GeoPackage export.
#' @param sf_data sf/data.frame with route attributes.
#' @param integer_columns identifier columns that should be stored as integers.
#' @param drop_columns source columns to remove before writing.
#' @return cleaned sf object safe to write with `st_write(..., driver = "GPKG")`.
sanitize_gpkg_features <- function(sf_data,
                                   integer_columns = c("route_od_id", "od_id"),
                                   drop_columns = "fid") {
  cleaned <- sf_data
  columns_to_drop <- intersect(drop_columns, names(cleaned))

  if (length(columns_to_drop) > 0) {
    cleaned <- cleaned[, setdiff(names(cleaned), columns_to_drop), drop = FALSE]
  }

  columns_to_integer <- intersect(integer_columns, names(cleaned))

  for (column_name in columns_to_integer) {
    cleaned[[column_name]] <- suppressWarnings(as.integer(cleaned[[column_name]]))
  }

  cleaned
}

#' Parse traffic-light coordinates stored as JSON.
#' @param coord_string JSON string from the route file.
#' @return data.frame with lat, lon, osm_id columns.
parse_traffic_light_coords <- function(coord_string) {
  empty_coords <- data.frame(lat = numeric(0), lon = numeric(0), osm_id = numeric(0))

  if (is.null(coord_string) || length(coord_string) == 0) {
    return(empty_coords)
  }

  coord_string <- coord_string[[1]]

  if (length(coord_string) == 0 || is.null(coord_string) || is.na(coord_string)) {
    return(empty_coords)
  }

  coord_string <- as.character(coord_string)

  if (length(coord_string) == 0 || is.na(coord_string) || trimws(coord_string) == "") {
    return(empty_coords)
  }

  tryCatch({
    clean_string <- gsub('\\"', '"', coord_string)
    coords_list <- fromJSON(clean_string)

    if (length(coords_list) > 0) {
      if (is.data.frame(coords_list)) {
        return(data.frame(
          lat = as.numeric(coords_list$lat),
          lon = as.numeric(coords_list$lon),
          osm_id = as.numeric(coords_list$osm_id)
        ))
      }

      if (is.list(coords_list)) {
        return(data.frame(
          lat = sapply(coords_list, function(x) as.numeric(x$lat)),
          lon = sapply(coords_list, function(x) as.numeric(x$lon)),
          osm_id = sapply(coords_list, function(x) as.numeric(x$osm_id))
        ))
      }
    }

    empty_coords
  }, error = function(e) {
    empty_coords
  })
}

#' Cluster nearby traffic lights to avoid duplicate stopping points.
#' @param tl_coords data.frame with lat/lon columns.
#' @param distance_threshold clustering threshold in metres.
#' @return clustered traffic-light data.frame.
cluster_traffic_lights <- function(tl_coords, distance_threshold = 20) {
  if (nrow(tl_coords) <= 1) {
    return(tl_coords)
  }

  tl_sf <- st_as_sf(tl_coords, coords = c("lon", "lat"), crs = 4326)
  tl_sf <- st_transform(tl_sf, 27700)
  dist_matrix <- st_distance(tl_sf)

  clusters <- rep(NA, nrow(tl_coords))
  cluster_id <- 1

  for (i in seq_len(nrow(tl_coords))) {
    if (is.na(clusters[i])) {
      nearby <- which(as.numeric(dist_matrix[i, ]) <= distance_threshold)
      clusters[nearby] <- cluster_id
      cluster_id <- cluster_id + 1
    }
  }

  tl_coords %>%
    mutate(cluster = clusters) %>%
    group_by(cluster) %>%
    summarise(
      lat = mean(lat),
      lon = mean(lon),
      osm_id = paste(osm_id, collapse = ","),
      count = n(),
      .groups = "drop"
    )
}

#' Identify sharp turns and roundabouts from consecutive route segments.
#' @param route_sf segmented route sf object.
#' @param angle_threshold angle threshold in degrees.
#' @param roundabout_keywords keywords used to identify roundabouts.
#' @return data.frame describing detected turns.
identify_sharp_turns <- function(route_sf, angle_threshold = 45,
                                 roundabout_keywords = c("roundabout", "circular")) {
  route_sf$bearing <- line_bearing(route_sf, bidirectional = TRUE)

  sharp_turns <- list()
  turn_id <- 1
  has_highway_info <- "highway_drivenet" %in% names(route_sf)

  for (i in seq_len(nrow(route_sf) - 1)) {
    current_segment <- route_sf[i, ]
    next_segment <- route_sf[i + 1, ]

    angle_diff <- current_segment$bearing - next_segment$bearing
    angle_change <- abs(angle_diff)
    is_sharp_turn <- angle_change >= angle_threshold

    is_roundabout <- FALSE
    if (has_highway_info) {
      highway_type <- tolower(as.character(current_segment$highway_drivenet))
      is_roundabout <- any(vapply(
        roundabout_keywords,
        function(kw) grepl(kw, highway_type),
        logical(1)
      ))
    }

    if (is_sharp_turn || is_roundabout) {
      segment_coords <- st_coordinates(current_segment$geometry)
      turn_coords <- segment_coords[nrow(segment_coords), , drop = FALSE]

      turn_type <- "moderate_turn"
      if (is_roundabout) {
        turn_type <- "roundabout"
      } else if (angle_change >= 80 && angle_change <= 110) {
        turn_type <- "sharp_turn"
      }

      sharp_turns[[turn_id]] <- data.frame(
        turn_id = turn_id,
        segment_id = i,
        lat = turn_coords[1, "Y"],
        lon = turn_coords[1, "X"],
        angle_change = angle_change,
        is_roundabout = is_roundabout,
        turn_type = turn_type
      )
      turn_id <- turn_id + 1
    }
  }

  if (length(sharp_turns) == 0) {
    return(data.frame(
      turn_id = integer(0),
      segment_id = integer(0),
      lat = numeric(0),
      lon = numeric(0),
      angle_change = numeric(0),
      is_roundabout = logical(0),
      turn_type = character(0)
    ))
  }

  bind_rows(sharp_turns)
}

#' Find the position of a point along a route line.
#' @param linestring sfc LINESTRING geometry.
#' @param point_lat latitude/y coordinate.
#' @param point_lon longitude/x coordinate.
#' @return numeric distance along the line in metres.
find_position_along_line <- function(linestring, point_lat, point_lon) {
  tryCatch({
    if (is.na(st_crs(linestring))) {
      st_crs(linestring) <- 27700
    }

    point_sf <- st_sfc(st_point(c(point_lon, point_lat)), crs = 4326)
    point_sf <- st_transform(point_sf, st_crs(linestring))

    position <- as.numeric(st_line_locate_point(linestring, point_sf))
    total_length <- as.numeric(st_length(linestring))

    position * total_length
  }, error = function(e) {
    tryCatch({
      if (is.na(st_crs(linestring))) {
        st_crs(linestring) <- 27700
      }

      point_sf <- st_sfc(st_point(c(point_lon, point_lat)), crs = 4326)
      point_sf <- st_transform(point_sf, st_crs(linestring))
      nearest_point <- st_nearest_points(point_sf, linestring)[1]

      line_coords <- st_coordinates(linestring)[, 1:2]
      nearest_coords <- st_coordinates(nearest_point)[, 1:2]

      best_position <- 0
      cumulative_dist <- 0

      for (i in seq_len(nrow(line_coords) - 1)) {
        seg_start <- line_coords[i, ]
        seg_end <- line_coords[i + 1, ]
        seg_length <- sqrt(sum((seg_end - seg_start)^2))

        dist_to_start <- sqrt(sum((nearest_coords[1, ] - seg_start)^2))
        dist_to_end <- sqrt(sum((nearest_coords[1, ] - seg_end)^2))

        if (dist_to_start <= seg_length && dist_to_end <= seg_length) {
          best_position <- cumulative_dist + dist_to_start
          break
        }

        cumulative_dist <- cumulative_dist + seg_length
      }

      best_position
    }, error = function(e2) {
      NA_real_
    })
  })
}

#' Merge segmented route geometry into a single ordered line.
#' @param route_segments segmented sf route.
#' @return sfc LINESTRING geometry.
create_route_line_geometry <- function(route_segments) {
  if (nrow(route_segments) == 0) {
    return(st_sfc())
  }

  merged <- st_union(st_geometry(route_segments))
  merged <- st_line_merge(merged)

  if (inherits(merged, "sfc_GEOMETRYCOLLECTION")) {
    merged <- lwgeom::st_collection_extract(merged, "LINESTRING")
  }

  if (length(merged) > 1) {
    merged <- merged[1]
  }

  st_set_crs(merged, st_crs(route_segments))
}

#' Extract a sub-geometry between cumulative route distances.
#' @param route_line merged route line geometry.
#' @param start_distance starting distance along route.
#' @param end_distance ending distance along route.
#' @param total_length total route length in metres.
#' @return sfc LINESTRING geometry.
extract_part_geometry <- function(route_line, start_distance, end_distance, total_length) {
  if (length(route_line) == 0 || total_length <= 0 || start_distance >= end_distance) {
    return(route_line)
  }

  start_fraction <- max(0, min(1, start_distance / total_length))
  end_fraction <- max(0, min(1, end_distance / total_length))

  if (abs(end_fraction - start_fraction) < 1e-6) {
    end_fraction <- min(1, start_fraction + 1e-6)
  }

  part_geom <- suppressWarnings(
    lwgeom::st_linesubstring(route_line, from = start_fraction, to = end_fraction)
  )

  if (length(part_geom) == 0 || all(st_is_empty(part_geom))) {
    return(route_line)
  }

  st_set_crs(part_geom, st_crs(route_line))
}

#' Load the Scotland driving network used in the original notebook.
#' @param region OSM region name.
#' @return sf driving network.
load_driving_network <- function(region = DEFAULT_SPEED_PROFILE_REGION) {
  osm <- osmactive::get_travel_network(region)
  drive_net_major <- osmactive::get_driving_network(osm)
  drive_net_major <- osmactive::clean_speeds(drive_net_major)
  sf::st_cast(drive_net_major, "LINESTRING") |> sf::st_transform(27700)
}

#' Join 20m route segments to the driving network speed data.
#' @param routes segmented sf route object.
#' @param drive_net sf driving network.
#' @param buffer_dist buffer distance in metres.
#' @param join_dist join distance in metres.
#' @return sf object containing route segments with speed attributes.
join_routes_with_speeds <- function(routes, drive_net, buffer_dist = 10, join_dist = 2) {
  routes <- ensure_identifier_column(
    route_data = routes,
    id_col = "od_id",
    candidates = c("od_id", "route_od_id", "fid")
  )

  route_buffer <- routes |>
    st_buffer(dist = buffer_dist) |>
    st_union()

  drive_net_clipped <- drive_net[sf::st_union(route_buffer), , op = sf::st_intersects]

  routes_joined <- stplanr::rnet_join(
    rnet_x = routes,
    rnet_y = drive_net_clipped |>
      transmute(
        maxspeed_drivenet = maxspeed_clean,
        highway_drivenet = highway
      ) |>
      sf::st_cast(to = "LINESTRING"),
    key_column = "od_id",
    dist = join_dist,
    segment_length = 10
  )

  if (!"od_id" %in% names(routes_joined)) {
    stop(
      "rnet_join did not preserve od_id. Available columns: ",
      paste(names(routes_joined), collapse = ", ")
    )
  }

  routes_with_speeds <- routes_joined |>
    sf::st_drop_geometry() |>
    group_by(od_id) |>
    summarise(
      maxspeed_drivenet = osmactive:::most_common_value(maxspeed_drivenet),
      highway_drivenet = osmactive:::most_common_value(highway_drivenet),
      .groups = "drop"
    ) |>
    mutate(
      maxspeed_drivenet = case_when(
        is.na(maxspeed_drivenet) ~ NA_real_,
        maxspeed_drivenet == "" ~ NA_real_,
        str_detect(maxspeed_drivenet, "^\\d+$") ~ as.numeric(maxspeed_drivenet),
        str_detect(maxspeed_drivenet, "^\\d+\\s*mph$") ~ as.numeric(str_extract(maxspeed_drivenet, "\\d+")),
        str_detect(maxspeed_drivenet, "^\\d+\\s*km/h$") ~ as.numeric(str_extract(maxspeed_drivenet, "\\d+")),
        maxspeed_drivenet %in% c("variable", "none", "signals") ~ NA_real_,
        TRUE ~ suppressWarnings(as.numeric(maxspeed_drivenet))
      )
    )

  left_join(routes, routes_with_speeds, by = "od_id")
}

#' Load the route, segment it, and attach driving-network speed data.
#' @param route_od_id route identifier to process.
#' @param route_file path to the GeoJSON route file.
#' @param region region used when loading the OSM driving network.
#' @param drive_net optional preloaded driving network to reuse across routes.
#' @return sf object matching the original notebook's `rnet_tl_joined`.
load_route_with_speed_data <- function(route_od_id = 3266,
                                       route_file = DEFAULT_SPEED_PROFILE_ROUTE_FILE,
                                       region = DEFAULT_SPEED_PROFILE_REGION,
                                       drive_net = NULL) {
  route_file_ext <- tolower(tools::file_ext(route_file))

  if (route_file_ext == "gpkg") {
    available_layers <- sf::st_layers(route_file)

    if (nrow(available_layers) == 0) {
      stop("No layers found in ", route_file)
    }

    layer_name <- available_layers$name[[1]]
    route_queries <- c(
      sprintf('SELECT * FROM "%s" WHERE route_od_id = %d', layer_name, as.integer(route_od_id)),
      sprintf('SELECT * FROM "%s" WHERE od_id = %d', layer_name, as.integer(route_od_id))
    )

    rnet_tl <- NULL

    for (route_query in route_queries) {
      rnet_tl <- tryCatch(
        sf::read_sf(route_file, query = route_query, quiet = TRUE),
        error = function(e) NULL
      )

      if (!is.null(rnet_tl) && nrow(rnet_tl) > 0) {
        break
      }
    }

    if (is.null(rnet_tl) || nrow(rnet_tl) == 0) {
      stop("No route found for route_od_id == ", route_od_id, " in ", route_file)
    }
  } else {
    rnet_tl <- sf::read_sf(route_file)
  }

  rnet_tl <- ensure_identifier_column(
    route_data = rnet_tl,
    id_col = "od_id",
    candidates = c("od_id", "route_od_id", "fid")
  )
  rnet_tl <- sf::st_cast(rnet_tl, "LINESTRING") |> sf::st_transform(27700)
  rnet_tl <- rnet_tl |>
    dplyr::group_by(od_id) |>
    dplyr::mutate(.part_id = dplyr::row_number()) |>
    dplyr::ungroup()
  rnet_tl <- stplanr::line_segment(rnet_tl, segment_length = 100)
  rnet_tl_sample <- rnet_tl |> filter(od_id == route_od_id)

  if (nrow(rnet_tl_sample) == 0) {
    stop("No route found for od_id == ", route_od_id)
  }

  rnet_tl_sample_50m <- rnet_tl_sample |>
    stplanr::line_segment(segment_length = 20)

  rnet_tl_sample_50m$od <- paste0(
    rnet_tl_sample_50m$od_id, "_p", rnet_tl_sample_50m$.part_id
  )
  rnet_tl_sample_50m <- rnet_tl_sample_50m |> mutate(od_id = seq_len(nrow(rnet_tl_sample_50m)))

  if (is.null(drive_net)) {
    drive_net <- load_driving_network(region = region)
  } else if (sf::st_crs(drive_net) != sf::st_crs(rnet_tl_sample_50m)) {
    drive_net <- sf::st_transform(drive_net, sf::st_crs(rnet_tl_sample_50m))
  }

  join_routes_with_speeds(rnet_tl_sample_50m, drive_net, buffer_dist = 10, join_dist = 2)
}

#' Build stopping points and route break points from the joined route network.
#' @param rnet_tl_joined route network with speed and traffic-light attributes.
#' @return list containing break_points_df and original_route_with_cumulative.
build_break_points <- function(rnet_tl_joined) {
  all_network_tl <- list()
  has_traffic_light_coords <- "traffic_light_coordinates" %in% names(rnet_tl_joined)

  for (i in seq_len(nrow(rnet_tl_joined))) {
    route_row <- rnet_tl_joined[i, ]

    if (has_traffic_light_coords) {
      tl_coords <- parse_traffic_light_coords(route_row[["traffic_light_coordinates"]])

      if (nrow(tl_coords) > 0) {
        tl_coords$source_od_id <- route_row$od_id
        all_network_tl[[i]] <- tl_coords
      }
    }
  }

  rnet_with_global_idx <- rnet_tl_joined
  rnet_with_global_idx$.global_idx <- seq_len(nrow(rnet_with_global_idx))
  all_sharp_turns <- rnet_with_global_idx |>
    split(rnet_with_global_idx$od) |>
    lapply(function(part) {
      turns <- identify_sharp_turns(part, angle_threshold = 45)
      if (nrow(turns) > 0) {
        turns$segment_id <- part$.global_idx[turns$segment_id]
      }
      turns
    }) |>
    dplyr::bind_rows()
  if (nrow(all_sharp_turns) > 0) {
    all_sharp_turns$turn_id <- seq_len(nrow(all_sharp_turns))
    all_sharp_turns <- all_sharp_turns %>%
      filter(is_roundabout | (angle_change >= 45 & angle_change <= 155))
  }

  break_points <- list()
  break_point_counter <- 1

  rnet_with_cumulative <- rnet_tl_joined %>%
    mutate(original_row_idx = row_number()) %>%
    group_by(od) %>%
    mutate(
      segment_length = as.numeric(st_length(geometry)),
      cumulative_distance = cumsum(segment_length),
      segment_index_in_route = row_number()
    ) %>%
    ungroup()

  original_route_with_cumulative <- rnet_with_cumulative
  route_lengths <- original_route_with_cumulative %>%
    group_by(od) %>%
    summarise(route_length = max(cumulative_distance), .groups = "drop")
  route_length_lookup <- setNames(route_lengths$route_length, route_lengths$od)

  if (nrow(all_sharp_turns) > 0) {
    for (i in seq_len(nrow(all_sharp_turns))) {
      turn <- all_sharp_turns[i, ]
      segment_idx <- turn$segment_id
      turn_segment <- original_route_with_cumulative %>%
        filter(original_row_idx == segment_idx)

      if (nrow(turn_segment) == 1) {
        turn_position <- turn_segment$cumulative_distance
        od_identifier <- turn_segment$od
        route_total_length <- route_lengths %>%
          filter(od == od_identifier) %>%
          pull(route_length)

        if (length(route_total_length) > 0 && !is.na(route_total_length)) {
          break_points[[break_point_counter]] <- data.frame(
            stop_id = break_point_counter,
            route_idx = turn_segment$original_row_idx,
            od_id = od_identifier,
            position_along_route = turn_position,
            route_length = route_total_length,
            distance_to_route = 0,
            stopping_type = turn$turn_type,
            stopping_id = paste0("TURN_", turn$turn_id),
            angle_change = turn$angle_change,
            lat = turn$lat,
            lon = turn$lon
          )
          break_point_counter <- break_point_counter + 1
        }
      }
    }
  }

  if (length(all_network_tl) > 0) {
    all_tl_df <- bind_rows(all_network_tl)
    unique_tl <- all_tl_df %>%
      group_by(osm_id) %>%
      slice_head(n = 1) %>%
      ungroup()
    clustered_tl <- cluster_traffic_lights(unique_tl, distance_threshold = 20)
    tl_sf <- st_as_sf(clustered_tl, coords = c("lon", "lat"), crs = 4326)
    tl_sf <- st_transform(tl_sf, st_crs(rnet_tl_joined))

    for (i in seq_len(nrow(tl_sf))) {
      tl_point <- tl_sf[i, ]
      nearest_segment_idx <- st_nearest_feature(tl_point, rnet_with_cumulative)
      min_distance <- as.numeric(
        st_distance(tl_point, rnet_with_cumulative[nearest_segment_idx, ])
      )

      if (min_distance < 10) {
        matching_segment <- rnet_with_cumulative[nearest_segment_idx, ]
        position <- matching_segment$cumulative_distance
        route_total_length <- route_length_lookup[[as.character(matching_segment$od)]]

        if (!is.null(route_total_length) && !is.na(route_total_length)) {
          tl_coords_matrix <- st_coordinates(tl_point)
          break_points[[break_point_counter]] <- data.frame(
            stop_id = break_point_counter,
            route_idx = matching_segment$original_row_idx,
            od_id = matching_segment$od,
            position_along_route = position,
            route_length = route_total_length,
            distance_to_route = min_distance,
            stopping_type = "traffic_light",
            stopping_id = paste0("TL_", i),
            angle_change = NA,
            lat = tl_coords_matrix[1, "Y"],
            lon = tl_coords_matrix[1, "X"]
          )
          break_point_counter <- break_point_counter + 1
        }
      }
    }
  }

  if (length(break_points) == 0) {
    break_points_df <- data.frame()
  } else {
    break_points_df <- bind_rows(break_points) %>%
      mutate(requires_stop = stopping_type %in% STOPPING_TYPES_REQUIRE_STOP)
  }

  list(
    break_points_df = break_points_df,
    original_route_with_cumulative = original_route_with_cumulative
  )
}

#' Split the route into route parts at the computed break points.
#' @param rnet_tl_joined joined route network.
#' @param break_points_df break-point table.
#' @param original_route_with_cumulative route segments with cumulative distance.
#' @return sf object containing route parts.
build_route_parts <- function(rnet_tl_joined, break_points_df, original_route_with_cumulative) {
  if (nrow(break_points_df) == 0) {
    return(
      rnet_tl_joined %>%
        mutate(
          part_id = row_number(),
          original_od_id = od,
          segment_start = 0,
          segment_end = as.numeric(st_length(geometry)),
          segment_length = segment_end,
          has_break_start = TRUE,
          has_break_end = TRUE,
          route_part_id = part_id
        )
    )
  }

  break_by_route <- break_points_df %>%
    arrange(od_id, position_along_route) %>%
    group_by(od_id) %>%
    summarise(
      break_points = list(
        pick(everything()) %>%
          select(position_along_route, stopping_type, requires_stop) %>%
          distinct(position_along_route, .keep_all = TRUE)
      ),
      break_count = n(),
      .groups = "drop"
    )

  unique_routes <- rnet_tl_joined %>%
    group_by(od) %>%
    summarise(
      total_length = sum(as.numeric(st_length(geometry))),
      .groups = "drop"
    )

  all_parts <- list()
  current_part_id <- 1

  for (i in seq_len(nrow(unique_routes))) {
    current_od <- unique_routes$od[i]
    route_total_length <- unique_routes$total_length[i]

    route_segments <- rnet_tl_joined %>% filter(od == current_od)
    route_segments_with_cumulative <- original_route_with_cumulative %>%
      filter(od == current_od) %>%
      mutate(segment_start_cum = cumulative_distance - segment_length)

    route_line <- create_route_line_geometry(route_segments)

    if (length(route_line) == 0 || all(st_is_empty(route_line))) {
      next
    }

    route_length_from_geom <- as.numeric(st_length(route_line))
    route_total_length_effective <- route_total_length
    if (is.na(route_total_length_effective) || route_total_length_effective <= 0) {
      route_total_length_effective <- route_length_from_geom
    }

    route_breaks <- break_by_route %>% filter(od_id == current_od)

    if (nrow(route_breaks) > 0) {
      breaks_df <- route_breaks$break_points[[1]] %>%
        arrange(position_along_route)
      positions <- breaks_df$position_along_route
      requires_stop_flags <- breaks_df$requires_stop
      segment_starts <- c(0, positions)
      segment_ends <- c(positions, route_total_length_effective)

      for (j in seq_along(segment_starts)) {
        if (segment_ends[j] > segment_starts[j]) {
          part_data <- route_segments[1, ]
          part_data$part_id <- current_part_id
          part_data$original_od_id <- current_od
          part_data$segment_start <- segment_starts[j]
          part_data$segment_end <- segment_ends[j]
          part_data$segment_length <- segment_ends[j] - segment_starts[j]
          start_requires_stop <- if (j == 1) FALSE else requires_stop_flags[j - 1]
          end_requires_stop <- if (j <= length(requires_stop_flags)) requires_stop_flags[j] else FALSE
          part_data$has_break_start <- start_requires_stop
          part_data$has_break_end <- end_requires_stop

          overlapping_segments <- route_segments_with_cumulative %>%
            filter(
              segment_start_cum < segment_ends[j],
              cumulative_distance > segment_starts[j]
            )

          if (nrow(overlapping_segments) > 0) {
            part_data$maxspeed_drivenet <- osmactive:::most_common_value(overlapping_segments$maxspeed_drivenet)
            part_data$highway_drivenet <- osmactive:::most_common_value(overlapping_segments$highway_drivenet)
            part_data$avg_speed_limit <- mean(overlapping_segments$maxspeed_drivenet, na.rm = TRUE)
            part_data$speed_zone_count <- overlapping_segments %>%
              filter(!is.na(maxspeed_drivenet)) %>%
              pull(maxspeed_drivenet) %>%
              unique() %>%
              length()
          } else {
            part_data$avg_speed_limit <- part_data$maxspeed_drivenet
            part_data$speed_zone_count <- ifelse(is.na(part_data$maxspeed_drivenet), 0, 1)
          }

          part_geom <- extract_part_geometry(
            route_line,
            segment_starts[j],
            segment_ends[j],
            route_total_length_effective
          )
          st_geometry(part_data) <- part_geom

          all_parts[[current_part_id]] <- part_data
          current_part_id <- current_part_id + 1
        }
      }
    } else {
      part_data <- route_segments[1, ]
      part_data$part_id <- current_part_id
      part_data$original_od_id <- current_od
      part_data$segment_start <- 0
      part_data$segment_end <- route_total_length_effective
      part_data$segment_length <- route_total_length_effective
      part_data$has_break_start <- FALSE
      part_data$has_break_end <- FALSE

      overlapping_segments <- route_segments_with_cumulative
      if (nrow(overlapping_segments) > 0) {
        part_data$maxspeed_drivenet <- osmactive:::most_common_value(overlapping_segments$maxspeed_drivenet)
        part_data$highway_drivenet <- osmactive:::most_common_value(overlapping_segments$highway_drivenet)
        part_data$avg_speed_limit <- mean(overlapping_segments$maxspeed_drivenet, na.rm = TRUE)
        part_data$speed_zone_count <- overlapping_segments %>%
          filter(!is.na(maxspeed_drivenet)) %>%
          pull(maxspeed_drivenet) %>%
          unique() %>%
          length()
      } else {
        part_data$avg_speed_limit <- part_data$maxspeed_drivenet
        part_data$speed_zone_count <- ifelse(is.na(part_data$maxspeed_drivenet), 0, 1)
      }

      st_geometry(part_data) <- route_line
      all_parts[[current_part_id]] <- part_data
      current_part_id <- current_part_id + 1
    }
  }

  bind_rows(all_parts) %>%
    group_by(original_od_id) %>%
    arrange(segment_start, .by_group = TRUE) %>%
    mutate(
      is_first_part = row_number() == 1,
      is_last_part = row_number() == n(),
      has_break_start = ifelse(is_first_part, TRUE, has_break_start),
      has_break_end = ifelse(is_last_part, TRUE, has_break_end)
    ) %>%
    ungroup() %>%
    select(-is_first_part, -is_last_part) %>%
    mutate(route_part_id = part_id)
}

#' Generate a single-segment speed profile.
#' @param segment_length segment length in metres.
#' @param speed_limit_mph speed limit in mph.
#' @param resolution output resolution in metres.
#' @return data.frame containing distance and speed columns.
generate_speed_profile <- function(segment_length, speed_limit_mph, resolution = 1) {
  speed_limit_ms <- speed_limit_mph * 0.44704
  if (is.na(speed_limit_ms) || speed_limit_ms <= 0) {
    speed_limit_ms <- 30 * 0.44704
  }

  accel_distance <- (speed_limit_ms^2) / (2 * MAX_ACCELERATION_MS2)
  brake_distance <- (speed_limit_ms^2) / (2 * MAX_DECELERATION_MS2)
  distances <- seq(0, segment_length, by = resolution)
  speeds <- numeric(length(distances))

  for (i in seq_along(distances)) {
    d <- distances[i]
    distance_to_end <- segment_length - d

    if (segment_length <= (accel_distance + brake_distance)) {
      midpoint <- segment_length / 2

      if (d <= midpoint) {
        speeds[i] <- sqrt(2 * MAX_ACCELERATION_MS2 * d)
      } else {
        max_speed_at_midpoint <- sqrt(2 * MAX_ACCELERATION_MS2 * midpoint)
        brake_dist_from_midpoint <- d - midpoint
        speeds[i] <- sqrt(
          max_speed_at_midpoint^2 - 2 * MAX_DECELERATION_MS2 * brake_dist_from_midpoint
        )
      }

      speeds[i] <- min(speeds[i], speed_limit_ms)
    } else {
      if (d <= accel_distance) {
        speeds[i] <- sqrt(2 * MAX_ACCELERATION_MS2 * d)
      } else if (distance_to_end <= brake_distance) {
        speeds[i] <- sqrt(2 * MAX_DECELERATION_MS2 * distance_to_end)
      } else {
        speeds[i] <- speed_limit_ms
      }
    }

    speeds[i] <- max(0, min(speeds[i], speed_limit_ms))
  }

  speeds[1] <- 0
  speeds[length(speeds)] <- 0

  data.frame(
    distance = distances,
    speed_ms = speeds,
    speed_mph = speeds / 0.44704,
    segment_length = segment_length,
    speed_limit_mph = speed_limit_mph,
    accel_distance = accel_distance,
    brake_distance = brake_distance
  )
}

#' Generate a speed profile for a route part, preserving the original notebook logic.
#' @param route_part_data sf/data.frame rows for a single route part.
#' @param original_route_with_cumulative original 20m route segments with cumulative distances.
#' @param initial_speed_ms speed at the start of the part in m/s.
#' @return list with profile, segments, speed_zones, metadata, and speed continuity info.
generate_route_part_profile <- function(route_part_data, original_route_with_cumulative,
                                        initial_speed_ms = 0) {
  part_meta <- route_part_data %>% slice(1)
  part_start <- part_meta$segment_start
  part_end <- part_meta$segment_end
  part_length <- part_meta$segment_length
  part_od <- part_meta$original_od_id

  segments_raw <- original_route_with_cumulative %>%
    filter(od == part_od) %>%
    mutate(
      seg_start = cumulative_distance - segment_length,
      seg_end = cumulative_distance
    ) %>%
    filter(
      seg_end > part_start,
      seg_start < part_end
    ) %>%
    mutate(
      segment_start = pmax(seg_start, part_start) - part_start,
      segment_end = pmin(seg_end, part_end) - part_start,
      segment_length = segment_end - segment_start,
      has_break_start = part_meta$has_break_start & segment_start == 0,
      has_break_end = part_meta$has_break_end & segment_end == part_length
    ) %>%
    filter(segment_length > 0)

  if (nrow(segments_raw) == 0) {
    segments_raw <- part_meta %>%
      mutate(
        segment_start = 0,
        segment_end = part_length,
        segment_length = part_length
      )
  }

  segments <- segments_raw %>%
    arrange(segment_start) %>%
    mutate(
      speed_limit_clean = ifelse(
        is.na(maxspeed_drivenet) | maxspeed_drivenet <= 0,
        30,
        maxspeed_drivenet
      )
    )

  total_length <- part_length
  most_common_speed <- as.numeric(names(sort(table(segments$speed_limit_clean), decreasing = TRUE))[1])

  segments$speed_group <- cumsum(c(
    TRUE,
    segments$speed_limit_clean[-1] !=
      segments$speed_limit_clean[-length(segments$speed_limit_clean)]
  ))

  speed_groups <- segments %>%
    group_by(speed_group, speed_limit_clean) %>%
    summarise(
      group_length = sum(segment_length),
      start_pos = min(segment_start),
      .groups = "drop"
    ) %>%
    arrange(start_pos)

  significant_groups <- speed_groups %>%
    filter(group_length >= 500 | speed_limit_clean == most_common_speed)

  if (nrow(significant_groups) == 0) {
    significant_groups <- data.frame(
      speed_group = 1,
      speed_limit_clean = most_common_speed,
      group_length = total_length,
      start_pos = 0
    )
  }

  speed_zones <- data.frame(
    start_position = c(0, cumsum(significant_groups$group_length)[-nrow(significant_groups)]),
    end_position = cumsum(significant_groups$group_length),
    speed_limit = significant_groups$speed_limit_clean
  )

  distances <- seq(0, total_length, by = 5)
  speeds_ms <- numeric(length(distances))
  has_break_at_start <- any(segments$has_break_start)
  has_break_at_end <- any(segments$has_break_end)
  starting_speed_ms <- if (has_break_at_start) 0 else initial_speed_ms
  dt <- 0.5
  current_speed_ms <- starting_speed_ms

  for (i in seq_along(distances)) {
    d <- distances[i]

    current_zone <- speed_zones[
      speed_zones$start_position <= d & speed_zones$end_position > d,
    ]
    if (nrow(current_zone) == 0) {
      current_zone <- tail(speed_zones, 1)
    }
    target_speed_ms <- current_zone$speed_limit[1] * 0.44704
    distance_to_end <- total_length - d

    brake_distance_end <- if (current_speed_ms > 0) {
      (current_speed_ms^2) / (2 * MAX_DECELERATION_MS2)
    } else {
      0
    }

    upcoming_zones <- speed_zones[speed_zones$start_position > d, ]
    need_to_brake_for_zone_change <- FALSE

    if (nrow(upcoming_zones) > 0) {
      next_zone <- upcoming_zones[1, ]
      distance_to_zone_change <- next_zone$start_position - d
      next_speed_ms <- next_zone$speed_limit * 0.44704

      if (current_speed_ms > next_speed_ms) {
        brake_distance_zone <- (current_speed_ms^2 - next_speed_ms^2) /
          (2 * MAX_DECELERATION_MS2)
        if (distance_to_zone_change <= brake_distance_zone + 50) {
          need_to_brake_for_zone_change <- TRUE
          target_speed_ms <- next_speed_ms
        }
      }
    }

    if (d == 0) {
      speeds_ms[i] <- current_speed_ms
      next
    } else if (has_break_at_end && distance_to_end <= brake_distance_end + 30) {
      required_decel <- (current_speed_ms^2) / (2 * max(distance_to_end, 1))
      required_decel <- min(required_decel, MAX_DECELERATION_MS2)
      new_speed_squared <- current_speed_ms^2 - 2 * required_decel * 5
      current_speed_ms <- max(0, sqrt(max(0, new_speed_squared)))
      speeds_ms[i] <- current_speed_ms
    } else if (need_to_brake_for_zone_change) {
      speed_diff_squared <- current_speed_ms^2 - target_speed_ms^2
      if (speed_diff_squared > 0) {
        decel_rate <- min(MAX_DECELERATION_MS2 * 0.7, speed_diff_squared / (2 * 5))
        new_speed_squared <- current_speed_ms^2 - 2 * decel_rate * 5
        current_speed_ms <- max(target_speed_ms, sqrt(max(0, new_speed_squared)))
      } else {
        current_speed_ms <- target_speed_ms
      }
      speeds_ms[i] <- current_speed_ms
    } else {
      if (current_speed_ms < target_speed_ms) {
        accel_rate <- min(MAX_ACCELERATION_MS2, (target_speed_ms - current_speed_ms) / dt)
        current_speed_ms <- min(target_speed_ms, current_speed_ms + accel_rate * dt)
      } else if (current_speed_ms > target_speed_ms) {
        decel_rate <- min(
          MAX_DECELERATION_MS2 * 0.5,
          (current_speed_ms - target_speed_ms) / dt
        )
        current_speed_ms <- max(target_speed_ms, current_speed_ms - decel_rate * dt)
      }
      speeds_ms[i] <- current_speed_ms
    }

    speeds_ms[i] <- max(0, min(speeds_ms[i], target_speed_ms))
  }

  if (length(speeds_ms) > 5) {
    window_size <- 5
    smoothed_speeds <- numeric(length(speeds_ms))

    for (i in seq_along(speeds_ms)) {
      start_idx <- max(1, i - window_size %/% 2)
      end_idx <- min(length(speeds_ms), i + window_size %/% 2)
      smoothed_speeds[i] <- mean(speeds_ms[start_idx:end_idx])
    }

    speeds_ms <- smoothed_speeds
  }

  if (has_break_at_start) {
    speeds_ms[1] <- 0
  }
  if (has_break_at_end) {
    speeds_ms[length(speeds_ms)] <- 0
  }

  profile_df <- data.frame(
    distance = distances,
    speed_ms = speeds_ms,
    speed_mph = speeds_ms / 0.44704,
    total_length = total_length
  )

  profile_df$speed_limit_mph <- sapply(profile_df$distance, function(d) {
    zone <- speed_zones[speed_zones$start_position <= d & speed_zones$end_position > d, ]
    if (nrow(zone) == 0) {
      tail(speed_zones$speed_limit, 1)
    } else {
      zone$speed_limit[1]
    }
  })

  part_metadata <- part_meta %>%
    st_drop_geometry() %>%
    select(
      route_part_id,
      original_od_id,
      segment_start,
      segment_end,
      segment_length,
      has_break_start,
      has_break_end
    ) %>%
    distinct()

  list(
    profile = profile_df,
    segments = segments,
    speed_zones = speed_zones,
    metadata = part_metadata,
    has_break_at_start = has_break_at_start,
    has_break_at_end = has_break_at_end,
    start_speed_ms = starting_speed_ms,
    end_speed_ms = tail(speeds_ms, 1)
  )
}

#' Generate route-part speed profiles while preserving speed continuity.
#' @param rnet_with_parts route-part sf object.
#' @param original_route_with_cumulative original 20m segments with cumulative distances.
#' @return list of route-part profile objects.
generate_route_part_profiles <- function(rnet_with_parts, original_route_with_cumulative) {
  route_parts <- rnet_with_parts %>%
    arrange(route_part_id) %>%
    group_by(route_part_id) %>%
    group_split()

  route_part_profiles <- vector("list", length(route_parts))
  previous_end_speed_ms <- 0

  for (i in seq_along(route_parts)) {
    route_part_data <- route_parts[[i]] %>% ungroup()
    part_has_break_start <- any(route_part_data$has_break_start)
    part_initial_speed <- if (part_has_break_start) 0 else previous_end_speed_ms

    profile_result <- generate_route_part_profile(
      route_part_data,
      original_route_with_cumulative,
      initial_speed_ms = part_initial_speed
    )
    route_part_profiles[[i]] <- profile_result
    previous_end_speed_ms <- if (profile_result$has_break_at_end) {
      0
    } else {
      profile_result$end_speed_ms
    }
  }

  names(route_part_profiles) <- paste0("part_", seq_along(route_part_profiles))
  route_part_profiles
}

#' Compute the notebook-style trip speed distribution from combined profiles.
#' @param combined_profiles output of `compile_speed_profile_tables()$combined_profiles`.
#' @return data.frame matching speed_distribution_trip.csv.
compute_speed_distribution_trip <- function(combined_profiles) {
  dist_per_part <- combined_profiles %>%
    arrange(route_part_id, distance_within_part) %>%
    group_by(route_part_id) %>%
    mutate(segment_length_m = lead(distance_within_part) - distance_within_part) %>%
    ungroup() %>%
    filter(!is.na(segment_length_m) & segment_length_m > 0) %>%
    mutate(speed_kmh = speed_ms * 3.6)

  breaks <- seq(0, 100, by = 10)
  labels <- paste0(breaks[-length(breaks)], "-", breaks[-1], " km/h")

  speed_distribution <- dist_per_part %>%
    mutate(
      speed_bin = cut(
        speed_kmh,
        breaks = breaks,
        labels = labels,
        include.lowest = TRUE,
        right = TRUE
      )
    )

  if (any(dist_per_part$speed_kmh > 100, na.rm = TRUE)) {
    speed_distribution <- speed_distribution %>%
      mutate(
        speed_bin = ifelse(speed_kmh > 100, ">=100 km/h", as.character(speed_bin)),
        speed_bin = factor(speed_bin, levels = c(labels, ">=100 km/h"))
      )
  }

  trip_total_m <- sum(dist_per_part$segment_length_m)
  all_bins <- levels(speed_distribution$speed_bin)

  speed_distribution %>%
    group_by(speed_bin) %>%
    summarise(distance_m = sum(segment_length_m), .groups = "drop") %>%
    right_join(tibble::tibble(speed_bin = factor(all_bins, levels = all_bins)), by = "speed_bin") %>%
    mutate(
      distance_m = tidyr::replace_na(distance_m, 0),
      proportion = if (trip_total_m > 0) distance_m / trip_total_m else 0,
      proportion_percent = round(proportion * 100, 2)
    ) %>%
    arrange(speed_bin)
}

#' Compute a simple urban/rural speed distribution table.
#' @param combined_profiles speed-profile point table with `urban_rural_class`.
#' @return data.frame for all_speed_profiles_simple.csv.
compute_speed_distribution_simple_by_area <- function(combined_profiles) {
  dist_per_part <- combined_profiles %>%
    arrange(route_part_id, distance_within_part) %>%
    group_by(route_part_id) %>%
    mutate(segment_length_m = lead(distance_within_part) - distance_within_part) %>%
    ungroup() %>%
    filter(!is.na(segment_length_m) & segment_length_m > 0) %>%
    mutate(
      speed_kmh = speed_ms * 3.6,
      UR8Name = ifelse(
        is.na(UR8Name) | trimws(UR8Name) == "",
        "Unknown",
        UR8Name
      ),
      urban_rural_class = ifelse(
        is.na(urban_rural_class) | trimws(urban_rural_class) == "",
        "Unknown",
        urban_rural_class
      )
    )

  breaks <- seq(0, 100, by = 10)
  labels <- paste0(breaks[-length(breaks)], "-", breaks[-1], " km/h")

  dist_per_part <- dist_per_part %>%
    mutate(
      speed_bin = cut(
        speed_kmh,
        breaks = breaks,
        labels = labels,
        include.lowest = TRUE,
        right = TRUE
      )
    )

  if (any(dist_per_part$speed_kmh > 100, na.rm = TRUE)) {
    dist_per_part <- dist_per_part %>%
      mutate(
        speed_bin = ifelse(speed_kmh > 100, ">=100 km/h", as.character(speed_bin)),
        speed_bin = factor(speed_bin, levels = c(labels, ">=100 km/h"))
      )
  }

  all_bins <- levels(dist_per_part$speed_bin)
  simple_area_levels <- c("Urban", "Rural")
  ur8_area_levels <- sort(unique(dist_per_part$UR8Name[dist_per_part$UR8Name != "Unknown"]))
  area_levels <- c(simple_area_levels, ur8_area_levels)

  area_long <- bind_rows(
    dist_per_part %>%
      filter(urban_rural_class %in% simple_area_levels) %>%
      transmute(area_label = urban_rural_class, speed_bin, segment_length_m),
    dist_per_part %>%
      filter(UR8Name %in% ur8_area_levels) %>%
      transmute(area_label = UR8Name, speed_bin, segment_length_m)
  )

  area_totals <- area_long %>%
    group_by(area_label) %>%
    summarise(total_distance_m = sum(segment_length_m), .groups = "drop")

  area_speed_distribution <- area_long %>%
    group_by(area_label, speed_bin) %>%
    summarise(distance_m = sum(segment_length_m), .groups = "drop") %>%
    right_join(
      tidyr::expand_grid(
        area_label = area_levels,
        speed_bin = factor(all_bins, levels = all_bins)
      ),
      by = c("area_label", "speed_bin")
    ) %>%
    mutate(distance_m = tidyr::replace_na(distance_m, 0)) %>%
    left_join(area_totals, by = "area_label") %>%
    mutate(
      share_percent = ifelse(
        is.na(total_distance_m) | total_distance_m <= 0,
        0,
        round(distance_m / total_distance_m * 100, 2)
      )
    ) %>%
    mutate(area_label = factor(area_label, levels = area_levels)) %>%
    arrange(speed_bin, area_label)

  area_speed_distribution %>%
    select(speed_bin, area_label, share_percent) %>%
    tidyr::pivot_wider(
      names_from = area_label,
      values_from = share_percent,
      values_fill = 0,
      names_glue = "{area_label} Share (%)"
    ) %>%
    rename(`Speed range (km/h)` = speed_bin)
}

#' Add UR8Name classification to each speed-profile point.
#' @param combined_profiles speed-profile point table.
#' @param rnet_with_parts route-part sf object.
#' @param urban_rural_file path to the urban/rural shapefile.
#' @param map_class_col shapefile column used for classification.
#' @return `combined_profiles` with an added classification column.
classify_speed_profiles_by_map_class <- function(combined_profiles, rnet_with_parts,
                                                 urban_rural_file = DEFAULT_URBAN_RURAL_FILE,
                                                 map_class_col = "UR8Name",
                                                 output_col = "urban_rural_class") {
  urban_rural_shp <- st_read(urban_rural_file, quiet = TRUE)

  if (!map_class_col %in% names(urban_rural_shp)) {
    stop("Column '", map_class_col, "' not found in the urban/rural shapefile.")
  }

  if (st_crs(urban_rural_shp) != st_crs(rnet_with_parts)) {
    urban_rural_shp <- st_transform(urban_rural_shp, st_crs(rnet_with_parts))
  }

  route_part_geoms <- rnet_with_parts %>%
    select(route_part_id, geometry) %>%
    st_as_sf()

  profile_rows <- combined_profiles %>%
    mutate(row_id = seq_len(n()))

  profile_points_list <- list()
  point_counter <- 1

  route_part_groups <- profile_rows %>%
    group_by(route_part_id) %>%
    group_split()

  for (part_rows in route_part_groups) {
    route_part_geom <- route_part_geoms %>%
      filter(route_part_id == part_rows$route_part_id[1]) %>%
      slice(1)

    if (nrow(route_part_geom) == 0 || st_is_empty(route_part_geom$geometry[[1]])) {
      next
    }

    part_geom <- route_part_geom$geometry[[1]]
    part_geom_type <- as.character(st_geometry_type(part_geom))

    if (!part_geom_type %in% c("LINESTRING", "MULTILINESTRING")) {
      warning(sprintf("Skipping route_part_id %s: geometry type '%s' (expected LINESTRING/MULTILINESTRING)",
                      part_rows$route_part_id[1], part_geom_type))
      next
    }

    part_length <- as.numeric(st_length(part_geom))

    if (is.na(part_length) || part_length <= 0) {
      next
    }

    for (i in seq_len(nrow(part_rows))) {
      fraction <- min(1, max(0, part_rows$distance_within_part[i] / part_length))
      point_geom <- tryCatch(
        st_line_sample(part_geom, sample = fraction),
        error = function(e) NULL
      )

      if (!is.null(point_geom) && length(point_geom) > 0 && !st_is_empty(point_geom)) {
        profile_points_list[[point_counter]] <- st_as_sf(
          data.frame(row_id = part_rows$row_id[i]),
          geometry = point_geom,
          crs = st_crs(rnet_with_parts)
        )
        point_counter <- point_counter + 1
      }
    }
  }

  raw_classes <- rep("Unknown", nrow(combined_profiles))

  if (length(profile_points_list) > 0) {
    profile_points_sf <- do.call(rbind, profile_points_list)
    profile_with_class <- st_join(
      profile_points_sf,
      urban_rural_shp %>% select(all_of(map_class_col)),
      join = st_intersects,
      left = TRUE
    ) %>%
      st_drop_geometry()

    matched_classes <- as.character(profile_with_class[[map_class_col]])
    matched_classes[is.na(matched_classes) | trimws(matched_classes) == ""] <- "Unknown"
    raw_classes[profile_with_class$row_id] <- matched_classes
  }

  raw_classes[is.na(raw_classes) | trimws(raw_classes) == ""] <- "Unknown"

  combined_profiles[[map_class_col]] <- raw_classes
  combined_profiles[[output_col]] <- dplyr::case_when(
    grepl("urban|town", raw_classes, ignore.case = TRUE) ~ "Urban",
    grepl("rural", raw_classes, ignore.case = TRUE) ~ "Rural",
    TRUE ~ "Unknown"
  )

  combined_profiles
}

#' Create a named colour palette for route-map layers.
#' @param level_values ordered character vector of legend levels.
#' @param brewer_name RColorBrewer palette name.
#' @return named character vector of hex colours.
create_route_map_palette <- function(level_values, brewer_name = "Set2") {
  if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
    stop("Package 'RColorBrewer' is required to generate route-map palettes.")
  }

  level_values <- unique(as.character(level_values))
  level_values <- level_values[!is.na(level_values) & trimws(level_values) != ""]

  if (length(level_values) == 0) {
    return(setNames(character(0), character(0)))
  }

  palette_size <- min(8, max(3, length(level_values)))
  base_palette <- RColorBrewer::brewer.pal(palette_size, brewer_name)

  setNames(
    grDevices::colorRampPalette(base_palette)(length(level_values)),
    level_values
  )
}

#' Prepare per-route layers for an interactive context map.
#' @param route_result list returned by `run_speed_profile_pipeline()`.
#' @param urban_rural_map optional preloaded urban/rural sf object.
#' @param urban_rural_file path to the urban/rural shapefile when
#'   `urban_rural_map` is `NULL`.
#' @param map_class_col urban/rural classification column.
#' @param route_buffer_distance route buffer distance in metres used to clip
#'   background polygons.
#' @return list of sf layers and metadata for route-map rendering.
prepare_route_context_map_data <- function(route_result,
                                           urban_rural_map = NULL,
                                           urban_rural_file = DEFAULT_URBAN_RURAL_FILE,
                                           map_class_col = "UR8Name",
                                           route_buffer_distance = 250) {
  if (is.null(route_result$rnet_tl_joined) || !inherits(route_result$rnet_tl_joined, "sf")) {
    stop("route_result$rnet_tl_joined must be an sf object.")
  }

  route_segments <- route_result$rnet_tl_joined

  if (nrow(route_segments) == 0) {
    stop("route_result$rnet_tl_joined does not contain any route segments.")
  }

  if (is.null(urban_rural_map)) {
    urban_rural_map <- st_read(urban_rural_file, quiet = TRUE)
  }

  if (!inherits(urban_rural_map, "sf")) {
    stop("urban_rural_map must be an sf object.")
  }

  if (!map_class_col %in% names(urban_rural_map)) {
    stop("Column '", map_class_col, "' not found in the urban/rural shapefile.")
  }

  ur_class_vals <- as.character(urban_rural_map[[map_class_col]])
  ur_class_vals[is.na(ur_class_vals) | trimws(ur_class_vals) == ""] <- "Unknown"
  ur8_levels <- sort(unique(c(ur_class_vals, "Unknown")))
  urban_rural_map$ur_class_map <- factor(ur_class_vals, levels = ur8_levels)

  if (st_crs(route_segments) != st_crs(urban_rural_map)) {
    route_segments <- st_transform(route_segments, st_crs(urban_rural_map))
  }

  route_segments <- route_segments %>%
    mutate(
      speed_limit_mph = dplyr::case_when(
        is.na(maxspeed_drivenet) ~ 30,
        maxspeed_drivenet <= 0 ~ 30,
        TRUE ~ as.numeric(maxspeed_drivenet)
      )
    )

  route_segments$speed_limit_label <- ifelse(
    is.na(route_segments$speed_limit_mph),
    "Unknown",
    paste0(
      format(round(route_segments$speed_limit_mph, 1), trim = TRUE, scientific = FALSE),
      " mph"
    )
  )

  route_buffer_geom <- route_segments %>%
    st_geometry() %>%
    st_union() %>%
    st_buffer(route_buffer_distance)

  urban_rural_clipped <- suppressWarnings(
    st_intersection(st_make_valid(urban_rural_map), route_buffer_geom)
  )
  urban_rural_clipped <- st_make_valid(urban_rural_clipped)

  if (nrow(urban_rural_clipped) == 0) {
    urban_rural_clipped <- urban_rural_map[0, ]
  }

  clipped_classes <- as.character(urban_rural_clipped[[map_class_col]])
  clipped_classes[is.na(clipped_classes) | trimws(clipped_classes) == ""] <- "Unknown"
  urban_rural_clipped[[map_class_col]] <- clipped_classes
  urban_rural_clipped$ur_class_map <- factor(clipped_classes, levels = ur8_levels)

  route_segments_by_area <- route_segments
  route_segments_by_area[[map_class_col]] <- "Unknown"
  route_segments_by_area$ur_class_map <- factor("Unknown", levels = ur8_levels)

  if (nrow(urban_rural_clipped) > 0) {
    route_segments_by_area <- suppressWarnings(
      st_intersection(
        st_make_valid(route_segments),
        urban_rural_clipped %>% select(all_of(map_class_col), ur_class_map)
      )
    )
    route_segments_by_area <- suppressWarnings(
      st_collection_extract(route_segments_by_area, "LINESTRING")
    )

    if (nrow(route_segments_by_area) == 0) {
      route_segments_by_area <- route_segments
      route_segments_by_area[[map_class_col]] <- "Unknown"
      route_segments_by_area$ur_class_map <- factor("Unknown", levels = ur8_levels)
    } else {
      route_classes <- as.character(route_segments_by_area[[map_class_col]])
      route_classes[is.na(route_classes) | trimws(route_classes) == ""] <- "Unknown"
      route_segments_by_area[[map_class_col]] <- route_classes
      route_segments_by_area$ur_class_map <- factor(route_classes, levels = ur8_levels)
    }
  }

  break_points_df <- route_result$break_points_df

  if (is.null(break_points_df) || nrow(break_points_df) == 0) {
    break_points <- st_sf(
      data.frame(
        stop_id = integer(0),
        position_along_route = numeric(0),
        stopping_type = character(0),
        stopping_id = character(0),
        angle_change = numeric(0),
        lon = numeric(0),
        lat = numeric(0)
      ),
      geometry = st_sfc(crs = st_crs(route_segments))
    )
  } else {
    break_points <- st_as_sf(
      break_points_df,
      coords = c("lon", "lat"),
      crs = st_crs(route_segments),
      remove = FALSE
    )
  }

  start_end_points <- st_sf(
    data.frame(
      marker_type = character(0),
      marker_order = integer(0)
    ),
    geometry = st_sfc(crs = st_crs(route_segments))
  )

  route_line <- create_route_line_geometry(route_segments)

  if (length(route_line) > 0 && !all(st_is_empty(route_line))) {
    route_coords <- st_coordinates(route_line)

    if (nrow(route_coords) >= 1) {
      start_end_df <- data.frame(
        marker_type = c("Start", "End"),
        marker_order = c(1L, 2L),
        x = c(route_coords[1, "X"], route_coords[nrow(route_coords), "X"]),
        y = c(route_coords[1, "Y"], route_coords[nrow(route_coords), "Y"])
      )

      start_end_points <- st_as_sf(
        start_end_df,
        coords = c("x", "y"),
        crs = st_crs(route_segments),
        remove = FALSE
      )
    }
  }

  list(
    urban_rural_polygons = urban_rural_clipped,
    route_segments_speed = route_segments,
    route_segments_by_area = route_segments_by_area,
    break_points = break_points,
    start_end_points = start_end_points,
    ur8_levels = ur8_levels,
    map_class_col = map_class_col,
    route_buffer_distance = route_buffer_distance
  )
}

#' Build an interactive route-context map.
#' @param route_map_data list returned by `prepare_route_context_map_data()`.
#' @param route_od_id optional route identifier shown in the map title.
#' @param map_class_col urban/rural classification column.
#' @return leaflet htmlwidget.
build_route_context_leaflet <- function(route_map_data,
                                        route_od_id = NULL,
                                        map_class_col = "UR8Name") {
  if (!requireNamespace("leaflet", quietly = TRUE)) {
    stop("Package 'leaflet' is required to generate route context maps.")
  }

  if (!requireNamespace("htmltools", quietly = TRUE)) {
    stop("Package 'htmltools' is required to generate route context maps.")
  }

  clean_label <- function(x) {
    x <- as.character(x)
    x[is.na(x) | trimws(x) == ""] <- "Unknown"
    htmltools::htmlEscape(x)
  }

  to_wgs84 <- function(layer) {
    if (!inherits(layer, "sf") || is.na(st_crs(layer))) {
      return(layer)
    }

    st_transform(layer, 4326)
  }

  urban_rural_polygons <- route_map_data$urban_rural_polygons
  route_segments_speed <- route_map_data$route_segments_speed
  route_segments_by_area <- route_map_data$route_segments_by_area
  break_points <- route_map_data$break_points
  start_end_points <- route_map_data$start_end_points

  display_ur_levels <- sort(unique(c(
    as.character(urban_rural_polygons$ur_class_map),
    as.character(route_segments_by_area$ur_class_map),
    "Unknown"
  )))
  display_ur_levels <- display_ur_levels[
    !is.na(display_ur_levels) & trimws(display_ur_levels) != ""
  ]

  if (length(display_ur_levels) == 0) {
    display_ur_levels <- "Unknown"
  }

  ur_fill_palette_values <- create_route_map_palette(display_ur_levels, "Set2")
  ur_line_palette_values <- create_route_map_palette(display_ur_levels, "Dark2")
  ur_fill_palette <- leaflet::colorFactor(
    palette = unname(ur_fill_palette_values),
    domain = display_ur_levels,
    na.color = "#9CA3AF"
  )
  ur_line_palette <- leaflet::colorFactor(
    palette = unname(ur_line_palette_values),
    domain = display_ur_levels,
    na.color = "#4B5563"
  )

  speed_domain <- route_segments_speed$speed_limit_mph
  speed_domain <- speed_domain[is.finite(speed_domain)]

  if (length(speed_domain) == 0) {
    speed_domain <- c(0, 1)
  } else if (length(unique(speed_domain)) == 1) {
    speed_domain <- c(speed_domain[1], speed_domain[1] + 1)
  }

  speed_palette <- leaflet::colorNumeric(
    palette = "YlOrRd",
    domain = speed_domain,
    na.color = "#6B7280"
  )

  turn_palette_values <- c(
    traffic_light = "#2563EB",
    sharp_turn = "#DC2626",
    roundabout = "#7C3AED",
    moderate_turn = "#EA580C",
    other = "#475569"
  )
  turn_labels <- c(
    traffic_light = "Traffic light",
    sharp_turn = "Sharp turn",
    roundabout = "Roundabout",
    moderate_turn = "Moderate turn",
    other = "Other"
  )

  urban_rural_polygons$popup_html <- paste0(
    "<strong>", clean_label(map_class_col), ":</strong> ",
    clean_label(urban_rural_polygons[[map_class_col]])
  )
  urban_rural_polygons$fill_color <- unname(
    ur_fill_palette(as.character(urban_rural_polygons$ur_class_map))
  )
  route_segments_speed$segment_length_m <- round(as.numeric(st_length(route_segments_speed)), 1)
  route_segments_speed$line_color <- unname(speed_palette(route_segments_speed$speed_limit_mph))
  route_segments_speed$popup_html <- paste0(
    "<strong>Speed limit:</strong> ", clean_label(route_segments_speed$speed_limit_label),
    "<br><strong>OSM highway:</strong> ", clean_label(route_segments_speed$highway_drivenet),
    "<br><strong>Segment length:</strong> ",
    format(route_segments_speed$segment_length_m, trim = TRUE, scientific = FALSE),
    " m"
  )

  route_segments_by_area$segment_length_m <- round(as.numeric(st_length(route_segments_by_area)), 1)
  route_segments_by_area$line_color <- unname(
    ur_line_palette(as.character(route_segments_by_area$ur_class_map))
  )
  route_segments_by_area$popup_html <- paste0(
    "<strong>", clean_label(map_class_col), ":</strong> ",
    clean_label(route_segments_by_area[[map_class_col]]),
    "<br><strong>Speed limit:</strong> ", clean_label(route_segments_by_area$speed_limit_label),
    "<br><strong>Segment length:</strong> ",
    format(route_segments_by_area$segment_length_m, trim = TRUE, scientific = FALSE),
    " m"
  )

  if (nrow(break_points) > 0) {
    break_points$turn_key <- ifelse(
      break_points$stopping_type %in% names(turn_palette_values),
      break_points$stopping_type,
      "other"
    )
    break_points$turn_color <- unname(turn_palette_values[break_points$turn_key])
    break_points$popup_html <- paste0(
      "<strong>Stopping type:</strong> ",
      clean_label(tools::toTitleCase(gsub("_", " ", break_points$stopping_type))),
      "<br><strong>Marker id:</strong> ", clean_label(break_points$stopping_id),
      "<br><strong>Distance along route:</strong> ",
      format(round(break_points$position_along_route, 1), trim = TRUE, scientific = FALSE),
      " m",
      ifelse(
        is.na(break_points$angle_change),
        "",
        paste0(
          "<br><strong>Angle change:</strong> ",
          format(round(break_points$angle_change, 1), trim = TRUE, scientific = FALSE),
          " deg"
        )
      )
    )
  }

  if (nrow(start_end_points) > 0) {
    start_end_points$popup_html <- paste0(
      "<strong>", clean_label(start_end_points$marker_type), "</strong>"
    )
  }

  urban_rural_polygons_wgs84 <- to_wgs84(urban_rural_polygons)
  route_segments_speed_wgs84 <- to_wgs84(route_segments_speed)
  route_segments_by_area_wgs84 <- to_wgs84(route_segments_by_area)
  break_points_wgs84 <- to_wgs84(break_points)
  start_end_points_wgs84 <- to_wgs84(start_end_points)

  map_title <- if (length(route_od_id) == 0 || is.na(route_od_id)) {
    "Route context map"
  } else {
    paste("Route", route_od_id, "context map")
  }

  route_map <- leaflet::leaflet(options = leaflet::leafletOptions(preferCanvas = TRUE)) %>%
    leaflet::addProviderTiles("CartoDB.Positron") %>%
    leaflet::addControl(
      html = htmltools::HTML(
        paste0(
          "<div style=\"background: rgba(255,255,255,0.92); ",
          "padding: 8px 10px; border-radius: 4px; line-height: 1.35;\">",
          "<strong>", htmltools::htmlEscape(map_title), "</strong><br>",
          "Speed limits, turning points, and urban/rural context",
          "</div>"
        )
      ),
      position = "topleft"
    )

  if (nrow(urban_rural_polygons_wgs84) > 0) {
    route_map <- route_map %>%
      leaflet::addPolygons(
        data = urban_rural_polygons_wgs84,
        fillColor = ~fill_color,
        fillOpacity = 0.35,
        color = "#4B5563",
        weight = 0.5,
        opacity = 0.5,
        popup = ~popup_html,
        group = "Urban/rural polygons"
      )
  }

  route_map <- route_map %>%
    leaflet::addPolylines(
      data = route_segments_speed_wgs84,
      color = ~line_color,
      weight = 5,
      opacity = 0.95,
      popup = ~popup_html,
      group = "Route by speed limit"
    )

  if (nrow(route_segments_by_area_wgs84) > 0) {
    route_map <- route_map %>%
      leaflet::addPolylines(
        data = route_segments_by_area_wgs84,
        color = ~line_color,
        weight = 4,
        opacity = 0.85,
        popup = ~popup_html,
        group = "Route by area class"
      )
  }

  if (nrow(break_points_wgs84) > 0) {
    route_map <- route_map %>%
      leaflet::addCircleMarkers(
        data = break_points_wgs84,
        color = ~turn_color,
        fillColor = ~turn_color,
        radius = 5,
        weight = 1,
        fillOpacity = 0.95,
        popup = ~popup_html,
        group = "Turning points"
      )
  }

  if (nrow(start_end_points_wgs84) > 0) {
    route_map <- route_map %>%
      leaflet::addCircleMarkers(
        data = start_end_points_wgs84,
        color = "#111827",
        fillColor = "#F9FAFB",
        radius = 6,
        weight = 2,
        fillOpacity = 1,
        popup = ~popup_html,
        group = "Route start/end"
      )
  }

  route_map <- route_map %>%
    leaflet::addLegend(
      position = "bottomright",
      pal = speed_palette,
      values = route_segments_speed$speed_limit_mph,
      title = "Speed limit (mph)",
      opacity = 1
    ) %>%
    leaflet::addLegend(
      position = "topright",
      colors = unname(ur_fill_palette_values),
      labels = names(ur_fill_palette_values),
      title = map_class_col,
      opacity = 0.8
    ) %>%
    leaflet::addLayersControl(
      overlayGroups = c(
        "Urban/rural polygons",
        "Route by speed limit",
        "Route by area class",
        "Turning points",
        "Route start/end"
      ),
      options = leaflet::layersControlOptions(collapsed = FALSE)
    )

  if (nrow(break_points_wgs84) > 0) {
    present_turn_levels <- unique(break_points$turn_key)
    route_map <- route_map %>%
      leaflet::addLegend(
        position = "bottomleft",
        colors = unname(turn_palette_values[present_turn_levels]),
        labels = unname(turn_labels[present_turn_levels]),
        title = "Turning points",
        opacity = 0.95
      )
  }

  route_bbox <- st_bbox(route_segments_speed_wgs84)
  route_map <- route_map %>%
    leaflet::fitBounds(
      lng1 = route_bbox["xmin"],
      lat1 = route_bbox["ymin"],
      lng2 = route_bbox["xmax"],
      lat2 = route_bbox["ymax"]
    )

  if (nrow(route_segments_by_area_wgs84) > 0) {
    route_map <- route_map %>%
      leaflet::hideGroup("Route by area class")
  }

  route_map
}

#' Build a mapview route-context map matching the original notebook style.
#' @param route_map_data list returned by `prepare_route_context_map_data()`.
#' @return mapview object.
build_route_context_mapview <- function(route_map_data) {
  if (!requireNamespace("mapview", quietly = TRUE)) {
    stop("Package 'mapview' is required to generate route context maps.")
  }

  to_wgs84 <- function(layer) {
    if (!inherits(layer, "sf") || is.na(st_crs(layer))) {
      return(layer)
    }

    st_transform(layer, 4326)
  }

  urban_rural_polygons <- to_wgs84(route_map_data$urban_rural_polygons)
  route_segments_speed <- to_wgs84(route_map_data$route_segments_speed)
  route_segments_by_area <- to_wgs84(route_map_data$route_segments_by_area)
  break_points <- to_wgs84(route_map_data$break_points)

  stop_colors <- c(
    traffic_light = "red",
    sharp_turn = "orange",
    moderate_turn = "gold",
    roundabout = "purple",
    other = "black"
  )

  route_map <- NULL

  if (nrow(urban_rural_polygons) > 0) {
    route_map <- mapview::mapview(
      urban_rural_polygons,
      zcol = "ur_class_map",
      alpha.regions = 0.25,
      layer.name = "Urban/rural polygons"
    )
  }

  speed_layer <- mapview::mapview(
    route_segments_speed,
    zcol = "speed_limit_mph",
    lwd = 4,
    layer.name = "Roads (speed limit)"
  )

  if (is.null(route_map)) {
    route_map <- speed_layer
  } else {
    route_map <- route_map + speed_layer
  }

  if (nrow(route_segments_by_area) > 0) {
    route_map <- route_map + mapview::mapview(
      route_segments_by_area,
      zcol = "ur_class_map",
      lwd = 3,
      layer.name = "Route by area class"
    )
  }

  if (nrow(break_points) > 0) {
    stopping_types <- unique(as.character(break_points$stopping_type))

    for (stop_type in stopping_types) {
      stops_subset <- break_points[break_points$stopping_type == stop_type, ]

      if (nrow(stops_subset) == 0) {
        next
      }

      stop_key <- if (stop_type %in% names(stop_colors)) {
        stop_type
      } else {
        "other"
      }

      route_map <- route_map + mapview::mapview(
        stops_subset,
        col.regions = stop_colors[[stop_key]],
        cex = 6,
        layer.name = tools::toTitleCase(gsub("_", " ", stop_type))
      )
    }
  }

  route_map
}

#' Compile the four output tables from route-part profiles.
#' @param route_part_profiles list returned by `generate_route_part_profiles()`.
#' @return list containing the four CSV tables as data.frames.
compile_speed_profile_tables <- function(route_part_profiles) {
  all_profiles <- list()

  for (i in seq_along(route_part_profiles)) {
    part_profile <- route_part_profiles[[i]]
    profile <- part_profile$profile
    meta <- part_profile$metadata

    if (is.null(meta) || nrow(meta) == 0) {
      meta <- data.frame(
        route_part_id = i,
        original_od_id = NA,
        segment_start = 0,
        segment_end = max(profile$distance, na.rm = TRUE),
        segment_length = max(profile$distance, na.rm = TRUE),
        has_break_start = FALSE,
        has_break_end = FALSE
      )
    }

    profile_clean <- profile %>%
      mutate(
        route_part_id = meta$route_part_id[1],
        original_od_id = meta$original_od_id[1],
        part_distance_start = meta$segment_start[1],
        part_distance_end = meta$segment_end[1],
        distance_along_route = distance + meta$segment_start[1],
        has_break_start = meta$has_break_start[1],
        has_break_end = meta$has_break_end[1]
      ) %>%
      select(
        original_od_id,
        route_part_id,
        distance_within_part = distance,
        distance_along_route,
        part_distance_start,
        part_distance_end,
        speed_ms,
        speed_mph,
        speed_limit_mph,
        total_length,
        has_break_start,
        has_break_end
      )

    all_profiles[[i]] <- profile_clean
  }

  combined_profiles <- bind_rows(all_profiles)
  speed_distribution <- compute_speed_distribution_trip(combined_profiles)
  speed_distribution_simple <- speed_distribution %>%
    transmute(`Speed range (km/h)` = as.character(speed_bin), `Share (%)` = proportion_percent)

  summary_stats <- bind_rows(lapply(route_part_profiles, function(part_profile) {
    profile <- part_profile$profile
    meta <- part_profile$metadata

    if (is.null(meta) || nrow(meta) == 0) {
      meta <- data.frame(
        route_part_id = NA,
        original_od_id = NA,
        segment_start = 0,
        segment_end = max(profile$distance, na.rm = TRUE),
        segment_length = max(profile$distance, na.rm = TRUE),
        has_break_start = FALSE,
        has_break_end = FALSE
      )
    }

    data.frame(
      original_od_id = meta$original_od_id[1],
      route_part_id = meta$route_part_id[1],
      part_distance_start = meta$segment_start[1],
      part_distance_end = meta$segment_end[1],
      total_length_m = max(profile$distance),
      max_speed_mph = max(profile$speed_mph),
      avg_speed_mph = mean(profile$speed_mph),
      start_speed_mph = part_profile$start_speed_ms / 0.44704,
      end_speed_mph = part_profile$end_speed_ms / 0.44704,
      num_data_points = nrow(profile),
      num_speed_zones = nrow(part_profile$speed_zones),
      has_break_start = part_profile$has_break_at_start,
      has_break_end = part_profile$has_break_at_end
    )
  }))

  list(
    combined_profiles = combined_profiles,
    speed_distribution = speed_distribution,
    speed_distribution_simple = speed_distribution_simple,
    summary_stats = summary_stats
  )
}

#' Save the four notebook output tables.
#' @param output_tables list returned by `compile_speed_profile_tables()` or
#'   `run_speed_profile_pipeline()`.
#' @param output_dir directory where CSV files should be written.
#' @return invisibly returns the output file paths.
write_speed_profile_outputs <- function(output_tables,
                                        output_dir = "output/speed_profiles_modular") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  all_speed_profiles_path <- file.path(output_dir, "all_speed_profiles.csv")
  speed_distribution_path <- file.path(output_dir, "speed_distribution_trip.csv")
  all_speed_profiles_simple_path <- file.path(output_dir, "all_speed_profiles_simple.csv")
  summary_path <- file.path(output_dir, "speed_profile_summary.csv")

  write.csv(output_tables$combined_profiles, all_speed_profiles_path, row.names = FALSE)
  write.csv(output_tables$speed_distribution, speed_distribution_path, row.names = FALSE)
  write.csv(output_tables$speed_distribution_simple, all_speed_profiles_simple_path, row.names = FALSE)
  write.csv(output_tables$summary_stats, summary_path, row.names = FALSE)

  invisible(list(
    all_speed_profiles = all_speed_profiles_path,
    speed_distribution_trip = speed_distribution_path,
    all_speed_profiles_simple = all_speed_profiles_simple_path,
    speed_profile_summary = summary_path
  ))
}

#' Compute and optionally save the exact notebook outputs for one route.
#' @param route_od_id route identifier to process.
#' @param route_file path to the GeoJSON route file.
#' @param region region used when loading the OSM driving network.
#' @param drive_net optional preloaded driving network to reuse across routes.
#' @param output_dir optional directory for the four CSV outputs.
#' @return list containing route parts, route-part profiles, and the four output tables.
run_speed_profile_pipeline <- function(route_od_id = 3266,
                                       route_file = DEFAULT_SPEED_PROFILE_ROUTE_FILE,
                                       region = DEFAULT_SPEED_PROFILE_REGION,
                                       drive_net = NULL,
                                       urban_rural_file = DEFAULT_URBAN_RURAL_FILE,
                                       map_class_col = "UR8Name",
                                       output_col = "urban_rural_class",
                                       output_dir = NULL) {
  rnet_tl_joined <- load_route_with_speed_data(
    route_od_id = route_od_id,
    route_file = route_file,
    region = region,
    drive_net = drive_net
  )
  break_point_results <- build_break_points(rnet_tl_joined)
  rnet_with_parts <- build_route_parts(
    rnet_tl_joined = rnet_tl_joined,
    break_points_df = break_point_results$break_points_df,
    original_route_with_cumulative = break_point_results$original_route_with_cumulative
  )
  route_part_profiles <- generate_route_part_profiles(
    rnet_with_parts = rnet_with_parts,
    original_route_with_cumulative = break_point_results$original_route_with_cumulative
  )
  output_tables <- compile_speed_profile_tables(route_part_profiles)
  output_tables$combined_profiles <- classify_speed_profiles_by_map_class(
    combined_profiles = output_tables$combined_profiles,
    rnet_with_parts = rnet_with_parts,
    urban_rural_file = urban_rural_file,
    map_class_col = map_class_col,
    output_col = output_col
  )
  output_tables$speed_distribution_simple <- compute_speed_distribution_simple_by_area(
    output_tables$combined_profiles
  )

  results <- c(
    list(
      rnet_tl_joined = rnet_tl_joined,
      break_points_df = break_point_results$break_points_df,
      original_route_with_cumulative = break_point_results$original_route_with_cumulative,
      rnet_with_parts = rnet_with_parts,
      route_part_profiles = route_part_profiles
    ),
    output_tables
  )

  if (!is.null(output_dir)) {
    write_speed_profile_outputs(results, output_dir = output_dir)
  }

  results
}
