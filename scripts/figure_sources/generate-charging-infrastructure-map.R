#| label: generate-charging-infrastructure-map
#| echo: false
#| eval: false
#| output: false
library(sf)
library(ggplot2)
library(dplyr)
library(patchwork)
charging_network_region <- list()

# Load charging network data for all available regions
infra_available_regions <- c("HITRANS", "Nestrans", "SESTRAN", "SPT", "SWESTRANS", "Tactran", "ZetTrans")

for (region in infra_available_regions) {
    file_path <- opath(region, "enhanced_charging_infrastructure.gpkg")
    if (file.exists(file_path)) {
        charging_network_region[[region]] <- st_read(file_path, quiet = TRUE) |>
            dplyr::select(
                nearest_charger_distance, chargers_within_1km, chargers_within_2km,
                chargers_within_5km, accessibility_score, public_charging_reliance,
                charging_bottleneck, infrastructure_factor, capacity_factor,
                gap_severity_index, gap_severity_category
            ) |>
            mutate(region = region)
    }
}

# Combine all regions
scotland_charging_network <- bind_rows(charging_network_region)

cat(paste("Total output areas with charging data:", nrow(scotland_charging_network), "\n"))
cat(paste(
    "Accessibility score range:",
    sprintf("%.3f", min(scotland_charging_network$accessibility_score, na.rm = TRUE)),
    "to",
    sprintf("%.3f", max(scotland_charging_network$accessibility_score, na.rm = TRUE)), "\n"
))

# Load charging station location data
charger_locations <- st_read("../data/charger_location_data/ocm_scotland_points_local.geojson",
    quiet = TRUE
)

cat(paste("Total charging locations:", nrow(charger_locations), "\n"))
cat(paste("Total charging points:", sum(charger_locations$num_points, na.rm = TRUE), "\n"))

# Create regional boundaries
regional_boundaries <- scotland_charging_network |>
    st_make_valid() |>
    st_buffer(0) |>
    group_by(region) |>
    summarise(do_union = TRUE) |>
    ungroup()

cat(paste("Regional boundaries created:", nrow(regional_boundaries), "regions\n"))

# Restrict charger points to the Scottish landmass (drop offshore / out-of-area points)
charger_locations <- sf::st_transform(charger_locations, sf::st_crs(regional_boundaries))
charger_locations <- charger_locations[lengths(sf::st_intersects(charger_locations, regional_boundaries)) > 0, ]
cat(paste("Charging locations within Scotland:", nrow(charger_locations), "\n"))

# Create map colored by accessibility score
accessibility_map <- ggplot() +
    # Plot output areas with accessibility scores
    geom_sf(
        data = scotland_charging_network,
        aes(fill = accessibility_score),
        color = NA,
        size = 0
    ) +
    # Add region boundaries
    geom_sf(
        data = regional_boundaries,
        fill = NA,
        color = "black",
        size = 0.8
    ) +
    # Add charging station points sized by number of charging points
    geom_sf(
        data = charger_locations,
        aes(size = num_points),
        color = "black",
        fill = "yellow",
        shape = 21,
        alpha = 0.7,
        stroke = 0.5
    ) +
    # Use gradient: red (poor access) -> yellow -> green (good access)
    scale_fill_gradientn(
        colors = c(
            "#D73027", "#F46D43", "#FDAE61", "#FEE090", "#FFFFBF",
            "#E0F3F8", "#ABD9E9", "#74ADD1", "#4575B4", "#313695"
        ),
        name = "Accessibility\nScore",
        limits = c(0.2, 1.0),
        breaks = seq(0.2, 1.0, 0.2),
        labels = sprintf("%.1f", seq(0.2, 1.0, 0.2)),
        na.value = "grey90"
    ) +
    # Scale point sizes
    scale_size_continuous(
        name = "Charging\nPoints",
        range = c(0.5, 4),
        breaks = c(1, 2, 5, 10, 20)
    ) +
    theme_minimal() +
    theme(
        axis.text = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        panel.grid = element_blank(),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        legend.position = "right",
        legend.key.height = unit(1.5, "cm"),
        legend.key.width = unit(0.5, "cm"),
        legend.spacing.y = unit(0.3, "cm")
    )

# Display the map
print(accessibility_map)

# Save the combined Scotland-wide charging network data
st_write(scotland_charging_network,
    "../output/scotland_charging_network_combined.gpkg",
    delete_dsn = TRUE,
    quiet = TRUE
)
ggsave("../figs/scotland_charging_infrastructure_map.png",
    accessibility_map,
    width = 12,
    height = 10,
    dpi = 300
)

# Helper to build city zoom plots around given lon/lat (WGS84)
make_city_zoom <- function(city_name, lon, lat, buffer_km = 8) {
    point_wgs <- st_sfc(st_point(c(lon, lat)), crs = 4326)
    point_bng <- st_transform(point_wgs, 27700)
    bbox_bng <- st_bbox(st_buffer(point_bng, buffer_km * 1000))
    bbox_poly_bng <- st_as_sfc(bbox_bng)
    bbox_poly_data <- st_transform(bbox_poly_bng, st_crs(scotland_charging_network))
    bb <- st_bbox(bbox_poly_data)
    accessibility_map +
        coord_sf(
            xlim = c(bb["xmin"], bb["xmax"]),
            ylim = c(bb["ymin"], bb["ymax"]),
            expand = FALSE
        ) +
        labs(title = city_name) +
        theme(
            legend.position = "none",
            plot.title = element_text(size = 12, face = "bold")
        )
}

# Build six city zoom maps
glasgow <- make_city_zoom("Glasgow", -4.2518, 55.8642, 8)
edinburgh <- make_city_zoom("Edinburgh", -3.1883, 55.9533, 8)
aberdeen <- make_city_zoom("Aberdeen", -2.0954, 57.1497, 8)
inverness <- make_city_zoom("Inverness", -4.2247, 57.4778, 8)
dundee <- make_city_zoom("Dundee", -2.9707, 56.4620, 8)
dumfries <- make_city_zoom("Dumfries", -3.6110, 55.0699, 8)

# Add city name labels to the national panel (Comment 1.3)
charging_cities <- st_as_sf(
    data.frame(
        name = c("Inverness", "Aberdeen", "Glasgow", "Edinburgh", "Dumfries", "Dundee"),
        lon = c(-4.2247, -2.0954, -4.2518, -3.1883, -3.6110, -2.9707),
        lat = c(57.4778, 57.1497, 55.8642, 55.9533, 55.0699, 56.4620)
    ),
    coords = c("lon", "lat"),
    crs = 4326
) |>
    st_transform(st_crs(scotland_charging_network))

accessibility_map_labelled <- accessibility_map +
    geom_sf(data = charging_cities, color = "black", fill = "white", shape = 21, size = 2.8, stroke = 0.7) +
    geom_sf_label(
        data = charging_cities, aes(label = name),
        size = 4.4, fontface = "bold", colour = "black", fill = "white",
        alpha = 0.78, label.size = 0.25, label.padding = unit(0.14, "lines"),
        nudge_y = 0.12
    )

# Assemble 3x3 layout with explicit rectangular areas (main map spans middle column)
A <- inverness # top-left
B <- accessibility_map_labelled # middle column (spans rows)
C <- aberdeen # top-right
D <- glasgow # mid-left
E <- edinburgh # mid-right
F <- dumfries # bottom-left
G <- dundee # bottom-right

layout <- c(
    area(t = 1, l = 1, b = 1, r = 1), # A
    area(t = 1, l = 2, b = 3, r = 2), # B spans all rows in middle column
    area(t = 1, l = 3, b = 1, r = 3), # C
    area(t = 2, l = 1, b = 2, r = 1), # D
    area(t = 2, l = 3, b = 2, r = 3), # E
    area(t = 3, l = 1, b = 3, r = 1), # F
    area(t = 3, l = 3, b = 3, r = 3) # G
)

combined_map <- (A + B + C + D + E + F + G) +
    plot_layout(design = layout, widths = c(0.7, 2.6, 0.7), heights = c(1, 1, 1), guides = "collect") &
    theme(legend.position = "right")

# Print and save combined figure
print(combined_map)

ggsave("../figs/scotland_charging_infrastructure_map_with_city_zooms.png",
    combined_map,
    width = 14,
    height = 12,
    dpi = 300
)
