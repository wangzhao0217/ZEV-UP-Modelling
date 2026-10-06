#| label: scotland-adoption-propensity-map
#| echo: false
#| fig-cap: "Scotland-wide adoption propensity distribution across all Output Areas, demonstrating spatial variation in demographic-based EV adoption likelihood."
#| fig-width: 12
#| message: false
#| fig-height: 10
#| eval: false
library(sf)
library(dplyr)
library(ggplot2)
library(viridis)
library(patchwork)


# Load all regional GPKG files from census_year output directory
scotland_adoption_propensity <- NULL

for (r in regions) {
    gpkg_path <- opath(r, "adoption_propensity_categorized.gpkg")
    cat(paste("Loading:", r, "\n"))
    if (file.exists(gpkg_path)) {
        region_data <- st_read(gpkg_path, quiet = TRUE) |>
            select(final_adoption_propensity) |>
            mutate(region = r)
        scotland_adoption_propensity <- rbind(scotland_adoption_propensity, region_data)
    } else {
        warning(paste("File not found:", gpkg_path))
    }
}

cat(paste("\nTotal Output Areas loaded:", nrow(scotland_adoption_propensity), "\n"))

# Fix invalid OA geometries and transform to British National Grid (EPSG:27700) to match Figure 10
scotland_adoption_propensity <- sf::st_make_valid(scotland_adoption_propensity)
scotland_adoption_propensity <- sf::st_transform(scotland_adoption_propensity, 27700)

# Create regional boundaries for overlay
cat("Creating regional boundaries...\n")

# Fix any invalid geometries and create boundaries
regional_boundaries <- scotland_adoption_propensity |>
    st_make_valid() |>
    st_buffer(0) |>
    group_by(region) |>
    summarise(do_union = TRUE) |>
    ungroup()

cat(paste("Regional boundaries created:", nrow(regional_boundaries), "regions\n"))

# Create map colored by adoption propensity score
adoption_map <- ggplot() +
    # Plot output areas with adoption propensity
    geom_sf(
        data = scotland_adoption_propensity,
        aes(fill = final_adoption_propensity),
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
    # Use RdYlBu gradient matching Figure 10 style (deep red -> yellow -> deep blue)
    scale_fill_gradientn(
        colors = c(
            "#A50026", "#D73027", "#F46D43", "#FDAE61", "#FEE090", "#FFFFBF",
            "#E0F3F8", "#ABD9E9", "#74ADD1", "#4575B4", "#313695"
        ),
        name = "Adoption\nPropensity",
        limits = c(0, 1),
        breaks = seq(0, 1, 0.2),
        labels = sprintf("%.1f", seq(0, 1, 0.2)),
        na.value = "grey90"
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
print(adoption_map)

# Save the combined Scotland-wide adoption propensity data
st_write(scotland_adoption_propensity,
    "../output/scotland_adoption_propensity_combined.gpkg",
    delete_dsn = TRUE,
    quiet = TRUE
)

# Save the plot
ggsave("../figs/scotland_adoption_propensity_map.png",
    adoption_map,
    width = 12,
    height = 10,
    dpi = 300
)

cat("\nSaved combined data to: ../output/scotland_adoption_propensity_combined.gpkg\n")
cat("Saved map to: ../figs/scotland_adoption_propensity_map.png\n")

# Helper to build city zoom plots around given lon/lat (WGS84) in EPSG:27700 (matching Figure 10)
make_adoption_city_zoom <- function(city_name, lon, lat, buffer_km = 8) {
    point_wgs <- st_sfc(st_point(c(lon, lat)), crs = 4326)
    point_bng <- st_transform(point_wgs, 27700)
    bbox_bng <- st_bbox(st_buffer(point_bng, buffer_km * 1000))
    adoption_map +
        coord_sf(
            xlim = c(bbox_bng["xmin"], bbox_bng["xmax"]),
            ylim = c(bbox_bng["ymin"], bbox_bng["ymax"]),
            expand = FALSE
        ) +
        labs(title = city_name) +
        theme(
            legend.position = "none",
            plot.title = element_text(size = 12, face = "bold")
        )
}

# Build six city adoption zoom maps
adopt_glasgow <- make_adoption_city_zoom("Glasgow", -4.2518, 55.8642, 8)
adopt_edinburgh <- make_adoption_city_zoom("Edinburgh", -3.1883, 55.9533, 8)
adopt_aberdeen <- make_adoption_city_zoom("Aberdeen", -2.0954, 57.1497, 8)
adopt_inverness <- make_adoption_city_zoom("Inverness", -4.2247, 57.4778, 8)
adopt_dundee <- make_adoption_city_zoom("Dundee", -2.9707, 56.4620, 8)
adopt_dumfries <- make_adoption_city_zoom("Dumfries", -3.6110, 55.0699, 8)

# Save each city plot individually
ggsave("../figs/adoption_zoom_glasgow.png", adopt_glasgow, width = 6, height = 5, dpi = 300)
ggsave("../figs/adoption_zoom_edinburgh.png", adopt_edinburgh, width = 6, height = 5, dpi = 300)
ggsave("../figs/adoption_zoom_aberdeen.png", adopt_aberdeen, width = 6, height = 5, dpi = 300)
ggsave("../figs/adoption_zoom_inverness.png", adopt_inverness, width = 6, height = 5, dpi = 300)
ggsave("../figs/adoption_zoom_dundee.png", adopt_dundee, width = 6, height = 5, dpi = 300)
ggsave("../figs/adoption_zoom_dumfries.png", adopt_dumfries, width = 6, height = 5, dpi = 300)

# Add city name labels to the national panel (Comment 1.3 / Figure 10 style)
adoption_cities <- st_as_sf(
    data.frame(
        name = c("Inverness", "Aberdeen", "Glasgow", "Edinburgh", "Dumfries", "Dundee"),
        lon = c(-4.2247, -2.0954, -4.2518, -3.1883, -3.6110, -2.9707),
        lat = c(57.4778, 57.1497, 55.8642, 55.9533, 55.0699, 56.4620)
    ),
    coords = c("lon", "lat"),
    crs = 4326
) |>
    st_transform(27700)

# Add city name labels to the national panel with prominent leader lines and markers
city_pts <- st_coordinates(adoption_cities)
city_label_df <- data.frame(
    name = adoption_cities$name,
    x = city_pts[, 1],
    y = city_pts[, 2],
    xend = city_pts[, 1] + c(-26000, 26000, -26000, 26000, 10000, 24000),
    yend = city_pts[, 2] + c(26000, 16000, -20000, -18000, -22000, 22000)
)

adoption_map_labelled <- adoption_map +
    # Prominent leader lines connecting city marker to label box
    geom_segment(
        data = city_label_df,
        aes(x = x, y = y, xend = xend, yend = yend),
        color = "#111111", linewidth = 1.2
    ) +
    # Prominent city location marker (white circle with thick black outline)
    geom_point(
        data = city_label_df,
        aes(x = x, y = y),
        shape = 21, color = "#111111", fill = "white", size = 5.0, stroke = 1.6
    ) +
    # Inner center pin dot
    geom_point(
        data = city_label_df,
        aes(x = x, y = y),
        shape = 16, color = "#111111", size = 2.0
    ) +
    # City label box
    geom_label(
        data = city_label_df,
        aes(x = xend, y = yend, label = name),
        size = 4.4, fontface = "bold", colour = "#111111", fill = "white",
        alpha = 0.95, label.size = 0.55, label.padding = unit(0.22, "lines")
    )

# Assemble combined 3x3 layout identical to Figure 10 charging map
A <- adopt_inverness
B <- adoption_map_labelled
C <- adopt_aberdeen
D <- adopt_glasgow
E <- adopt_edinburgh
F <- adopt_dumfries
G <- adopt_dundee

adopt_layout <- c(
    area(t = 1, l = 1, b = 1, r = 1), # A
    area(t = 1, l = 2, b = 3, r = 2), # B spans all rows in middle column
    area(t = 1, l = 3, b = 1, r = 3), # C
    area(t = 2, l = 1, b = 2, r = 1), # D
    area(t = 2, l = 3, b = 2, r = 3), # E
    area(t = 3, l = 1, b = 3, r = 1), # F
    area(t = 3, l = 3, b = 3, r = 3) # G
)

combined_adoption_map <- (A + B + C + D + E + F + G) +
    plot_layout(design = adopt_layout, widths = c(0.7, 2.6, 0.7), heights = c(1, 1, 1), guides = "collect") &
    theme(legend.position = "right")

print(combined_adoption_map)

ggsave("../figs/scotland_adoption_propensity_map_with_city_zooms.png",
    combined_adoption_map,
    width = 14,
    height = 12,
    dpi = 300
)
