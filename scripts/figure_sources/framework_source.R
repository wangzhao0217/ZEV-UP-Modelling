# Needed packages
# install.packages(c("httr", "magick", "knitr"))  # run once if needed
library(httr)
library(magick)
library(knitr)

dir.create("figs", showWarnings = FALSE)

mermaid_txt <- '
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#ffffff",
    "lineColor": "#000000",
    "lineWidth": 2,
    "fontSize": "18px",
    "fontFamily": "Arial, sans-serif"
  },
  "flowchart": {
    "defaultRenderer": "elk",
    "htmlLabels": true,
    "nodeSpacing": 30,
    "rankSpacing": 50,
    "padding": 14,
    "curve": "basis"
  }
}}%%
flowchart TD
  subgraph DS["Data Sources"]
    direction LR
    A1["Scottish Census<br>Output Areas"]
    A2["Origin–Destination<br>travel data"]
    A3["Network-routed<br>distances &amp; paths"]
    A4["Public charging<br>station locations"]
  end

  S1["<b>Stage 1</b><br>Spatial join of census<br>demographics to OD pairs"]

  subgraph SC["Scoring Components (Stages 2–5)"]
    direction LR
    S2["<b>Stage 2</b><br>Adoption propensity<br>from demographics"]
    S3["<b>Stage 3</b><br>Trip-purpose<br>suitability weights"]
    S4["<b>Stage 4</b><br>Charging infrastructure<br>accessibility"]
    S5["<b>Stage 5</b><br>Range feasibility<br>(80 km effective range)"]
  end

  S6["<b>Stage 6</b><br>Trip substitution estimate<br><i>N = f × R × W</i>"]
  S7["<b>Stage 7</b><br>EV type selection<br>(2-seat / 4-seat)"]
  S8["<b>Stage 8</b><br>Separate capacity check<br>and total assessment"]

  A1 --> S1
  A2 --> S1
  S1 --> S2
  S1 --> S3
  A3 --> S5
  A4 --> S4
  S2 -- "AP_i ∈ [0,1]" --> S6
  S3 -- "W_p ∈ [0,1]" --> S6
  S4 -- "Charging access C_ij" --> S6
  S5 -- "RF_ij ∈ {0,1}" --> S6
  S6 --> S7
  S7 --> S8

  classDef data fill:#d0e4f7,stroke:#2b5c8a,stroke-width:2px,color:#1a3a5c
  classDef integrate fill:#e8d5f5,stroke:#6b3fa0,stroke-width:2px,color:#3d1f6d
  classDef score fill:#d4edda,stroke:#2d7a3a,stroke-width:2px,color:#1b4d24
  classDef converge fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#664d00
  classDef output fill:#f8d7da,stroke:#a94442,stroke-width:2px,color:#6b2c2e

  class A1,A2,A3,A4 data
  class S1 integrate
  class S2,S3,S4,S5 score
  class S6,S7 converge
  class S8 output
'

png_path <- "figs/flowchart_R2_v4.png"
jpg_path <- "figs/flowchart_R2_v4.jpg"

# If the JPG already exists (e.g., pre-rendered or checked-in),
# skip online rendering to support offline builds.
if (!file.exists(jpg_path)) {
    # Try rendering via Kroki when internet is available; fail gracefully offline.
    try(
        {
            resp <- httr::POST(
                url = "https://kroki.io/mermaid/png",
                body = mermaid_txt,
                encode = "raw",
                httr::add_headers(`Content-Type` = "text/plain; charset=utf-8"),
                httr::write_disk(png_path, overwrite = TRUE)
            )
            httr::stop_for_status(resp)

            # Convert PNG -> JPG (great for LaTeX/PDF inclusion)
            img <- magick::image_read(png_path)
            img <- magick::image_background(img, "white", flatten = TRUE)
            magick::image_write(img, path = jpg_path, format = "jpeg", quality = 100)
        },
        silent = TRUE
    )
}

# # Insert once (avoid also adding a Markdown image below to prevent duplicates)
# knitr::include_graphics(jpg_path)
