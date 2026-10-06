# Install the recorded R package versions for manuscript rendering.
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grepl("^--file=", args)])
root <- normalizePath(file.path(dirname(script), ".."))
packages <- read.csv(file.path(root, "environment", "R-packages.csv"))
options(repos = c(CRAN = "https://cloud.r-project.org"))
for (i in seq_len(nrow(packages))) {
    package <- packages$package[i]
    version <- as.character(packages$version[i])
    if (!requireNamespace(package, quietly = TRUE) ||
        packageVersion(package) != package_version(version)) {
        if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
        remotes::install_version(package, version = version,
                                 dependencies = NA, upgrade = "never")
    }
}
