library(terra)
library(raster)

# base folder containing yearly MOD13A1 folders
base_dir <- "Y:/Duanyang/VIIRS V2/VNP13A1A2"

# output folder
outdir <- "E:/akash/BioSat/EVI"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# helper: extract bit ranges from uint16
get_bits <- function(x, from, to) {
  mask <- bitwShiftL(1L, (to - from + 1L)) - 1L
  bitwAnd(bitwShiftR(x, from), mask)
}

# loop through years
for (yr in 2001:2018) {
  
  folder <- file.path(base_dir, paste0("MOD13A1_", yr))
  
  if (!dir.exists(folder)) {
    cat("Skipping year", yr, "- folder not found:", folder, "\n")
    next
  }
  
  cat("\n========== YEAR:", yr, "==========\n")
  
  # loop through h and v tiles
  for (h in 8:13) {
    for (v in 4:6) {
      
      tile <- sprintf("h%02dv%02d", h, v)
      cat("\nProcessing tile:", tile, "for year:", yr, "\n")
      
      # find all matching HDF files for this year and tile
      hdf_files <- list.files(
        path = folder,
        pattern = paste0("A", yr, "\\d{3}\\.", tile, ".*\\.hdf$"),
        full.names = TRUE
      )
      
      if (length(hdf_files) == 0) {
        cat("No files found for", tile, "in", yr, "\n")
        next
      }
      
      evi_list <- list()
      evi_raw <- list()
      for (i in seq_along(hdf_files)) {
        hdf_file <- hdf_files[i]
        cat("  Reading:", basename(hdf_file), "\n")
        
        r <- try(rast(hdf_file), silent = TRUE)
        if (inherits(r, "try-error")) {
          cat("  Failed to read:", basename(hdf_file), "\n")
          next
        }
        
        # read layers
        evi <- try(r[["\"500m 16 days EVI\""]], silent = TRUE)
        #summaryQA <- try(r[["\"500m 16 days pixel reliability\""]], silent = TRUE)
        dqa <- try(r[["\"500m 16 days VI Quality\""]], silent = TRUE)
        
        #if (inherits(evi, "try-error") || inherits(summaryQA, "try-error") || inherits(dqa, "try-error")) {
        #  cat("  Required layers missing in:", basename(hdf_file), "\n")
        #  next
        #}
        
        # replace fill values BEFORE scaling
        evi[evi == -3000] <- NA
        
        # apply MODIS scale factor
        evi <- evi * 0.0001
        
        # summary QA mask
        # 0 = good, 1 = marginal, 2 = snow/ice, 3 = cloudy
        #mask_good <- summaryQA == 0
        
        # detailed QA bit extraction
        dqa_i <- as.int(dqa)
        
        vi_quality  <- app(dqa_i, get_bits, from = 15,  to = 16)   # 00 VI produced, good quality; https://lpdaac.usgs.gov/documents/104/MOD13_ATBD.pdf
        #usefulness  <- app(dqa_i, get_bits, from = 2,  to = 5)   # 
        #aerosol     <- app(dqa_i, get_bits, from = 6,  to = 7)   # 
        #adj_cloud   <- app(dqa_i, get_bits, from = 8,  to = 8)   # 
        mixed_cloud <- app(dqa_i, get_bits, from = 7, to = 7)  # 0 no
        #land_mask   <- app(dqa_i, get_bits, from = 11, to = 12)  # 1 land
        snow_mask   <- app(dqa_i, get_bits, from = 4, to = 4)  # 0 no snow
        #shadow_mask <- app(dqa_i, get_bits, from = 15, to = 15)  # 0 no shadow
        
        mask_detailed <- (vi_quality <= 00) &
         # (usefulness <= 2) &
         # (aerosol <= 2) &
          #(adj_cloud == 0) &
          (mixed_cloud == 0) &
         # (land_mask == 1) &
          (snow_mask == 0)# &
        #  (shadow_mask == 0)
        
        # apply masks in sequence
        #evi_corrected <- mask(evi, mask_good, maskvalues = 0, updatevalue = NA)
        evi_corrected <- mask(evi, mask_detailed, maskvalues = 0, updatevalue = NA)
        
        evi_list[[length(evi_list) + 1]] <- evi_corrected
        evi_raw[[length(evi_raw) + 1]] <- evi
      }
      
      if (length(evi_list) == 0) {
        cat("No valid EVI layers for", tile, "in", yr, "\n")
        next
      }
      
      # stack all rasters
      evi_stack <- rast(evi_list)
      
      # interpolate through time
      evi_brick <- raster::brick(evi_stack)
      evi_interp_brick <- raster::approxNA(evi_brick, method = "linear", rule = 2)
      evi_interp <- terra::rast(evi_interp_brick)
      
      # median
      evi_median <- app(evi_interp, fun = median, na.rm = TRUE)
      
      median_file <- file.path(outdir, paste0("EVI_pixelwise_median_", yr, "_", tile, ".tif"))
      writeRaster(evi_median, median_file, overwrite = TRUE)
      cat("Saved median:", median_file, "\n")
      
      # 90th percentile
      evi_p90 <- app(
        evi_interp,
        fun = function(x) quantile(x, probs = 0.90, na.rm = TRUE)
      )
      
      p90_file <- file.path(outdir, paste0("EVI_pixelwise_90Percentile_", yr, "_", tile, ".tif"))
      writeRaster(evi_p90, p90_file, overwrite = TRUE)
      cat("Saved p90:", p90_file, "\n")
    }
  }
}



## mosaicking globally

library(terra)

outdir <- "E:/akash/BioSat/EVI/median"
mosaic_dir <- file.path(outdir, "mosaic")
dir.create(mosaic_dir, recursive = TRUE, showWarnings = FALSE)

for (yr in 2001:2018) {
  
  cat("\nCreating mosaics for year:", yr, "\n")
  
  # median
  median_tiles <- list.files(
    outdir,
    pattern = paste0("^EVI_pixelwise_median_", yr, "_h\\d{2}v\\d{2}\\.tif$"),
    full.names = TRUE
  )
  
  if (length(median_tiles) > 0) {
    median_mosaic <- do.call(merge, lapply(median_tiles, rast))
    writeRaster(
      median_mosaic,
      file.path(mosaic_dir, paste0("EVI_pixelwise_median_", yr, "_mosaic.tif")),
      overwrite = TRUE
    )
  }
  
  
}
