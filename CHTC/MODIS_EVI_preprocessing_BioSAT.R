library(terra)

# folder containing HDF files
folder <- "Y:/Duanyang/VIIRS V2/VNP13A1A2/MOD13A1_2009"
outdir <- "V:\\akash\\temp"
# list all HDF files with h19v07
hdf_files <- list.files(
  path = folder,
  pattern = "h19v07.*\\.hdf$",
  full.names = TRUE
)

# read EVI from each file and prepare it
evi_list <- lapply(hdf_files, function(hdf_file) {
  cat("Processing:", basename(hdf_file), "\n")
  
  r <- rast(hdf_file)
  evi <- r[["\"500m 16 days EVI\""]]
  
  # apply MODIS scale factor
  evi <- evi * 0.0001
  
  # replace fill value with NA before scaling
  evi[evi == -3000] <- NA
  
  ## QA Masking
  summaryQA <- r[["\"500m 16 days pixel reliability\""]]      # adjust if needed
  dqa <- r[["\"500m 16 days VI Quality\"" ]] 
  
  # strict
  mask_good <- summaryQA == 0 ## SummaryQA values (0 good, 1 marginal, 2 snow/ice, 3 cloudy)
  
  # Helper: extract bit ranges from uint16
  get_bits <- function(x, from, to) {
    # from/to are bit positions (0 = least significant bit)
    mask <- bitwShiftL(1L, (to - from + 1L)) - 1L
    bitwAnd(bitwShiftR(x, from), mask)
  }
  # Convert dqa to integer for bit ops
  dqa_i <- as.int(dqa)
  ## more details on bits https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf
  vi_quality   <- app(dqa_i, get_bits, from=0, to=1)   # 0=good; 1=check; 2=cloudy; 3=not produced
  usefulness   <- app(dqa_i, get_bits, from=2, to=5)   # 0 best, higher worse
  aerosol      <- app(dqa_i, get_bits, from=6, to=7)   # 0 climatology, 1 low, 2 mid, 3 high
  adj_cloud    <- app(dqa_i, get_bits, from=8, to=8)   # 1 = yes
  mixed_cloud  <- app(dqa_i, get_bits, from=10, to=10) # 1 = yes
  land_mask    <- app(dqa_i, get_bits, from=11, to=13)
  snow_mask    <- app(dqa_i, get_bits, from=14, to=14)
  shadow_mask  <- app(dqa_i, get_bits, from=15, to=15)
  # Example: "good enough" mask
  mask_detailed <- (vi_quality <= 1) & (usefulness <= 2) & (aerosol <= 2) &
    (adj_cloud == 0) & (mixed_cloud == 0) & (land_mask == 1) & (snow_mask == 0) & (shadow_mask == 0)
  
  evi_corrected <- mask(evi, mask_good, maskvalues = 0, updatevalue = NA)
  evi_corrected <- mask(evi, mask_detailed, maskvalues = 0, updatevalue = NA)
  return(evi)
})

# stack all EVI rasters
evi_stack <- rast(evi_list)

# SpatRaster -> RasterBrick
evi_brick <- raster::brick(evi_stack)

# linear interpolation through time
evi_interp_brick <- raster::approxNA(evi_brick, method = "linear", rule = 2)

# back to SpatRaster
evi_interp <- terra::rast(evi_interp_brick)

# calculate pixel-wise median across all layers
evi_median <- app(evi_interp, fun = median, na.rm = TRUE)

# check result
print(evi_median)
plot(evi_median, main = "Pixel-wise Median EVI")
writeRaster(evi_median,
            file.path(outdir, "EVI_pixelwise_median_h19v07.tif"),
            overwrite = TRUE)


# pixel-wise 90th percentile
evi_p90 <- app(
  evi_interp,
  fun = function(x) quantile(x, probs = 0.90, na.rm = TRUE)
)

# plot
plot(evi_p90, main = "Pixel-wise 90th Percentile of EVI")
# save output
writeRaster(evi_p90,
            file.path(outdir, "EVI_pixelwise_90Percentile_h19v07.tif"),
            overwrite = TRUE)






