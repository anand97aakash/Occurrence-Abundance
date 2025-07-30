#args <- commandArgs(trailingOnly = TRUE)

library(keras)
library(raster)
library(caret)
library(abind)

Sys.setenv(CUDA_VISIBLE_DEVICES = "0")
options(stringsAsFactors = FALSE)

# loading the library
suppressPackageStartupMessages(library(disdat))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(forcats)) # for handling factor variables

use_condaenv("C:\\Users\\aanand37\\AppData\\Local\\ESRI\\conda\\envs\\tf-gpu")


# set output directory for prediction csv files:
outdir <- "F:\\akash\\PROJECTS\\sasha"
if(!file.exists(outdir)){
  dir.create(file.path(outdir))
  print("The directory is created")
}


######## make 20 validation set

# Load required library
library(dplyr)

# Read the CSV file
input_file <- "F:\\akash\\aa_PhD\\Occurance Abundance CNN\\species\\Occurance\\presence_absence_forest_HAFL.csv"
df <- read.csv(input_file)

# Set seed for reproducibility (optional)
set.seed(42)

# Calculate number of rows to sample (20%)
sample_size <- floor(0.20 * nrow(df))

# Randomly sample 20% of rows
sampled_rows <- df %>% sample_n(sample_size)

# Get the remaining 80%
remaining_rows <- anti_join(df, sampled_rows)

# Write the sampled rows to a new CSV
write.csv(sampled_rows, "F:\\akash\\aa_PhD\\Occurance Abundance CNN\\species\\Occurance\\presence_absence_forest_20_percent_HAFL.csv", row.names = FALSE)

# Optionally, save the remaining 80% as well
write.csv(remaining_rows, "F:\\akash\\aa_PhD\\Occurance Abundance CNN\\species\\Occurance\\presence_absence_forest_80_percent_HAFL.csv", row.names = FALSE)

############################



# provide names for regions to be modeled - here we model all 6:
#regions <- c("AWT", "CAN", "NSW", "NZ", "SA", "SWI")
#regions <- c("NSW")
regions <- c("CAN")
# specify names of all categorical variables across all regions:
categoricalvars <- c("ontveg", "vegsys", "toxicats", "age", "calc")
# the list of uncorrelated variables
covars <- list(
  AWT = c("bc04",  "bc05",  "bc06",  "bc12",  "bc15",  "slope", "topo", "tri"),
  CAN = c("alt", "asp2", "ontprec", "ontslp", "onttemp", "ontveg", "watdist"), 
  NSW = c("cti", "disturb", "mi", "rainann", "raindq", "rugged", "soildepth", "soilfert", "solrad", "tempann", "topo", "vegsys"), 
  NZ = c("age", "deficit", "hillshade", "mas", "mat", "r2pet", "slope", "sseas", "toxicats", "tseas", "vpd"), 
  SA = c("sabio12", "sabio15", "sabio17", "sabio18", "sabio2", "sabio4", "sabio5", "sabio6"), 
  SWI = c("bcc", "calc", "ccc", "ddeg", "nutri", "pday", "precyy", "sfroyy", "slope", "sradyy", "swb", "topo")
)

chip_sizes <- c(32) #,64,128)

#lr_rate <- c(0.01, 0.05,0.001,0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001, 0.000005) # add 0.0001 when doing 16 batches

batch <-  c(16)  #16 is remaining

normalize_raster <- function(raster_stack) {
  if (!inherits(raster_stack, "RasterStack")) {
    stop("Input must be a RasterStack.")
  }
  num_bands <- nlayers(raster_stack)
  normalized_raster_stack <- stack()
  for (i in 1:num_bands) {
    band_values <- getValues(raster_stack[[i]])
    mean_value <- mean(band_values, na.rm = TRUE)
    std_dev <- sd(band_values, na.rm = TRUE)
    normalized_values <- (band_values - mean_value) / std_dev
    raster_object_normalized <- raster_stack[[i]]
    values(raster_object_normalized) <- normalized_values
    normalized_raster_stack <- addLayer(normalized_raster_stack, raster_object_normalized)
  }
  names(normalized_raster_stack) <- paste0("Band_", 1:num_bands)
  return(normalized_raster_stack)
}

pad_raster_layer <- function(raster_layer, extra_rows, extra_cols, extra_value) {
  nrows <- nrow(raster_layer)
  ncols <- ncol(raster_layer)
  larger_raster_layer <- raster(nrow = nrows + 2 * extra_rows, ncol = ncols + 2 * extra_cols, crs = crs(raster_layer))
  extent(larger_raster_layer) <- extent(raster_layer) + c(-extra_cols * res(raster_layer)[1], extra_cols * res(raster_layer)[1], -extra_rows * res(raster_layer)[2], extra_rows * res(raster_layer)[2])
  values(larger_raster_layer) <- extra_value
  subset_raster <- raster_layer[]
  larger_raster_layer[(extra_rows + 1):(extra_rows + nrows), (extra_cols + 1):(extra_cols + ncols)] <- subset_raster
  return(larger_raster_layer)
}

############# Normalize the raster stack
## raster stack
# raster_folder <- "F:\\akash\\aa_PhD\\Occurance Abundance CNN\\raster\\rasters"
# 
# # List all raster files (e.g., .tif files)
# raster_files <- list.files(raster_folder, pattern = "\\.tif$", full.names = TRUE)
# 
# # Stack the rasters
# raster_stack <- stack(raster_files)
# 
# ##
# ##raster_stack<-brick("F:\\akash\\aa_PhD\\Occurance Abundance CNN\\raster\\Stacked.tif")
# ##names(raster_stack) <- c("Shrubland", "Precip", "Grassland", "Forest", "DEM", "CumDHI", "VarDHI", "Tmax")
# 
# extent_stack <- extent(raster_stack)
# new_raster <- raster(extent_stack, res = res(raster_stack), crs = crs(raster_stack))
# filled_raster_stack <- stack(lapply(1:nlayers(raster_stack), function(i) setValues(new_raster, 0)))
# 
# for (i in 1:nlayers(raster_stack)) {
#   layer_values <- raster_stack[[i]]
#   filled_raster_stack[[i]] <- layer_values
# }
# 
# filled_raster_stack <- reclassify(filled_raster_stack, cbind(NA, 0))
# extra_rows <- chip_sizes
# extra_cols <- chip_sizes
# extra_value <- 0
# 
# larger_raster_stack <- stack(lapply(1:nlayers(filled_raster_stack), function(i) {
#   pad_raster_layer(filled_raster_stack[[i]], extra_rows, extra_cols, extra_value)
# }))
# 
# replace_na_with_zero <- function(x) {
#   x[is.na(x)] <- 0
#   return(x)
# }
# 
# raster_stack_zero_na <- calc(larger_raster_stack, replace_na_with_zero)
# raster_stack_zero_na_stack <- stack(raster_stack_zero_na)
# env_data <- normalize_raster(raster_stack_zero_na_stack)
# #names(env_data) <- c("Shrubland", "Precip", "Grassland", "Forest", "DEM", "CumDHI", "VarDHI", "Tmax")
# writeRaster(env_data,"F:\\akash\\aa_PhD\\Occurance Abundance CNN\\raster\\Normalized_Stacked_8bands.tif")

env_data <- brick("F:\\akash\\aa_PhD\\Occurance Abundance CNN\\raster\\Normalized_Stacked_8bands.tif")
#env_data <- raster("X:\\aanand37\\from_GAIA\\aa_Chapter 1\\predictors\\4x4\\CumDHI_4x4_average.tif")
#######################



set.seed(40)  # Set a global seed for reproducibility


for (batch in batch) {
  for (chip in chip_sizes) {
    #for (lr in lr_rate) {
    n <- 0
    for (r in regions) {
      presences <-  read.csv("F:\\akash\\aa_PhD\\Occurance Abundance CNN\\species\\Occurance\\presence_absence_forest_80_percent_HAFL.csv")
      #background <- read.csv(paste0("X:\\aanand37\\PhD\\Elith et al 2006\\papers\\2021 paper with new models and codes\\ecm1486-sup-0003-datas1\\DataS1\\background_50k\\", r, ".csv"))
      # Load environmental data and species occurrence data
      # raster_dir <- paste0("X:\\aanand37\\from_GAIA\\aa_Chapter 1\\predictors\\4x4\\Clipped_contUS\\Albers")
      # raster_files <- list.files(raster_dir, pattern = "*.tif$", full.names = TRUE)
      # raster_stack <- stack(raster_files)
      # 
      
      
      
      chip_size <- chip * res(env_data)[1]
      bands <- nlayers(env_data)
      desired_dimensions <- c(chip, chip, bands)
      
      # for (grp in unique(presences$group)) {
      #   evaluation <- disEnv(r)			#, grp)
      #   presence_subset <- presences[presences$group == grp, ]
      #   
      #   for (i in 1:ncol(presences)) {
      #     if (colnames(presences)[i] %in% categoricalvars) {
      #       fac_col <- colnames(presences)[i]
      #       presences[, fac_col] <- as.factor(presences[, fac_col])
      #       evaluation[, fac_col] <- as.factor(evaluation[, fac_col])
      #       evaluation[, fac_col] <- forcats::fct_expand(evaluation[, fac_col], levels(presences[, fac_col]))
      #       evaluation[, fac_col] <- forcats::fct_relevel(evaluation[, fac_col], levels(presences[, fac_col]))
      #     }
      #   }
      
      evaluation <- read.csv("F:\\akash\\aa_PhD\\Occurance Abundance CNN\\species\\Occurance\\presence_absence_forest_20_percent_HAFL.csv")
      chip_list_eval_out <- list()
      
      for (i in 1:nrow(evaluation)) {
        lon <- evaluation[i, "PointLongi"]
        lat <- evaluation[i, "PointLatit"]
        bbox <- extent(lon - chip_size / 2, lon + chip_size / 2, lat - chip_size / 2, lat + chip_size / 2)
        chip_eval <- crop(env_data, bbox)
        chip_array_eval <- array(chip_eval, dim = c(dim(chip_eval)[1], dim(chip_eval)[2], dim(chip_eval)[3]))
        
        if (all(dim(chip_array_eval) == desired_dimensions)) {
          chip_list_eval_out[[i]] <- chip_array_eval
        } else {
          print(paste("Skipping chip", i, "due to mismatched dimensions"))
        }
      }
      
      filtered_chip_list_eval <- chip_list_eval_out[sapply(chip_list_eval_out, function(arr) all(dim(arr) == c(chip, chip, bands)))]
      pr_bg_chips_eval <- abind(filtered_chip_list_eval, along = 4)
      pr_bg_chips_eval <- aperm(pr_bg_chips_eval, c(4, 1, 2, 3))
      pr_bg_chips_eval[is.na(pr_bg_chips_eval)] <- 0
      
      #species <- unique(presence_subset$spid)
      # Remove "nsw30" if it exists in the list as per elith paper
      #species <- setdiff(species, "nsw30")
      
      #for (s in species) {
        #n <- n + 1
        set.seed(sum(presences$occ) + 300 + n)  # Seed set before species loop
        sp_presence <- presences#[presences$spid == s, ]
        
        chip_list <- list()
        label_vector <- numeric()
        
        for (i in 1:nrow(sp_presence)) {
          lon <- sp_presence[i, "PointLongi"]
          lat <- sp_presence[i, "PointLatit"]
          label <- sp_presence[i, "occurance"]
          bbox <- extent(lon - chip_size / 2, lon + chip_size / 2, lat - chip_size / 2, lat + chip_size / 2)
          chip_raster <- crop(env_data, bbox)
          chip_array <- array(chip_raster, dim = c(dim(chip_raster)[1], dim(chip_raster)[2], dim(chip_raster)[3]))
          
          if (all(dim(chip_array) == desired_dimensions)) {
            chip_list[[i]] <- chip_array
            label_vector <- c(label_vector, label)
          } else {
            print(paste("Skipping chip", i, "due to mismatched dimensions"))
          }
        }
        
        filtered_chip_list <- chip_list[sapply(chip_list, function(arr) all(dim(arr) == c(chip, chip, bands)))]
        presence_chips <- abind(filtered_chip_list, along = 4)
        presence_chips <- aperm(presence_chips, c(4, 1, 2, 3))
        presence_chips[is.na(presence_chips)] <- 0
        presence_array <- abind(presence_chips, along = 1)
        
        # num_presence_records <- nrow(presence_array)
        # selected_background <- background[sample(nrow(background), num_presence_records), ]
        # 
        # chip_list <- list()
        # label_vector <- numeric()
        # 
        # for (i in 1:nrow(selected_background)) {
        #   lon <- selected_background[i, "x"]
        #   lat <- selected_background[i, "y"]
        #   label <- selected_background[i, "occ"]
        #   bbox <- extent(lon - chip_size / 2, lon + chip_size / 2, lat - chip_size / 2, lat + chip_size / 2)
        #   chip_raster <- crop(env_data, bbox)
        #   chip_array <- array(chip_raster, dim = c(dim(chip_raster)[1], dim(chip_raster)[2], dim(chip_raster)[3]))
        #   
        #   if (all(dim(chip_array) == desired_dimensions)) {
        #     chip_list[[i]] <- chip_array
        #     label_vector <- c(label_vector, label)
        #   } else {
        #     print(paste("Skipping chip", i, "due to mismatched dimensions"))
        #   }
        # }
        # 
        # filtered_chip_list_background <- chip_list[sapply(chip_list, function(arr) all(dim(arr) == c(chip, chip, bands)))]
        # background_chips <- abind(filtered_chip_list_background, along = 4)
        # background_chips <- aperm(background_chips, c(4, 1, 2, 3))
        # background_chips[is.na(background_chips)] <- 0
        
        combined_arrays <- presence_array #abind(presence_array, background_chips, along = 1)
        labels <- sp_presence[,"occurance"] #c(rep(1, nrow(presence_array)), rep(0, nrow(background_chips)))
        set.seed(123)  # Seed set before shuffling
        perm_indices <- sample(length(labels))
        full_data <- combined_arrays[perm_indices, , , ]
        full_labels <- labels[perm_indices]
        train_indices <- createDataPartition(full_labels, p = 0.8, list = FALSE)
        train_raster_array <- full_data[train_indices, , , , drop = FALSE]
        train_label_vector <- full_labels[train_indices]
        test_raster_array <- full_data[-train_indices, , , , drop = FALSE]
        test_label_vector <- full_labels[-train_indices]
        
        # Define VGG-16 architecture
        # model <- keras_model_sequential() %>%
        #   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same", input_shape = c(chip, chip, bands), kernel_regularizer = regularizer_l2(0.00001)) %>%
        #   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        #   
        #   layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        #   
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        #   
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   #layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        #   
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   #layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        #   
        #   layer_flatten() %>%
        #   layer_dense(units = 4096, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
        #   layer_dropout(rate = 0.5) %>%
        #   layer_dense(units = 4096, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
        #   layer_dropout(rate = 0.5) %>%
        #   layer_dense(units = 1000, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
        #   layer_dense(units = 1, activation = "sigmoid")
        
        # # Define VGG-19 architecture
        # model <- keras_model_sequential() %>%
        #   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same", input_shape = c(chip, chip, bands), kernel_regularizer = regularizer_l2(0.00001)) %>%
        #   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        # 
        #   layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        # 
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        # 
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        # 
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
        #   layer_batch_normalization() %>%
        #   layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
        #   layer_dropout(rate = 0.25) %>%
        # 
        #   layer_flatten() %>%
        #   layer_dense(units = 4096, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
        #   layer_dropout(rate = 0.5) %>%
        #   layer_dense(units = 4096, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
        #   layer_dropout(rate = 0.5) %>%
        #   layer_dense(units = 1000, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
        #   layer_dense(units = 1, activation = "sigmoid")
        
        # #Define the model architecture
        model <- keras_model_sequential() %>%
          layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same", input_shape = c(chip, chip, bands), kernel_initializer = initializer_he_normal()) %>%
          #layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          layer_batch_normalization() %>%
          layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
          layer_dropout(rate = 0.25) %>%
          
          layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          layer_batch_normalization() %>%
          layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
          layer_dropout(rate = 0.25) %>%
          
          layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          #layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          layer_batch_normalization() %>%
          layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
          layer_dropout(rate = 0.25) %>%
          
          layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          #layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          #layer_conv_2d(filters = 512, kernel_size = c(3, 3), activation = "relu", padding = "same", kernel_initializer = initializer_he_normal()) %>%
          layer_batch_normalization() %>%
          layer_max_pooling_2d(pool_size = c(2, 2), strides = 2) %>%
          layer_dropout(rate = 0.25) %>%
          
          layer_flatten() %>%
          layer_dense(units = 2048, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(units = 2048, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
          layer_dropout(rate = 0.5) %>%
          layer_dense(units = 1000, activation = "relu", kernel_initializer = initializer_he_normal()) %>%
          layer_dense(units = 1, activation = "sigmoid")
        
        # Compile the model
        model %>% compile(
          loss = "binary_crossentropy",
          optimizer = optimizer_adam(learning_rate = 0.001),
          metrics = c("accuracy")
        )
        
        reduce_lr <- callback_reduce_lr_on_plateau(
          monitor = 'val_loss',
          factor = 0.2,
          patience = 5,
          min_lr = 1e-6
        )
        #print_lr = PrintLearningRate()
        
        # Callbacks for early stopping and learning rate reduction
        early_stopping <- callback_early_stopping(monitor = "val_loss", patience = 10)
        #reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.2, patience = 5, min_lr = 0.000005)
        
        ptm <- proc.time()
        # Train the model
        history <- model %>% fit(
          train_raster_array, train_label_vector,
          epochs = 100,
          batch_size = batch,
          validation_data = list(test_raster_array, test_label_vector),
          callbacks = list(reduce_lr,early_stopping)    #, reduce_lr)
        )
        t <- proc.time() - ptm
        
        final_lr <- k_get_value(model$optimizer$learning_rate)
        
        out_file <- evaluation[, 1:4]
        out_file$spid <- sp_presence$spid[1]
        #out_file$region <- r
        out_file$model <- "CNN_VGG19"
        
        batch_size <- 50
        predictions <- numeric(nrow(pr_bg_chips_eval))
        
        for (i in seq(1, nrow(pr_bg_chips_eval), by = batch_size)) {
          indices <- i:min(i + batch_size - 1, nrow(pr_bg_chips_eval))
          current_batch <- pr_bg_chips_eval[indices, , ,]
          predictions[indices] <- as.numeric(predict(model, current_batch, device = "cpu"))
        }
        
        out_file$prediction <- predictions
        out_file$time <- t[3]
        write.csv(out_file, sprintf("%s/%s_CNN_%s_%s_batch%s_with_dropout0.25.csv", outdir, s, chip,final_lr, batch), row.names = FALSE)
        print(n)
      }
    }
  }
}
}

After line 453
# Round the predictions
out_file$rounded_prediction <- round(out_file$prediction)

# Compare with occurance: 1 if correct, 0 if not
out_file$correct <- ifelse(out_file$rounded_prediction == out_file$occurance, 1, 0)

# Count total correct predictions
correct_count <- sum(out_file$correct)

# Calculate accuracy
accuracy <- correct_count / nrow(out_file)

# Print results
cat("Correct predictions:", correct_count, "\n")
cat("Accuracy:", round(accuracy * 100, 2), "%\n")

##############
## Making Maps

library(raster)
library(myspatial)
library(disdat)
library(parallel)
library(doParallel)
library(terra)
library(sf)

# shapefile_sp <- as(shapefile, "Spatial")
# clipped <- crop(env_data, shapefile_sp)
# clipped <- mask(clipped, shapefile_sp)
# writeRaster(clipped, "clipped_stack.tif", overwrite=TRUE)

r <- brick("F:\\akash\\aa_PhD\\Occurance Abundance CNN\\raster\\WesternUS_Normalized_Stacked_8bands.tif")
chip_size <- 32


slice_raster_parallel_foreach <- function(raster_data, chip_size,input_raster) {
  #raster_data<- r_mask
  nrows <- nrow(raster_data)
  ncols <- ncol(raster_data)
  
  # Calculate padding
  pad_rows <- 16#((chip_size-1)/2)# - 1
  pad_cols <- 16#((chip_size-1)/2)# - 1
  
  min_value <- cellStats(raster_data, stat = 'min', na.rm = TRUE)
  # Pad the raster if needed
  if (pad_rows > 0 || pad_cols > 0) {
    padded_extent <- extent(raster_data)
    
    # Adjust the extent symmetrically in all directions
    raster_data <- extend(
      raster_data,
      extent(
        xmin(padded_extent) - pad_cols * res(raster_data)[1],
        xmax(padded_extent) + pad_cols * res(raster_data)[1],
        ymin(padded_extent) - pad_rows * res(raster_data)[2],
        ymax(padded_extent) + pad_rows * res(raster_data)[2]
      ),
      value =  0#0.5  # You can change the fill value here if needed
    )
  }
  
  # Parallel processing of rows using foreach
  chips <- foreach(i = seq(1, nrow(input_raster) , by = 1), .combine = 'c', .packages = c('raster')) %dopar% {
    row_chips <- list()
    for (j in seq(1, ncol(input_raster), by = 1)) {
      crop_extent <- extent(
        xmin(raster_data) + (j - 1) * res(raster_data)[1],
        xmin(raster_data) + (j + chip_size - 1) * res(raster_data)[1],
        ymax(raster_data) - (i + chip_size - 1) * res(raster_data)[2],
        ymax(raster_data) - (i - 1) * res(raster_data)[2]
      )
      
      chip <- crop(raster_data, crop_extent)
      if (nrow(chip) == chip_size && ncol(chip) == chip_size) {
        chip_array <- round(as.array(chip), 4)
        row_chips[[length(row_chips) + 1]] <- chip_array
      }
    }
    return(row_chips)
  }
  
  return(chips)
}

library(sf)

# Load shapefile once outside the loop
#shapefile_path <- "X:\\aanand37\\PhD\\Elith et al 2006\\akash code edits\\modelling_codes\\Akash CNN codes\\CNN\\nz30\\nz_100splits_polygon.shp"
shapefile <- st_read(shapefile_path)
num_polygons <- nrow(shapefile_sp)#(shapefile)

# Setup parallel backend
num_cores <- 90
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Main processing loop
for (n in num_polygons) { # num_polygons 91:100
  tryCatch({
    single_polygon <- shapefile[n, ]
    shapefile_sp <- as(single_polygon, "Spatial")
    #buffer_shapefile <- st_buffer(single_polygon, dist = 1600)  # Buffer of 1600 meters, my pixel size were 100 and chip size 32 therefore 100x16
    buffer_shapefile <- st_buffer(sf::st_as_sf(single_polygon), dist = 1600)
    
    cropped_raster <- crop(r, extent(buffer_shapefile))
    masked_raster <- mask(cropped_raster, buffer_shapefile)
    
    r_mask <- reclassify(masked_raster, cbind(NA, 0))
    
    # Slice raster in parallel
    raster_chips <- slice_raster_parallel_foreach(r_mask, chip_size, r_mask)
    
    # Batch processing for predictions
    batch_size <- 5000
    num_chips <- length(raster_chips)
    num_batches <- ceiling(num_chips / batch_size)
    predictions <- vector("list", num_chips)
    
    for (i in 1:num_batches) {
      start_idx <- (i-1) * batch_size + 1
      end_idx <- min(i * batch_size, num_chips)
      
      batch <- array(
        data = unlist(raster_chips[start_idx:end_idx]), 
        dim = c(chip_size, chip_size, nlayers(r), end_idx - start_idx + 1)
      )
      
      batch <- aperm(batch, c(4,1,2,3))
      
      # Predict using the model
      batch_predictions <- predict(model, batch)
      
      predictions[start_idx:end_idx] <- split(batch_predictions, row(batch_predictions))
    }
    
    # Convert predictions to raster
    predictions_vector <- unlist(predictions)
    predictions_matrix <- matrix(predictions_vector, nrow = nrow(r_mask), ncol = ncol(r_mask), byrow = TRUE)
    predictions_raster <- raster(predictions_matrix)
    
    # Set extent and CRS
    extent(predictions_raster) <- extent(r_mask)
    crs(predictions_raster) <- crs(r_mask)
    final_pred <- mask(predictions_raster,single_polygon)
    # Save the raster
    writeRaster(final_pred, paste0("F:\\akash\\aa_PhD\\Elith et al 2006\\papers\\2021 paper with new models and codes\\predicted rasters\\Aug_NZ100_",n,".tif"), overwrite = TRUE)
    print(n)
  }, error = function(e) {
    message("Error with polygon ", n, ": ", e)
  })
}

# Close the parallel backend
stopCluster(cl)

# Visualize the predicted raster
plot(pred_raster)
