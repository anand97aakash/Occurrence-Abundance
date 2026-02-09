library(dplyr)
library(lubridate)
library(stringr)

cap <- read.csv(
  "S:/PhD/Ch4_Occurance Abundance CNN/MAPS data/Full data/MAPS_BANDING_capture_data.csv",
  sep = ",",            # auto-detect
  header = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)


library(dplyr)
library(lubridate)

cap1 <- cap %>%
  mutate(
    year = as.integer(year),
    DATE = suppressWarnings(mdy(DATE)),
    BAND = as.character(BAND),
    SPEC = as.character(SPEC),
    STATION = as.character(STATION),
    STA = as.character(STA)
  ) %>%
  filter(!is.na(year), year >= 2015, year <= 2018) %>%  # keep 2010–2020
  filter(!is.na(BAND), BAND != "") %>%
  distinct()


#### species and their total obs
library(dplyr)

prod_all <- cap1 %>%
  group_by(SPEC) %>%
  summarise(
    n_obs   = n(),
    n_years = n_distinct(year, na.rm = TRUE),
    first_year = suppressWarnings(min(year, na.rm = TRUE)),
    last_year  = suppressWarnings(max(year, na.rm = TRUE)),
    no_of_stations = n_distinct(STATION, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_obs))

## productivity calculation

target_spec <- "SOSP"   # change

prod <- cap1 %>%
  filter(SPEC == target_spec) %>%
  mutate(
    age_class = case_when(
      AGE == 1 ~ "juvenile",
      AGE >= 2 ~ "adult",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(age_class)) %>%
  group_by(STA, STATION, year, age_class) %>%
  summarise(n_ind = n_distinct(BAND), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = age_class, values_from = n_ind, values_fill = 0) %>%
  # effort: if NET is net number and ANET is adjusted net-hours, use ANET if it exists
  left_join(
    cap1 %>%
      filter(SPEC == target_spec) %>%
      group_by(STA, STATION, year) %>%
      summarise(net_effort = sum(as.numeric(ANET), na.rm = TRUE), .groups="drop"),
    by = c("STA","STATION","year")
  ) %>%
  mutate(
    juvenile_rate = juvenile / net_effort,
    adult_rate    = adult / net_effort,
    J_A_ratio     = juvenile / (adult + 1e-6)  # simple productivity ratio
  )
 
unique(prod$STATION)


### Average across years for each station
library(dplyr)

library(dplyr)

station_mean_simple <- prod %>%
  group_by(STA, STATION) %>%
  summarise(
    n_years = n_distinct(year),
    
    adult_mean = mean(adult, na.rm = TRUE),
    juvenile_mean = mean(juvenile, na.rm = TRUE),
    net_effort_mean = mean(net_effort, na.rm = TRUE),
    
    juvenile_rate_mean = mean(juvenile_rate, na.rm = TRUE),
    adult_rate_mean    = mean(adult_rate, na.rm = TRUE),
    J_A_ratio_mean     = mean(J_A_ratio, na.rm = TRUE),
    
    .groups = "drop"
  )
## replace Nan and Inf with 0
station_mean_simple <- station_mean_simple %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.x) | is.infinite(.x), 0, .x)))

#### combine with other station to make full training data with presence and absence

library(dplyr)

# 1) All stations present anywhere in the original raw data (cap)
all_stations <- cap %>%
  transmute(
    STA = as.character(STA),
    STATION = as.character(STATION)
  ) %>%
  distinct()

# 2) Expand station_mean_simple to include stations not in prod/station_mean_simple
station_mean_simple_full <- all_stations %>%
  left_join(station_mean_simple, by = c("STA", "STATION")) %>%
  mutate(
    across(where(is.numeric), ~ ifelse(is.na(.x) | is.nan(.x) | is.infinite(.x), 0, .x))
  )

## make all values have decimal point upto 3 points
station_mean_simple_full <- station_mean_simple_full %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))

## making presence absence column
station_mean_simple_full <- station_mean_simple_full %>%
  mutate(presence_absence = ifelse(n_years > 0, 1L, 0L)) %>%
  relocate(presence_absence, .after = last_col())

### Join the lat long info

library(dplyr)
library(readr)

loc <- read_csv("S:/PhD/Ch4_Occurance Abundance CNN/MAPS data/Full data/MAPS_STATION_location_and_operations.csv",
                show_col_types = FALSE) %>%
  mutate(
    STA = as.character(STA),
    STATION = as.character(STATION),
    DECLAT = as.numeric(DECLAT),
    DECLNG = as.numeric(DECLNG)
  )

station_mean_simple_full <- station_mean_simple_full %>%
  mutate(STA = as.character(STA), STATION = as.character(STATION)) %>%
  left_join(
    loc %>% select(STA, STATION, DECLAT, DECLNG),
    by = c("STA", "STATION")
  )

#### removing bad coordinates
station_mean_simple_full_clean <- station_mean_simple_full %>%
  mutate(
    DECLAT = as.numeric(DECLAT),
    DECLNG = as.numeric(DECLNG)
  ) %>%
  filter(
    !is.na(DECLAT), !is.na(DECLNG),
    DECLAT >= -85.05113, DECLAT <= 85.05113,   # safe for EPSG:3857
    DECLNG >= -180, DECLNG <= 180,
    !(DECLAT == 99.99999 & DECLNG == 1000)     # explicitly remove the dummy pair
  )

### Extract Value from the Raster
library(terra)
r <- rast("Q:\\aanand37\\PhD\\Ch4_Occurance Abundance CNN\\raster\\Embedding\\2017\\mosaicked_embedding_2017.tif")

# 2) Prepare station table (make sure coords are numeric)
stations <- station_mean_simple_full_clean %>%
  mutate(
    DECLAT = as.numeric(DECLAT),
    DECLNG = as.numeric(DECLNG)
  ) %>%
  filter(!is.na(DECLAT), !is.na(DECLNG))

# 3) Make points in lon/lat (EPSG:4326)
pts_ll <- vect(stations, geom = c("DECLNG", "DECLAT"), crs = "EPSG:4326")

# 4) Reproject points to raster CRS (EPSG:3857)
pts_3857 <- project(pts_ll, crs(r))

# Optional sanity check: keep only stations inside raster extent
inside <- relate(pts_3857, as.polygons(ext(r)), relation = "intersects")[,1]
pts_in <- pts_3857[inside, ]
stations_in <- station_mean_simple_full_clean[inside, ]

# 5) Extract all band values
vals <- terra::extract(r, pts_in, ID = FALSE)

# 6) Bind results back to station table
station_with_embed2017 <- bind_cols(stations_in, vals) %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

####Export as shapefile
v <- vect(
  station_with_embed2017,
  geom = c("DECLNG", "DECLAT"),
  crs  = "EPSG:4326"   # lon/lat
)

# 2) (Optional) also save a projected copy in EPSG:3857 (matches your raster)
v_3857 <- project(v, "EPSG:3857")
out_shp_3857 <- "Q:/aanand37/PhD/Ch4_Occurance Abundance CNN/raster/Embedding/2017/SOSP_extracted.shp"

writeVector(v_3857, out_shp_3857, overwrite = TRUE)

##### model training

library(ranger)
library(dplyr)
library(pROC)

# ---- Assume your final table is called station_with_embed2017 (or similar) ----
# It must contain: presence_absence and band columns like A00..A63
df <- station_with_embed2017

# 1) Identify embedding columns
embed_cols <- grep("^A\\d+$|^A\\d\\d$", names(df), value = TRUE)  # matches A00..A63

# If your names are like "A00", "A01", ... this is safest:
embed_cols <- grep("^A\\d\\d$", names(df), value = TRUE)

# 2) Prepare modeling data
dat <- df %>%
  select(presence_absence, all_of(embed_cols)) %>%
  mutate(
    presence_absence = as.factor(presence_absence)   # classification target
  ) %>%
  # remove rows with missing predictors/target
  filter(!is.na(presence_absence)) %>%
  filter(if_all(all_of(embed_cols), ~ !is.na(.x) & !is.nan(.x) & !is.infinite(.x)))

# Quick class balance check
table(dat$presence_absence)

# 3) Train/test split (stratified)
set.seed(42)
idx0 <- which(dat$presence_absence == "0")
idx1 <- which(dat$presence_absence == "1")

test_idx <- c(
  sample(idx0, size = floor(0.2 * length(idx0))),
  sample(idx1, size = floor(0.2 * length(idx1)))
)

train_dat <- dat[-test_idx, ]
test_dat  <- dat[ test_idx, ]

# 4) Handle class imbalance (optional but recommended)
# Weight the minority class higher
cls <- table(train_dat$presence_absence)
class_weights <- c("0" = 1, "1" = as.numeric(cls["0"] / cls["1"]))  # upweight class 1

# 5) Fit Random Forest
rf <- ranger(
  formula = presence_absence ~ .,
  data = train_dat,
  num.trees = 1000,
  mtry = floor(sqrt(length(embed_cols))),
  min.node.size = 5,
  importance = "permutation",
  probability = TRUE,
  class.weights = class_weights,
  seed = 42
)

## Accuracy

# Predicted probabilities for class "1"
pred_prob <- predict(rf, data = test_dat)$predictions[, "1"]

# AUC
auc_val <- as.numeric(auc(test_dat$presence_absence, pred_prob))
auc_val

# Choose threshold (0.5 default)
thr <- 0.5
pred_class <- ifelse(pred_prob >= thr, "1", "0") %>% factor(levels = c("0","1"))

# Confusion matrix
cm <- table(Pred = pred_class, True = test_dat$presence_absence)
cm

# Basic metrics
accuracy <- sum(diag(cm)) / sum(cm)
precision <- cm["1","1"] / max(1, sum(cm["1", ]))
recall <- cm["1","1"] / max(1, sum(cm[ , "1"]))
f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)

c(accuracy = accuracy, precision = precision, recall = recall, f1 = f1, auc = auc_val)

###### Prediction
# rf is your trained ranger model with probability=TRUE
# It was trained on predictors A00..A63 and response presence_absence (factor 0/1)

library(terra)

# Make sure raster layers match the model predictors
train_vars <- rf$forest$independent.variable.names
r_use <- r[[train_vars]]

# terra will call: fun(model, data, ...)
pred_fun <- function(model, data, ...) {
  # sometimes terra passes a matrix; ranger wants a data.frame
  if (!is.data.frame(data)) data <- as.data.frame(data)
  
  pr <- predict(model, data = data)$predictions
  
  # probability of class "1"
  pr[, "1"]
}

prob_raster <- terra::predict(
  r_use,
  rf,                 # <-- model goes here
  fun = pred_fun,
  filename = "Q:/aanand37/PhD/Ch4_Occurance Abundance CNN/raster/Embedding/2017/SOSP_presence_absence_2017.tif",
  overwrite = TRUE,
  na.rm = TRUE
)

prob_raster
