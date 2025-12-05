#!/usr/bin/env Rscript

## ---- Setup ----
library(future)
library(tidyverse)
library(readr)
library(tidymodels)
library(lubridate)

plan(multisession, workers = 8)

## ---- Load & preprocess ----
df <- read_csv("~/projects/Project2/Austin_Crash_Report_Data_-_Crash_Level_Records_20250407.csv.gz")

df <- df %>%
  mutate(
    injury = factor(tot_injry_cnt > 0,
                    levels = c(FALSE, TRUE),
                    labels = c("no_injury", "injury")),
    onsys_fl = factor(onsys_fl),
    private_dr_fl = factor(private_dr_fl),
    crash_time = mdy_hms(`Crash timestamp (US/Central)`),
    hour  = hour(crash_time),
    wday  = wday(crash_time, label = TRUE),
    month = month(crash_time, label = TRUE)
  )

## ---- Split ----
df_split <- initial_split(df, prop = 0.25)
train <- training(df_split)
test  <- testing(df_split)

## ---- Recipe ----
rec <- recipe(
  injury ~ latitude + longitude + crash_speed_limit +
           onsys_fl + units_involved + road_constr_zone_fl +
           hour + wday + month,
  data = train
)

## ---- Model ----
model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(model)

## ---- Cross-validation ----
folds <- vfold_cv(train, v = 10)

## ---- Grid ----
grid <- grid_regular(
  mtry(range = c(2, 8)),
  min_n(range = c(2, 20)),
  levels = 5
)

## ---- Tuning ----
res <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

## ---- Save results ----
scratch <- Sys.getenv("SCR", unset = "/stor/scratch/WCAAR/rhyan_scratch")
outdir  <- file.path(scratch, "rf_tuning_output")

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

saveRDS(res, file.path(outdir, "tuning_results.rds"))

best_res <- select_best(res, metric = "roc_auc")
write_csv(as.data.frame(best_res), file.path(outdir, "best_params.csv"))

cat("\n========= Finished Successfully =========\n")
print(best_res)
