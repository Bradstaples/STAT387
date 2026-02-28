### Lab: Deep Learning
### Note: this lab is slightly different from that in the August 2021 printing of ISLRII
### It is based on keras 2.6.0, while the original was based on keras 2.4.0
### The predict_classes() has been deprecated

## A Single Layer Network on the Hitters Data
rm(list=ls())
###
library(ISLR2)
Gitters <- na.omit(Hitters)
n <- nrow(Gitters)
set.seed(13)
ntest <- trunc(n / 3)
testid <- sample(1:n, ntest)
###
lfit <- lm(Salary ~ ., data = Gitters[-testid, ])
lpred <- predict(lfit, Gitters[testid, ])
with(Gitters[testid, ], mean(abs(lpred - Salary)))
###
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
y <- Gitters$Salary
###
library(glmnet)
cvfit <- cv.glmnet(x[-testid, ], y[-testid],
                   type.measure = "mae")
cpred <- predict(cvfit, x[testid, ], s = "lambda.min")
mean(abs(y[testid] - cpred))
###

###
library(torch)
library(luz) # high-level interface for torch
library(torchvision) # for datasets and image transformation
library(torchdatasets) # for datasets we are going to use
library(zeallot)
torch_manual_seed(13)
###

###
modnn <- nn_module(
  initialize = function(input_size) {
    self$hidden <- nn_linear(input_size, 50)
    self$activation <- nn_relu()
    self$dropout <- nn_dropout(0.4)
    self$output <- nn_linear(50, 1)
  },
  forward = function(x) {
    x %>%
      self$hidden() %>%
      self$activation() %>%
      self$dropout() %>%
      self$output()
  }
)
###

###
x <- scale(model.matrix(Salary ~ . - 1, data = Gitters))
###


###
x <- model.matrix(Salary ~ . - 1, data = Gitters) %>% scale()
###

###
modnn <- modnn %>%
  setup(
    loss = nn_mse_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_mae())
  ) %>%
  set_hparams(input_size = ncol(x))
###

###
fitted <- modnn %>%
  fit(
    data = list(x[-testid, ], matrix(y[-testid], ncol = 1)),
    valid_data = list(x[testid, ], matrix(y[testid], ncol = 1)),
    epochs = 20
  )
###

###
plot(fitted)
###


###
npred <- predict(fitted, x[testid, ])
mean(abs(y[testid] - as.numeric(npred)))
###

