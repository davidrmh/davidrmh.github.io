library(tidyr)
library(dplyr)
library(rstan)
library(tibble)
library(readr)
library(quadprog)

### Reads the data ###
path <- c("./data/EUR_MXN.csv",
          "./data/USD_MXN.csv",
          "./data/GBP_MXN.csv")
eur <- read_csv(path[1], 
                col_types = 'Ddddddd',
                na = c("", "NA", "null"))
usd <- read_csv(path[2], 
                col_types = 'Ddddddd',
                na = c("", "NA", "null"))
gbp <- read_csv(path[3], 
                col_types = 'Ddddddd',
                na = c("", "NA", "null"))

### Keeps only Date and Adj Close Columns
### Drops na too 
eur <- eur %>%
  select(Date, `Adj Close`) %>%
  drop_na()

usd <- usd %>%
  select(Date, `Adj Close`) %>%
  drop_na()

gbp <- gbp %>%
  select(Date, `Adj Close`) %>%
  drop_na()

### Renames Adj Close Column
names(eur) <- c('Date', 'eur_mxn')
names(usd) <- c('Date', 'usd_mxn')
names(gbp) <- c('Date', 'gbp_mxn')

### joins in one tibble
data <- inner_join(eur, usd, by = "Date")
data <- inner_join(data, gbp, by = 'Date')

write_csv(data, path = './data/exchange_rates.csv')

data <- read_csv('./data/exchange_rates.csv')
### Function to calculate the log returns

log_ret <- function(data){
  #INPUT
  #data: A tibble with
  #At least two columnas one of which
  #has name 'Date'
  #Data MUST be ordered from
  #oldest to newest
  
  #OUTPUT
  #A tibble with the log returns
  
  #Removes Date column
  prices <- data[,colnames(data) != 'Date']
  
  #number of observations
  n_obs <- nrow(prices)
  
  #log returns
  as_tibble(log(prices[2:n_obs,]/prices[1:(n_obs - 1),]))
}

ret <- log_ret(data)
write_csv(ret, './data/log_ret.csv')
ret <- read_csv('./data/log_ret.csv')

#Vector of mean returns
mean_ret <- apply(ret, 2, mean)

#Covariance matrix
cov_mat <- cov(ret)

#Data for STAN
#int<lower = 5> T;
#int<lower= 2> N;
#real<lower = N - 1> nu;
#real<lower = 0> tau;
#vector[N] eta;
#matrix[T, N] R;
#matrix[N, N] omega;
T <- nrow(ret)
N <- ncol(ret)
nu <- 12
data_stan <- list(
  T = T,
  N = N,
  nu = nu,
  tau = 200,
  eta = mean_ret,
  R = as.matrix(ret),
  omega = cov_mat * (nu - N -1)
)

#Fitting the model
#This takes a while!
fit <- stan(
  file = "bay_port.stan",
  data = data_stan,
  chains = 4,
  warmup = 1000,
  iter = 2000,
  cores = 2
)

#Saves the fitted model
saveRDS(fit, 'stan_fit.rds')
fit <-readRDS('stan_fit.rds')

#Some diagnostics
traceplot(fit, nrow = 4, pars = c('mu', 'sigma'))
print(fit, pars = c('mu', 'sigma'))

#Extract draws from the posterior
list_of_draws <- extract(fit)
#draws from the posterior distribution of sigma
sigma_post <- list_of_draws$sigma
#draws from the posterior distribution of mu
mu_post <- list_of_draws$mu

#Solves the optimization
#problem for distinct
#draws of the posterior
#parameters.
#Each draw will have an
#efficient frontier associated
#with it
n_frontiers <- 10
set.seed(54321)
n_draws <- nrow(mu_post)
sample_index <- sample(n_draws,
                       size = n_frontiers,
                       replace = FALSE)

#target (annual) return
mu_target <- seq(0.02,
                 #max(apply(mu_post, 2, max)),
                 0.20,
                 le = 100)

#Number of assets
N <- ncol(mu_post)

#aux for detecting last
#frontier
aux_last <- 0

par(bg = '#EEEEEC',
    mfcol = c(ceiling(n_frontiers / 2),2))
for(idx in sample_index){
  aux_last <- aux_last + 1
  
  #for the table that will
  #be use to create the graph
  mu_tib <-  c()
  var_opt <- c()
  #Last frontier is classical
  #framework
  if(aux_last == 1){
    mu_draw <- mean_ret
    sig_draw <- cov_mat
  }
  else{
    #draw from mu
    mu_draw <- mu_post[idx,]
    
    #draw from sigma
    sig_draw <- sigma_post[idx, ,]
  }
  
  #Solves the optimization
  #problem for each target value
  for(val in mu_target){
    
    A <- matrix(0, nrow = N,ncol = 2)
    #sum of weights equals 1
    A[,1] <- 1
    
    #the target return constrain
    A[,2] <- mu_draw * 252
    
    b0 <- c(1, val)
    sol <- solve.QP(2 * 252 * sig_draw,
                    dvec = rep(0, N),
                    Amat = A,
                    bvec = b0,
                    meq = 2)
    mu_tib <- c(mu_tib, val)
    var_opt <- c(var_opt, sol$value )
  }
  
  #Creates the tibble
  #For the chart
  chart_tib <- tibble(mu =  100 * mu_tib,
                      std = 100 * sqrt(var_opt))
  
  main <- 'Efficient Frontier'
  
  if(aux_last == 1){
    main <- 'Efficient Frontier \n Classical Framework'
  }
  
  plot(chart_tib$std, chart_tib$mu,
       col = 'blue', lwd = 2.5,
       main = main,
       sub = 'Annualized figures',
       xlab = 'Standard Deviation %',
       ylab = 'Expected return %',
       type = 'l',
       ylim = c(100 * min(mu_tib), 100 * max(mu_tib)))
  grid(col = 'black', lwd = 1.5)
}

  
