library(tidyr)
library(dplyr)
library(rstan)
library(tibble)
library(readr)
library(quadprog)
library(ggplot2)

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
n_draws <- nrow(mu_post)
sample_index <- sample(n_draws,
                        size = n_frontiers,
                        replace = FALSE)

#target return
mu_target <- seq(0.001,
                 #max(apply(mu_post, 2, max)),
                 0.01,
                 le = 100)

#for the table that will
#be use to create the graph
mu_tib <-  c()
var_opt <- c()
front_id <- c()
aux_id <- 1
cont_aux <- 0

#to store the optimal
#weights
w_opt <- matrix(0, ncol = N)

#Number of assets
N <- ncol(mu_post)

for(idx in sample_index){
  #draw from mu
  mu_draw <- mu_post[idx,]
  
  #draw from sigma
  sig_draw <- sigma_post[idx, ,]
  
  #last iteration is for the classical
  #frontier
  if(cont_aux == n_frontiers){
    mu_draw <- mean_ret
    sig_draw <- cov(ret)
  }
  
  #Solves the optimization
  #problem for each target value
  for(val in mu_target){
    A <- matrix(0, nrow = N,ncol = 2)
    #sum of weights equals 1
    A[,1] <- 1
    
    #At least the target return
    A[,2] <- mu_draw
    
    b0 <- c(1, val)
    sol <- solve.QP(2 * sig_draw,
                    dvec = rep(0, N),
                    Amat = A,
                    bvec = b0,
                    meq = 1)
    mu_tib <- c(mu_tib, val)
    var_opt <- c(var_opt, sol$value)
    front_id <- c(front_id, aux_id)
    w_opt <- rbind(w_opt, sol$solution)
  }
  aux_id <- aux_id + 1
  
}
w_opt <- w_opt[-1,]

#Creates the tibble
chart_tib <- tibble(mu = mu_tib,
                    var = var_opt,
                    frontier = as.factor(front_id))

g <- ggplot(data = chart_tib,
            mapping = 
              aes(x = var,
                  y = mu,
                  colour = frontier))

g_title <- paste(n_frontiers,
                 " Efficient Frontiers")
g + geom_line(lwd = 1.5) + 
  theme(legend.position = 'right') +
  ggtitle(g_title)
  
