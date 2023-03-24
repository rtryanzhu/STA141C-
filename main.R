setwd('D:\\1UCDavis\\2023W\\STA141C\\final project')
library('readr')
library(dplyr)
library(ggplot2)
library(ggrepel)
library(forcats)
library(scales)
library(Cairo) # anti-aliasing
library(patchwork) # subplot
library(ROCR)
library(e1071) # for SVM
library(randomForest) # for RF
library(caret)
library(sjPlot)
library(reshape2) # for melting correlation matrix
library(pROC)
library(broom) # for tidy output
library(class) # for knn
library(yardstick)# for f1 score
library(doParallel)
set.seed(2023)
# CairoWin()

# load processed training and testing data
# this part shall be processed only after generating processed data
load('train.Rdata')
load('test.Rdata')
train = train[,!(names(train) %in% c('nameOrig','nameDest','isFlaggedFraud'))]
test = test[,!(names(test) %in% c('nameOrig','nameDest','isFlaggedFraud'))]
##########

# Preprocessing
# load raw dataset
raw = read_csv('financial_data.csv')
df_ = data.frame(raw)

# no na values are here
colSums(is.na(df_))
#Convert step into 24-hr#
for (i in length(df_)){
  df_$hour = df_$step %% 24
}

df_ = df_[,!(names(df_) %in% c('nameOrig','nameDest','isFlaggedFraud'))]
risky_df = df_[df_$type %in% c('CASH_OUT','TRANSFER'),]
risky_df_fraud = risky_df[risky_df$isFraud==1,]
risky_df_nonfraud = risky_df[risky_df$isFraud==0,]

# don't save it locally
# names(df_)
##########
# EDA
#pie chart of 5 types of transaction #
unique(df_[c('type')])
df_type_table = table(df_$type)

df_type_table_df = data.frame(df_type_table)
sum_of_obs = dim(df_)[1]
df_type_table_df = df_type_table_df %>% mutate(pct = Freq/sum(Freq))
# Add a ratio to table of transaction type 
options(repr.plot.width = 1, repr.plot.height =1)
type_piechart = ggplot(df_type_table_df, 
       aes(x='', y = pct, fill = Var1))+
        geom_bar(width = 1, 
                 stat = 'identity')+
        coord_polar("y")+
        geom_text(aes(x = '',
                      label = scales::percent(pct, accuracy = 0.1)),
                  position = position_stack(vjust = .5),
                  color = 'black')+
        theme_bw()+
        theme(axis.text = element_blank(),
              axis.ticks = element_blank(),
              panel.grid  = element_blank(),
              panel.border = element_blank()
              )+
        labs(y= "", 
             x = "", 
             fill = 'Type of transaction') + 
        scale_fill_brewer(palette="Blues")
ggsave(type_piechart, filename = 'type_piechart_cairo.png', dpi = 300, type = 'cairo',
       width = 8, height = 4, units = 'in')


#__comparison plot of fraud and type of account__#
type_fraud_tab = with(df_, table(type, isFraud))

type_fraud_bp = ggplot(as.data.frame(type_fraud_tab),
       aes(x = type,
           Freq,
           fill = isFraud))+
  geom_col(position = 'dodge')+
  geom_text(aes(label = Freq),
            position = position_dodge(width = 0.9))+
  theme(
        panel.border = element_blank()
  )+
  labs(y= "Count", 
       x = "Type", 
       fill = 'Fraud')

ggsave(type_fraud_bp, filename = 'type_fraud_bp_cairo.png', dpi = 300, type = 'cairo',
       width = 8, height = 4, units = 'in')


#__stacked barplots on transaction amount__#
df1 = fraud_cases[c('amount','isFraud')]
df2 = df_[df_['isFraud']==0,][c('amount','isFraud')] # nonfraud data

p1 = ggplot(data = df1, aes(x = amount))+
  geom_histogram(fill = 'white',
                 colour = 'black',
                 bins = 50)+
  labs(y='Count',x = 'Amount',title = 'Non-fraudulent cases')

p2 = ggplot(data = df2, aes(x = amount))+
  geom_histogram(fill = 'white',
                 colour = 'black',
                 bins = 50)+
  labs(y='Count',x = 'Amount',title = 'Fraudulent cases')
fraud_amount_hist = p1/p2
ggsave(fraud_amount_hist, filename = 'fraud_amount_hist_cairo.png', dpi = 300, type = 'cairo',
       width = 8, height = 4, units = 'in')



#__Plot time vs fraud status histogram__#
fraud_time_hist = ggplot(df_,
       aes(x = hour,
           fill = isFraud
       ))+
  geom_density(alpha = 0.5,
               aes(fill = factor(isFraud)))+
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10))+
  theme_grey()+
  labs(y = 'Density',
       x = 'Time',
       fill = 'Status')
  
ggsave(fraud_time_hist, filename = 'fraud_time_hist_cairo.png', dpi = 300, type = 'cairo',
       width = 8, height = 4, units = 'in')


# save(fraud_cases, file = "fraud.RData")

# proportion of fraud transaction, f/non-f = 8213/6354407
fraud_index = which(df_$isFraud==1) # 8213
nonfraud_index = which(df_$isFraud==0) # 6354407

##########
# since fraud happens only in cash_out and transfer, we can omit other cases to
# reduce computational burden


# # original dataset is too large to fit a model 
sample_size = 100000
risky_df_sub = risky_df
risky_df_sub = risky_df[sample(nrow(risky_df),sample_size),]
# 
# # to be discussed on removal of isFlaggedFraud
risky_df_sub = risky_df_sub[,-(names(risky_df_sub) %in% c('step'))]
# 
# # factorize variables
risky_df_sub$type = as.factor(risky_df_sub$type)
risky_df_sub$isFraud = as.factor(risky_df_sub$isFraud)
# 
# # check type of variable before fitting
str(risky_df_sub)

# Random_sample
random_index = sample(2,nrow(risky_df_sub),replace= TRUE,prob=c(0.7,0.3))
# case tagged as 1 are for training
train = risky_df_sub[random_index ==1, ]
test = risky_df_sub[random_index == 2,]


#__Logistic regression__#
train_fraud = train[train$isFraud==1,]
train_nonfraud = train[train$isFraud==0,]
lr_sub = rbind(sample_n(train_fraud,500),sample_n(train_nonfraud,9500))
test_sub = sample_n(test, 5000)

start = proc.time()
cl = makePSOCKcluster(5) # boosting method, shall be applied to all methods
registerDoParallel(cl)
log_mod = glm(formula = isFraud~., family = binomial, data = lr_sub)
stopCluster(cl)
end = proc.time()
log_time = end - start

summary(log_mod)
# don't run anova on the model
log_mod.results = predict(log_mod, 
                          newdata = subset(test_sub, select = c(1:6,8)),
                          type = 'response')

log_mod.results = ifelse(log_mod.results > 0.5, 1,0)

log_mod_error = mean(log_mod.results != test_sub$isFraud)
confusionMatrix(as.factor(log_mod.results), test_sub$isFraud, mode = "everything", positive="1")

# accuracy = 0.9964, too good?
log_mod_accuracy = 1 - log_mod_error 

p = predict(log_mod, 
            newdata = subset(test_sub, select = c(1:6,8)),
            type = 'response')


pr = prediction(p, test_sub$isFraud)
prf = performance(pr, measure = 'tpr',x.measure = 'fpr')


log_mod_roc = plot(prf)
log_mod_roc
test_roc = roc(test_sub$isFraud~p,plot=TRUE,print.auc = TRUE)
ggsave(test_roc, filename = 'log_mod_roc_auc.png', dpi = 300, type = 'cairo',
       width = 8, height = 4, units = 'in')
# AUC=0.989
#__SVM__#

start = proc.time()
cl = makePSOCKcluster(5) # boosting method, shall be applied to all methods
registerDoParallel(cl)
svm_mod = svm(formula = isFraud~., 
              data = lr_sub,
              type = 'C-classification',
              kernel = 'radial',
              gamma = 0.1,
              cost = 10)
stopCluster(cl)
end = proc.time()
svm_time = end - start


summary(svm_mod)

svm_pred = predict(svm_mod, 
                   newdata = subset(test_sub, select = c(1:6,8)))

# 
# svm_cm = table(test_sub$isFraud, svm_pred)
# svm_cm

confusionMatrix(svm_pred, test_sub$isFraud, mode = "everything", positive="1")
p = predict(svm_mod, 
            newdata = subset(test_sub, select = c(1:6,8)),
            type = 'response')
pr = prediction(as.numeric(p), as.numeric(test_sub$isFraud))
prf = performance(pr, measure = 'tpr',x.measure = 'fpr')
svm_mod_roc = plot(prf)
test_roc = roc(test_sub$isFraud~as.numeric(p),plot=TRUE,print.auc = TRUE)

# auc = 0.846, elbow-shaped for (0,1) response instead of probability
# svm_accuracy = sum(diag(svm_cm))/sum(svm_cm)
# svm_accuracy # accuracy of 99.73%

plot(svm_mod, lr_sub,
     isFraud~type,
     slice = list(hour = 12,amount = 100))

#__RF__#
rf_start = proc.time()
# rf_train = sample_n(train, 10000)
# rf_mod = randomForest(isFraud~., 
#                       data = rf_train,
#                       proximity =TRUE)

cl = makePSOCKcluster(5) # boosting method, shall be applied to all methods
registerDoParallel(cl)
rf_mod = train(isFraud~.,
               data=  lr_sub,
               method = 'rf')

stopCluster(cl)
rf_stop = proc.time()

# Without parallelization
# user  system elapsed 
# 54.00    3.14  152.41 

# With parallelization
# user  system elapsed 
# 0.86    0.08   57.31 

rf_runtime = rf_stop - rf_start
rf_mod # model performance on training set

rf_pred = predict(rf_mod, test_sub)

confusionMatrix(rf_pred, test_sub$isFraud, mode = "everything", positive="1")

# p = predict(rf_mod, 
#             newdata = subset(test_sub, select = c(1:6,8)),
#             type = 'response')

pr = prediction(as.numeric(rf_pred), as.numeric(test_sub$isFraud))
prf = performance(pr, measure = 'tpr',x.measure = 'fpr')
svm_mod_roc = plot(prf)

test_roc = roc(test_sub$isFraud~as.numeric(rf_pred),plot=TRUE,print.auc = TRUE)
rf_cm = confusionMatrix(rf_pred, test_sub$isFraud,positive = "1")

rf_cm # accuracy = 99.82%
plot(rf_mod)
# AUC = 0.922
#__KNN__#
knn_sub = lr_sub
knn_sub$type = as.numeric(knn_sub$type)
knn_sub$isFraud = as.numeric(knn_sub$isFraud)
sample_id = sample(1:nrow(lr_sub),size=nrow(lr_sub)*0.7,replace = FALSE)
knn_train = knn_sub[sample_id,]
knn_test = knn_sub[-sample_id,]
knn_train_lab = knn_sub[sample_id,7]
knn_test_lab = knn_sub[-sample_id,7]
start = proc.time()
cl = makePSOCKcluster(5) # boosting method, shall be applied to all methods
registerDoParallel(cl)
knn_mod = knn(train = knn_train,
              test = knn_test,
              cl = knn_train_lab,
              k = 10)
stopCluster(cl)
stop = proc.time()
knn_time = stop-start

confusionMatrix(table(knn_mod,knn_test_lab))


pr = prediction(as.numeric(knn_mod), as.numeric(knn_test$isFraud))
prf = performance(pr, measure = 'tpr',x.measure = 'fpr')
knn_mod_roc = plot(prf)

test_roc = roc(knn_test$isFraud~as.numeric(knn_mod),plot=TRUE,print.auc = TRUE)
#auc = 0.870