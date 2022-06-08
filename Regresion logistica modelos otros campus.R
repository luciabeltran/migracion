#AUTOR: Lucía Beltrán Rocha
#email:beltranrochalucia@gmail.com
#packages
library(caTools)
library(ggplot2)
library(Epi)
library(rpart, lib.loc = "C:/Program Files/R/R-4.1.0/library")
library(factoextra)
library(rpart.plot)
library(caret)
library(ROCR)
library("rfVarImpOOB")
library(texreg)
library(coefplot)
library(effects)
library(AICcmodavg)
library(cvAUC)

# Dataset
log.reg =read.csv ('DESERCION A OTROS CAMPUS.csv')

#Data type : Categorical, Nominal, Numeric, Continuous, Discrete
log.reg$cve_escuela<-as.factor(log.reg$cve_escuela)
log.reg$VExperienciacInstitucion<-as.factor(log.reg$VExperienciacInstitucion)
log.reg$cve_programa<-as.factor(log.reg$cve_programa)
log.reg$cve_escuelaingreso<-as.factor(log.reg$cve_escuelaingreso)
log.reg$tipo_escuela<-as.factor(log.reg$tipo_escuela)
log.reg$cve_prospecto<-as.factor(log.reg$cve_prospecto)
log.reg$genero<-as.factor(log.reg$genero)
log.reg$NSE<-as.factor(log.reg$NSE)
log.reg$MotivoBaja<-as.factor(log.reg$MotivoBaja)
log.reg$Vpromedioacademico<-as.factor(log.reg$Vpromedioacademico)
log.reg$VDeportista<-as.factor(log.reg$VDeportista)
log.reg$VExcelenciaacademica<-as.factor(log.reg$VExcelenciaacademica)

#Splitting training data set and test data set
set.seed(88)
split = sample.split(log.reg$MotivoBaja,SplitRatio=.75)
log.train= subset(log.reg,split==T)
log.test= subset(log.reg,split==F)

###PROCESS FOR IDENTIFY PREDICTIVE VARIABLES

# EVALUATING FINAL MODELS
#Model 1
modelo.reg.log.df1 = glm(MotivoBaja ~   edad +  matematico + Becapromedio + 
                          PromedioPeriodoSeleccionado,data = log.train, family = binomial)
summary(modelo.reg.log.df1)

#Model 2
modelo.reg.log.df2 = glm(MotivoBaja ~edad +  matematico  + promedio_ingreso + 
                           AniosUniversidad + verbal + Reprobadasarea1 + PromedioPeriodoSeleccionado,data = log.train, family = binomial)
summary(modelo.reg.log.df2)

#Model 3
modelo.reg.log.df3 = glm(MotivoBaja ~   edad +  matematico  + Becapromedio  +
                           AniosUniversidad ,data = log.train, family = binomial)
summary(modelo.reg.log.df3)

# Comparing models
screenreg(list(modelo.reg.log.df1, modelo.reg.log.df2, modelo.reg.log.df3), caption="Comparación de modelos logit")

  #Comparing models AIC
  models <- list(modelo.reg.log.df1, modelo.reg.log.df2, modelo.reg.log.df3)
  model.names <- c('Modelo 1', 'Modelo 2', 'Modelo 3')
  # k parameters number, Model AICc score , Delta_AiCc: diference between the Model AIC and AIC , 
  aictab(cand.set = models, modnames = model.names)


#EVALUATING FINAL MODEL
modelo.reg.log.df = modelo.reg.log.df1


#Confidence Interval
confint(modelo.reg.log.df)
coefplot(modelo.reg.log.df)  + 
  theme_minimal() + 
  labs(title="Estimación de coeficientes con error estandar", 
       x="Estimación", 
       y="Variable", 
       caption="Migración a otros campus")

#Total effects: Table & graphs 
allEffects(modelo.reg.log.df)
plot(allEffects(modelo.reg.log.df))
#labs(title="Variables dependientes del modelo", 
 #        caption="Modelo predictivo de migración a otros campus")

#Predictions of values in training set
predict.train.log = predict(modelo.reg.log.df,newdata = log.train, type = "response")
tapply(predict.train.log, log.train$MotivoBaja,mean)

#Predictions of values in test set
predict.log = predict(modelo.reg.log.df, newdata = log.test, type = "response")

#confusion table 
confusion<-table(log.test$MotivoBaja,predict.log > .5)
confusion
# Confusion matrix
fourfoldplot(confusion, color = c("red", "green"),
             conf.level = 0, margin = 1, main = "Matriz de confusión")

n<-sum(confusion) # instances
nc<-sum(nrow(confusion)) # classes number
diag <- sum(diag(confusion)) # classes number correctly classified
rowsums <- (apply(confusion, 1, sum)) # instances number per class
colsums <- (apply(confusion, 2, sum)) # predicted number per class
p <- rowsums / n # instances per class
q <- colsums / n # instances per predicted class

#ROC 
tot<-colSums(confusion)#  number of instances w/ each test result
truepos<-unname(rev(cumsum(rev(confusion[2,2])))) # Number of true positives
falsepos<-unname(rev(cumsum(rev(confusion[1,2])))) # Number of false positives
falseneg<- unname(rev(cumsum(rev(confusion[2,1])))) # Number of false negatives
totneg<-sum(confusion[2,1]+confusion[2,2])  # The total number of negatives (one number)
trueneg<- unname(rev(cumsum(rev(confusion[1,1])))) # Number of true negatives
totpos<-sum(confusion[1,1]+confusion[2,1])  # The total number of positives (one number)


ROC(log.test$MotivoBaja,predict.log > .5,plot = "ROC")

#Accuracy
# (VP+VN)/(VP+FP+FN+VN)
exactitud <- sum(diag)/ n
exactitud
(truepos+trueneg)/n

#Precision
#(VP)/(VP+FP)
precision<- diag/colsums
precision
truepos/(truepos+falsepos)

#Sensibility
#VP/(VP+VN)
sensibilidad <- diag/ rowsums
sensibilidad
truepos/(truepos+trueneg)

#Specificity
#VN/(VN+FP)
especificidad<-trueneg/(trueneg+falsepos)
especificidad


#F-value
#(F-measure)
#F1 =2*(precision*recall)/(precision+recall)=tp/(tp+1/2(fp+fn) ),  siendo un modelo perfecto cuando F1=1.
#2*(precision*sensibility)/(precision+sensibility)
f1 <- 2*precision*sensibilidad / (precision+sensibilidad)
f1
f1<- truepos/(truepos+(1/(2*(falsepos+falseneg))))
f1


#Evaluation table
data.frame(precision,sensibilidad,f1)

# Mean
macroPrecision <- mean(precision)
macroExactitud <- mean(exactitud)
macroF1 <- mean(f1)
data.frame(macroPrecision, macroExactitud, macroF1)

#True positive rate/False positive rate

pred <- ifelse(modelo.reg.log.df$fitted.values < 0.5, 0, 1)
Accuracy(log.train,pred,2)

rocr.pred = prediction(predict.log, log.test$MotivoBaja)
rocr.perf = performance(rocr.pred, "tpr", "fpr")



#auc<-AUC(predict.log,log.test) GRAPH
plot(rocr.perf, print.cutoffs.at=seq(0,1,0.1), text.adj = c(-0.2,1.7))


plot(rocr.perf, avg = "threshold", colorize=TRUE, lwd= 3,main = "ROC AUC")
plot(rocr.perf, lty=3, col="grey78", add=TRUE)
