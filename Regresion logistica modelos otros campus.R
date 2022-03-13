#AUTOR: Lucía Beltrán Rocha
#email:beltranrochalucia@gmail.com
library(caTools)
library(ggplot2)
library(Epi)
library(rpart, lib.loc = "C:/Program Files/R/R-4.1.0/library")
#install.packages("factoextra")
library(factoextra)
library(rpart.plot)
library(caret)
library(ROCR)
#install.packages("rfVarImpOOB")
library("rfVarImpOOB")
library(texreg)
library(coefplot)
library(effects)
library(AICcmodavg)
library(cvAUC)

# Cargando set de datos
log.reg =read.csv ('DESERCION A OTROS CAMPUS.csv')

#Identificando los tipos de datos categoricos, nominales, discretos, continuos
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

#Distribuyendo los datos en un set de entrenamiento y un set de pruebas
set.seed(88)
split = sample.split(log.reg$MotivoBaja,SplitRatio=.75)
log.train= subset(log.reg,split==T)
log.test= subset(log.reg,split==F)

####IDENTIFICACIÓN DE LAS VARIABLES PREDICTORAS MAS SIGNIFICATIVAS DE ACUERDO A SU GRUPO

# EVALUACION DE LOS MODELOS FINALES CON LAS VARIABLES MAS SIGNIFICATIVAS DE LOS DIFERENTES MOMENTOS DEL DESERTOR

#Modelo final 1
modelo.reg.log.df1 = glm(MotivoBaja ~   edad +  matematico + Becapromedio + 
                          PromedioPeriodoSeleccionado,data = log.train, family = binomial)
summary(modelo.reg.log.df1)

#Modelo final 2
modelo.reg.log.df2 = glm(MotivoBaja ~edad +  matematico  + promedio_ingreso + 
                           AniosUniversidad + verbal + Reprobadasarea1 + PromedioPeriodoSeleccionado,data = log.train, family = binomial)
summary(modelo.reg.log.df2)

#Modelo final 3 
modelo.reg.log.df3 = glm(MotivoBaja ~   edad +  matematico  + Becapromedio  +
                           AniosUniversidad ,data = log.train, family = binomial)
summary(modelo.reg.log.df3)

# Comparación de los modelos
screenreg(list(modelo.reg.log.df1, modelo.reg.log.df2, modelo.reg.log.df3), caption="Comparación de modelos logit")

  #Comparacion del AIC de los modelos
  models <- list(modelo.reg.log.df1, modelo.reg.log.df2, modelo.reg.log.df3)
  model.names <- c('Modelo 1', 'Modelo 2', 'Modelo 3')
  # k numero de parametros, AICc score del modelo, Delta_AiCc diferencia entre el AIC y el AIC mejor modelo, 
  aictab(cand.set = models, modnames = model.names)


#EVALUACION DE DESEMPEÑO DEL MODELO FINAL
modelo.reg.log.df = modelo.reg.log.df1


#Intervalos de confianza
confint(modelo.reg.log.df)
coefplot(modelo.reg.log.df)  + 
  theme_minimal() + 
  labs(title="Estimación de coeficientes con error estandar", 
       x="Estimación", 
       y="Variable", 
       caption="Migración a otros campus")

#Tabla y gráfico de efectos totales promedio
allEffects(modelo.reg.log.df)
plot(allEffects(modelo.reg.log.df))
#labs(title="Variables dependientes del modelo", 
 #        caption="Modelo predictivo de migración a otros campus")

#Predicciones del valor esperado en el set de entrenamiento
predict.train.log = predict(modelo.reg.log.df,newdata = log.train, type = "response")
tapply(predict.train.log, log.train$MotivoBaja,mean)

#Predicciones del valor esperado en el set de prueba
predict.log = predict(modelo.reg.log.df, newdata = log.test, type = "response")

#Tabla de confusion de las predicciones
confusion<-table(log.test$MotivoBaja,predict.log > .5)
confusion

fourfoldplot(confusion, color = c("red", "green"),
             conf.level = 0, margin = 1, main = "Matriz de confusión")

#confusionMatrix((predict.log),(log.test))


n<-sum(confusion) # numero de instancias
nc<-sum(nrow(confusion)) # numero de clases
diag <- sum(diag(confusion)) # numero de instancias clasificadas correctamente
rowsums <- (apply(confusion, 1, sum)) #numero de instancias por clase
colsums <- (apply(confusion, 2, sum)) # numero de predicciones por clase
p <- rowsums / n # distribucion de instancias entre las clases
q <- colsums / n # distribucion de instancias entre las clases predichas

#ROC
tot<-colSums(confusion)# Number salidas w/ each test result
truepos<-unname(rev(cumsum(rev(confusion[1,1])))) # Number of true positives
falsepos<-unname(rev(cumsum(rev(confusion[1,2])))) # Number of false positives
falseneg<- unname(rev(cumsum(rev(confusion[2,1])))) # Number of false negativos
totneg<-sum(confusion[2,1]+confusion[2,2])  # The total number of negatives (one number)
trueneg<- unname(rev(cumsum(rev(confusion[2,2])))) # Number of true negativos
totpos<-sum(confusion[1,1]+confusion[1,2])  # The total number of positives (one number)
sens=truepos/totpos # Sensitivity (fraction true positives)
sens
omspec=falsepos/totneg # 1 ??? specificity (false positives)
sens=c(sens,0); omspec=c(omspec,0) # Numbers when we classify all as normal  
sens 
plot(omspec, sens, type="b", xlim=c(0,1), ylim=c(0,1), lwd=2,
     xlab="1 ??? Especificidad", ylab="Sensibilidad") # perhaps with xaxs="i"
grid()
abline(0,1, col="red", lty=2)

height = (sens[-1]+sens[-length(sens)])/2
width = -diff(omspec) # = diff(rev(omspec))
sum(height*width)


ROC(log.test$MotivoBaja,predict.log > .5,plot = "ROC")

#Accuracy
#Exactitud (acurracy) se refiere a cuán cerca del valor real se encuentra el valor medido.  
# (VP+VN)/(VP+FP+FN+VN)
exactitud <- sum(diag)/ n
exactitud
(truepos+trueneg)/n

#Precision
#precisión (precision) se refiere a la dispersión del conjunto de valores obtenidos
#de mediciones repetidas de una magnitud.  Cuanto menor es la dispersión mayor la precisión
#(VP)/(VP+FP)
precision<- diag/colsums
precision
truepos/(truepos+falsepos)

#Sensibilidad
#Sensibilidad (recall) se refiere a la respuesta que el instrumento de medición 
#tenga para medir una variable y que tan rápida sea este para estabilizar su medida.  
#VP/(VP+VN)
sensibilidad <- diag/ rowsums
sensibilidad
truepos/(truepos+trueneg)

#Especificidad
#También conocida como la Tasa de Verdaderos Negativos, ("true negative rate") o TN. Se 
#trata de los casos negativos que el algoritmo ha clasificado correctamente.  Expresa cuan bien puede el modelo detectar esa clase.
#Se calcula:  VN/(VN+FP)
especificidad<-trueneg/(trueneg+falsepos)
especificidad



#F-valor
#F-Valor (F-measure) medida de precisión que tiene una prueba.  
#F1 =2*(precision*recall)/(precision+recall)=tp/(tp+1/2(fp+fn) ),  siendo un modelo perfecto cuando F1=1.
#2*(precision*sensibilidad)/(precision+sensibilidad)
f1 <- 2*precision*sensibilidad / (precision+sensibilidad)
f1
f1<- truepos/(truepos+(1/(2*(falsepos+falseneg))))
f1

#Coeficiente phi asociación entre dos variables binarias, va de -1 a 1.  


#Tabla de evaluacion del modelo
data.frame(precision,sensibilidad,f1)

# PROMEDIO
macroPrecision <- mean(precision)
macroExactitud <- mean(exactitud)
macroF1 <- mean(f1)
data.frame(macroPrecision, macroExactitud, macroF1)


pred <- ifelse(modelo.reg.log.df$fitted.values < 0.5, 0, 1)
Accuracy(log.train,pred,2)

# OTROS GRÁFICOS

rocr.pred = prediction(predict.log, log.test$MotivoBaja)
rocr.perf = performance(rocr.pred, "tpr", "fpr")


height = (sensibilidad[-1]+sensibilidad[-length(sensibilidad)])/2
width = -diff(falsepos/totneg) # = diff(rev(falsepos/totneg))
height
width
auc<-sum(height*width)
auc
#auc<-AUC(predict.log,log.test)
plot(rocr.perf, print.cutoffs.at=seq(0,1,0.1), text.adj = c(-0.2,1.7))


plot(rocr.perf, avg = "threshold", colorize=TRUE, lwd= 3,main = "ROC AUC")
plot(rocr.perf, lty=3, col="grey78", add=TRUE)


rocr.perf <- performance(pred, "prec", "rec")
plot(rocr.perf, avg= "threshold", colorize=TRUE, lwd= 3,main= "... Precision/Recall graphs ...")
plot(rocr.perf, lty=3, col="grey78", add=TRUE)

rocr.perf <- performance(pred, "sens", "spec")
plot(rocr.perf, avg= "threshold", colorize=TRUE, lwd= 3, main="... Sensitivity/Specificity plots ...")
plot(rocr.perf, lty=3, col="grey78", add=TRUE)


rocr.perf <- performance(pred, "lift", "rpp")
plot(rocr.perf, avg= "threshold", colorize=TRUE, lwd= 3,main= "... and Lift charts.")
plot(perf, lty=3, col="grey78", add=TRUE)


plot(rocr.perf, colorize=TRUE, lwd=2,main='ROC curves from 10-fold cross-validation')

plot(rocr.perf, avg='vertical', spread.estimate='stderror',lwd=3,main='Vertical averaging + 1 standard error',col='blue')

plot(rocr.perf, avg='horizontal', spread.estimate='boxplot',lwd=3,main='Horizontal averaging + boxplots',col='blue')

plot(rocr.perf, avg='threshold', spread.estimate='stddev',lwd=2, main='Threshold averaging + 1 standard deviation',colorize=TRUE)


