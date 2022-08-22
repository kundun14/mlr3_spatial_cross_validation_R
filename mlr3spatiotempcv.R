# install.packages("mlr3")
# install.packages("mlr3verse") #  mlr3spatiotempcv
# install.packages("mlr3pipelines")
# install.packages("mlr3spatiotempcv")
# install.packages("blockCV")
# remotes::install_github("wilkelab/ggtext")
# install.packages("sperrorest")

# install.packages("pacman")
library(pacman)

# cargar paquetes
pacman::p_load(mlr3, 
               mlr3verse, 
               mlr3pipelines, #preprocessing operators to conveniently build ML pipelines
               mlr3spatiotempcv,
               blockCV, 
               ggplot2, #plots
               ggtext,   #plots  
               sperrorest,
               tidyverse,
               mlr3learners)



#objeto de clase Resampling
##define the data splits used for model assessment and selection (hyperparameter tuning) by ML algorithms

#mlr3spatial learners espaciales 

#lmr3 

# primero se crea un task/ data

  ## Spatial Task: Observations in the task have spatio-temporal information 
  ##(e.g. coordinates) ¡÷ mlr3spatiotempcv::TaskRegrST or 
  ## mlr3spatiotempcv::TaskClassifST in add-on package mlr3spatiotempcv.

# se crea un learner (algoritmo)
# train/test split
# prediccion
# seleccion de la medida de performance 
# estiamcion del performance (error de prediccion en el test data)
# resampling/repeticion

#METODOS

# Spatial leave-one-out

## Spatial leave-one-out with buffer - "spcv_buffer"
### spatial buffering method from package blockCV 

task = tsk("ecuador") # data
rsmp_buffer = rsmp("spcv_buffer", theRange = 1000) # spcv_buffer leave one out con buffer ( rango 1000 m)

autoplot(rsmp_buffer, size = 0.8, task = task, fold_id = 2)

## Leave-one-disc-out with optional buffer ¡X "spcv_disc"
# package sperrorest


#BLOQUES/ GRUPOS DE PUNTOS


# Leave-one-block-out cross-validation

# cada bloque es un fold


###Clustering-based: using coordinates ¡X "spcv_coords"

#### clusterizamiento en el espacio de la covariables o en el espacio geografico


#### k-means clustering of the coordinates

#### cada fold de evaluacion es un cluster espacial

rsmp_coords = rsmp("spcv_coords", folds = 5) #rsmp: crea el objeto, spcv_coords  

autoplot(rsmp_coords, fold_id = 5, task = task)

### Geometric: using rectangular blocks ¡X "spcv_tiles"
### Bloques rectagulares

rsmp_tiles = rsmp("spcv_tiles", nsplit = c(2L, 2L)) # 4 folds (hay zonas donde no hay puntos)

autoplot(rsmp_tiles, size = 1, fold_id = 2, task = task)


##Custom: "custom_cv" in mlr3
## bloques En base a un factor

### ejemplo : zonas altitudinales

breaks = quantile(x = task$data()$dem,
                  probs = seq(0, 1, length = 6)) 

# quantiles 0.0 0.2 0.4 0.6 0.8 1.0

# 0%                 20%     40%     60%     80%        100% 
# 1719.41 (minimo) 1917.21 2137.03 2314.44 2567.02 3113.36 (maximo)
# breaks : valores de corte
# seq(0, 1, length = 6) niveles 
#task$data() = dataframe
#as_tibble(task$data()) se puede convertir en tibble

# clases de elevacion 

zclass = cut(task$data()$dem,
             breaks = breaks, 
             include.lowest = TRUE) # incluir al valor mas bajo

#cut : numeric to factor -> crea intervalos como factores en funcion de los breaks

# ejemplo :
### la elevacion 1912. pertenece al nivel [1.72e+03,1.92e+03] 

rsmp_custom = rsmp("custom_cv") # creamos el objeto resampling  custom
#Instantiated: FALSE

rsmp_custom$instantiate(task, f = zclass) # incorporamos los niveles de elevacion como elemento de bloking
#Instantiated: TRUE
# cuantos folds se han creado? -> igual al numero de clases o niveles de elevacion = 5

autoplot(rsmp_custom, size = 0.8, task = task, fold_id = 5)




#Cross-validation at the block level


# observaciones -> bloques -> CV folds

###Geometric: using rectangular blocks ¡X "spcv_block"


## si existen cluster y el muestreo no es de cobertura
## "spcv_block" genera folds con observaciones irregulares

rsmp_block_random = rsmp("spcv_block", range = 1000, folds = 5)
#option selection = "random"  default
# tamano de los cuadrados es 1000*1000 m2
#el tamano de los bloques depende del rango, se puede estimar mendiante 
# spatialAutoRange() and rangeExplorer() to conduct a data-driven
# estimation of the distance at which the spatial autocorrelation within the data levels of

autoplot(rsmp_block_random, size = 1, fold_id = 1, task = task,
         show_blocks = TRUE, show_labels = TRUE)

# tres bloques contituyen un fold de evaluacion

#"cv" with grouping in mlr3
###s k-means clustering to generate classes that are used as blocks

# 8 bloques -> 3 folds 
# f1 = 3b y  f2=3b , f3=2b 

task_cv = tsk("ecuador")
group = as.factor(kmeans(task$coordinates(),centers= 8)$cluster) 
# k means en las coordenadas con 8 grupos

task_cv$cbind(data.frame("group" = group))
# anadir dataframe grupo ( clasificacion k means)

task_cv$set_col_roles("group", roles = "group")
#agrega el feature grupo al objeto resampling task_cv

rsmp_cv_group = rsmp("cv", folds = 3)$instantiate(task_cv)
#metodo cv normal de mlr3, instatiate agrupa las observaciones en 8 grupos/bloques
# que se reparten en 3 folds

print(rsmp_cv_group$instance)

autoplot(rsmp_cv_group, size = 1, task = task_cv, fold_id = 1)


#Clustering: using feature-based clustering ¡X "spcv_env"

## k means en el espacio de las features
## standarizacion previa



rsmp_env = rsmp("spcv_env", features = "distdeforest", folds = 5)
## variable para estratificar : distancia a bosque

rsmp_env_multi = rsmp("spcv_env", features = c("distdeforest", "slope"), folds = 5)
# variables estratificacion : distancia a bosque y pendiente

plot_env_single = autoplot(rsmp_env, size = 1, fold_id = 1, task = task) 
plot_env_multi = autoplot(rsmp_env_multi, size = 1, fold_id = 1, task = task)

library("patchwork")

plot_env_single + plot_env_multi # ambos graficos en el mismo panel


## SPATIAL VS NON SPATIAL CV
## data landslices
### binary classification -> Random forest
### area under the ROC curve (AUROC) as the performance measure
### Spatial CV -> leave-one-block-out CV using coordinate-based kmeans clustering
### 4 CV folds y 2 repeticiones



set.seed(42) # reproductibilidad

#Task preparation

# CREAR MANUALMENTE EL TASK EN BASE A UN DATAFRAME "ecuador"

#dataframe to task

backend = mlr3::as_data_backend(ecuador)  #dataframe - - - data.table::data.table() - - -> DataBackendDataTable


# TaskClassifST = Spatiotemporal Classification Task

task = TaskClassifST$new( id = "ecuador", #Method new(): crea un resampling class
                          backend = backend, 
                          target = "slides",
                          positive = "TRUE",
                          extra_args = list(
                          coordinate_names = c("x", "y"), 
                          coords_as_features = FALSE,
                          crs = "EPSG:32717")
                         )
#TRAIN/ TESTS DATA 
#Training and test sets are defined by the Resampling resampling.

#Model preparation

#learners

#random forest learner ("classif.ranger") paquete ranger
#predict_type = "probability" 
# variable de respuesta es la probalidad de 0 a 1 de ocurrencia de un landslice 


# library("mlr3learners")

learner = lrn("classif.ranger", predict_type = "prob")

# random forest de clasificacio, variable de respuesta 
# es la probalidad 0 a 1 de ocurrencia de un landslice

#Non-spatial cross-validation

# ESTATREGIA DE VALIDACION

rsmp_nsp = rsmp("repeated_cv", folds = 4, repeats = 2) # resampling object
#plot
autoplot(rsmp_nsp, size = 1, task = task, fold_id = 1, repeats_id = 2) # plot de la distribucion de observaciones por fold

#diferentes divisiones de la data producen deiferetnes estiamcines del error
# repeated_cv repite 2 veces la estimacion del error usando una division diferente en cada repeticion
# en cada repeticion se estiman 4 modelos
# en dos repeticiones son 8 modelos, se promedian los errores
# y finalmente se obtienen 4 estimaciones el error promedio de cada repeticion
# repeated_cv = Algorithm 1 en Krstajic et al. - 2014


# PREDICCIONES

rr_nsp = resample( task = task, 
                   learner = learner,
                   resampling = rsmp_nsp)

# Runs a resampling : Repeatedly apply 
# learner on a training set of  task to train a model, then use the 
# trained model to predict observations of a test set -> 8 modelos ()

#PLOT
# ResampleResults can be visualized via mlr3viz's autoplot() function.


#SELECCION DE LA METRICA DE PERFORMANCE
#AUROC ("classif.auc")

rr_nsp$aggregate(measures = msr("classif.auc"))

#classif.auc 
#0.7570533 


#SPATIAL CV  (coordinate-based clustering)

#### k-means clustering of the coordinates

#### cada fold de evaluacion es un cluster espacial

rsmp_sp = rsmp("repeated_spcv_coords",
               folds = 4, 
               repeats = 2)
#plot
autoplot(rsmp_sp, size = 1, task = task, fold_id = 1, repeats_id = 1) # plot de la distribucion de observaciones por fold

# PREDICCIONES
# la cv espcial genera predicciones diferentes a la cv no espacial
# por que los folds son genrados con el fin de reducir a ingluenica de cluster en la data

rr_sp = resample(task = task, 
                 learner = learner, 
                 resampling = rsmp_sp)

rr_sp$aggregate(measures = msr("classif.auc"))

# classif.auc 
# 0.6709433

# INterpetacion

# Non Sp AUC 0.76 > Sp AUC  0.67

#  spatial CV results better represent the 
# model¡¦s transferability to geologically
# and topographically similar areas adjacent to the training area.


# LA CV ESPACIAL CONSIERA LA CORRELACION ESPACIAL DE LAS OBSERVACIONES 

#RECOMENDACIONES 

# (1) mimics the predictive situation in which the model will be applied operationally, and
# (2) is consistent with the structure of the data.


# Resampling for hyperparameter tuning (molel selection and model assesment)

# supported by the mlr3 framework ----> que funciones??
# se recomienda usar  nested CV --> Algorithm 2 en Krstajic et al. - 2014
