#this file is the statistical analysis for the AI in Design JMD Special issue paper - submitted April 2021
#Sharon Ferguson
#looking for statistical differences by team strength and role in the number of topics in the topic model and the coherence of topics 
#robust anova using WRS2 package from : https://cran.r-project.org/web/packages/WRS2/index.html 


#RESOURCES:
#https://stats.stackexchange.com/questions/9477/how-to-draw-an-interaction-plot-with-confidence-intervals
#https://dornsife.usc.edu/assets/sites/239/docs/WRS2.pdf
#https://rcompanion.org/rcompanion/d_08a.html 
#https://rcompanion.org/handbook/G_09.html
#https://learningstatisticswithr.com/book/anova2.html#factorialanovaassumptions



#install and load packages
install.packages("readxl")
library("readxl")
library("dplyr")
install.packages("ggpubr")
library("ggpubr")
library("car")
if(!require(car)){install.packages("car")}
if(!require(psych)){install.packages("psych")}
if(!require(multcompView)){install.packages("multcompView")}
if(!require(lsmeans)){install.packages("lsmeans")}
if(!require(FSA)){install.packages("FSA")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(phia)){install.packages("phia")}
if(!require(rcompanion)){install.packages("rcompanion")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(WRS2)){install.packages("WRS2")}
if(!require(multcompView)){install.packages("multcompView")}
if(!require(psych)){install.packages("psych")}


#read in the data 
my_data <- read_excel("~/Documents/SlackData/TopicModels_April2021paper/topic_models_analysis.xlsx", sheet = "for R")

#first lets analyze topics per day

top_per_day <- my_data %>% select('Phase', 'Strength', 'Topics per day')# %>% filter(Phase != '4-FinalSelection') %>% mutate(top_day_log = (1/my_data$`Topics per day`))

#first we need to test assumptions

#test for homoscedasticity
leveneTest(top_per_day$`Topics per day`, interaction(top_per_day$Strength, top_per_day$Phase), center = median)
#does not meet assumption!

model = lm(`Topics per day` ~ Strength + Phase + Strength:Phase,
           data = top_per_day)

Anova(model,
      type = "III")

#test for normality of residuals
resid <- residuals( model ) 
shapiro.test( resid )

### MODEL: 
top_per_day <- top_per_day %>% group_by(Phase, Strength) # Group the data 
top_per_day$Phase <- factor(top_per_day$Phase) # convert to factor
top_per_day$Strength <- factor(top_per_day$Strength) # convert to factor

#post hoc test
t2way(`Topics per day` ~ Strength + Phase + Strength:Phase, data = top_per_day, tr = 0.2)
ef = mcp2atm(`Topics per day` ~ Strength + Phase + Strength:Phase, data = top_per_day, tr = 0.2)
ef$contrasts
ef

### Produce interaction plot

#phase interaction plot
top_per_day$Phase <- factor(top_per_day$Phase)
df <- with(top_per_day , aggregate(top_per_day$`Topics per day`, list(Phase = Phase), mean))
df$se <- with(top_per_day , aggregate(top_per_day$`Topics per day`, list(Phase = Phase), 
                                      function(x) sd(x)/sqrt(length(x))))[,2]
opar <- theme_update(panel.grid.major = theme_blank(),
                     panel.grid.minor = theme_blank())

gp <- ggplot(df, aes(Phase, x, group =1))+ geom_line() + ylab("Average Topics per Day")+
    geom_point() + geom_errorbar(aes(ymax=x+se, ymin=x-se), width=.1) + theme_bw() +theme(axis.text=element_text(size=16, angle = 90),
                                                                                          axis.title=element_text(size=16,face="bold", margin = margin(t = 50, r = 50, b = 50, l = 50)))
gp

#strength interaction plot
top_per_day$Strength <- factor(top_per_day$Strength)
df <- with(top_per_day , aggregate(top_per_day$`Topics per day`, list(Strength = Strength), mean))
df$se <- with(top_per_day , aggregate(top_per_day$`Topics per day`, list(Strength = Strength), 
                                      function(x) sd(x)/sqrt(length(x))))[,2]


gp <- ggplot(df, aes(Strength, x, group =1))+ geom_line() + ylab("Average Topics per Day")+
    geom_point() + geom_errorbar(aes(ymax=x+se, ymin=x-se), width=.1) + theme_bw() +theme(axis.text=element_text(size=16, angle = 90),
                                                                                          axis.title=element_text(size=16,face="bold", margin = margin(t = 50, r = 50, b = 50, l = 50)))
gp


#coherence UCI

UCI_df <- my_data %>% select('Phase', 'Strength', 'UCI') #%>% filter(Phase != '4-FinalSelection') %>% mutate(log_uci = 1/abs(UCI))#mutate(abs_uci = abs(UCI))#mutate(log_uci = log10(my_data$`UCI`))

#first we need to test for normality, homoscedasticity and multicollinearity

#test for homoscedasticity
leveneTest(UCI_df$UCI, interaction(UCI_df$Strength, UCI_df$Phase), center = median)
#meets assumption 

model = lm(UCI ~  Strength + Phase + Strength:Phase,
           data = UCI_df)

Anova(model,
      type = "III")

#test for normality of residuals
resid <- residuals( model ) 
hist(resid)
shapiro.test( resid )


UCI_df <- UCI_df %>% group_by(Phase, Strength) # Group the data 
UCI_df$Phase <- factor(UCI_df$Phase) # convert to factor
UCI_df$Strength <- factor(UCI_df$Strength) # convert to factor


t2way(UCI ~ Strength + Phase + Strength:Phase, data = UCI_df, tr = 0.2)
#post hoc tests
ef = mcp2atm(UCI ~ Strength + Phase + Strength:Phase, data = UCI_df, tr = 0.2)
ef$contrasts
ef

#UCI interaction plot - phase is the only main effect
UCI_df$Phase <- factor(UCI_df$Phase)
df <- with(UCI_df , aggregate(UCI_df$UCI, list(Phase = Phase), mean))
df$se <- with(UCI_df , aggregate(UCI_df$UCI, list(Phase = Phase), 
                                      function(x) sd(x)/sqrt(length(x))))[,2]


gp <- ggplot(df, aes(Phase, x, group =1))+ geom_line() + ylab("Average UCI Coherence\n")+
    geom_point() + geom_errorbar(aes(ymax=x+se, ymin=x-se), width=.1) + theme_bw() +theme(axis.text=element_text(size=16, angle = 90),
                                                                                            axis.title=element_text(size=16,face="bold", margin = margin(t = 50, r = 50, b = 50, l = 50)))
gp


#certainty coh - NOT USED IN FINAL PAPER ANALYSIS

coh_df <- my_data %>% select('Phase', 'Strength', 'coh')# %>% filter(Phase != '4-FinalSelection')

#first we need to test for normality, homoscedasticity and multicollinearity

#test for homoscedasticity
leveneTest(coh_df$'coh', interaction(coh_df$Strength, coh_df$Phase), center = median)
#meets assumption 

model = lm(coh_df$coh ~  coh_df$Strength + coh_df$Phase + coh_df$Strength:coh_df$Phase,
           data = coh_df)

Anova(model,
      type = "III")

#test for normality of residuals
resid <- residuals( model ) 
shapiro.test( resid )

coh_df <- coh_df %>% group_by(Phase, Strength) # Group the data 
coh_df$Phase <- factor(coh_df$Phase) # convert to factor
coh_df$Strength <- factor(coh_df$Strength) # convert to factor


t2way(coh ~ Strength + Phase + Strength:Phase, data = coh_df, tr = 0.2)
#post-hoc test
ef = mcp2atm(coh ~ Strength + Phase + Strength:Phase, data = coh_df, tr = 0.2)
ef$contrasts
ef


####STATISTICAL TEST FOR NUMBER OF TOPICS PER STRONG AND WEAK TEAMS###

my_data2 <- read.csv("~/Documents/SlackData/TopicModels_April2021paper/num_messages_per_team_phase.csv")

df <- my_data2 %>% select('phase', 'Strength', 'num_messages', 'num_messagesday') #%>% filter(Phase != '4-FinalSelection') %>% mutate(log_uci = 1/abs(UCI))#mutate(abs_uci = abs(UCI))#mutate(log_uci = log10(my_data$`UCI`))

#first we need to test for normality, homoscedasticity and multicollinearity

#test for homoscedasticity
leveneTest(df$num_messagesday, interaction(df$Strength, df$phase), center = median)
#does not meet assumption

model = lm(num_messagesday ~  Strength + phase + Strength:phase,
           data = df)

Anova(model,
      type = "III")

#test for normal residuals
resid <- residuals( model ) 
resid <- rstandard( model ) 
hist(resid)
shapiro.test( resid )

#set up the data to properly run the anova
df <- df %>% group_by(phase, Strength) # Group the data 
df$phase <- factor(df$phase) # convert to factor
df$Strength <- factor(df$Strength) # convert to factor


t2way(num_messagesday ~ Strength + phase + Strength:phase, data = df, tr = 0.2)
#post hoc test
ef = mcp2atm(num_messagesday ~ Strength + phase + Strength:phase, data = df, tr = 0.2)
ef$contrasts
ef


####THE CODE FOR A REGULAR ANOVA IF ASSUMPTIONS HAD BEEN MET#### - not used but good for reference
install.packages(multcomp)
library(multcomp)
pairwise.t.test(top_per_day$`Topics per day`, top_per_day$Phase, p.adjust.method ="bonferroni")


library(rcompanion)

Sum = groupwiseHuber(data = top_per_day,
                     group = c("Strength", "Phase"),
                     var = "Topics per day",
                     conf.level=0.95,
                     conf.type="wald")

library(ggplot2)

pd = position_dodge(.2)

ggplot(Sum, aes(x=Phase,
                y=M.Huber,
                color=Strength)) +
    geom_errorbar(aes(ymin=lower.ci,
                      ymax=upper.ci),
                  width=.2, size=0.7, position=pd) +
    geom_point(shape=15, size=4, position=pd) +
    theme_bw() +
    theme(
        axis.title.y = element_text(vjust= 1.8),
        axis.title.x = element_text(vjust= -0.5),
        axis.title = element_text(face = "bold")) +
    scale_color_manual(values = c("black", "blue"))

library(phia)

IM = interactionMeans(model)

IM
