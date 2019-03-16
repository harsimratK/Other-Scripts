0

View(DecadeInfo)
order(DecadeInfo$Total)
mydata <- read.csv("C:/Users/savi/Downloads/gtd/DecadeInfo.csv", sep=",") 
#View(mydata)
mydata
mydata<- mydata[order(mydata$Total),]
#View(mydata)
row.names(mydata) <- mydata$Row.Labels
#View(mydata)
mydata<- mydata[103:153,2:12]
#View(mydata)
mydata_matrix <- data.matrix(mydata)
#View(mydata_matrix)
mydata_heatmap <- heatmap(mydata_matrix, Rowv=NA, Colv=NA, col = heat.colors(256),scale = c("row", "column", "none"), margins=c(5,10))


heatmap(mydata_matrix, Rowv = NULL, Colv = NULL,
       distfun = dist,        scale = c("row", "column", "none"), na.rm = TRUE,
       margins = c(5, 5), 
       cexRow = 0.2 + 1/log10(nr), cexCol = 0.2 + 1/log10(nc),
       labRow = NULL, labCol = NULL, main = NULL,
       xlab = "Countries", ylab = NULL,
       )
