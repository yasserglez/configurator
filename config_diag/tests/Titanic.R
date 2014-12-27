library("datasets")
library("epitools")

# Write Titanic.csv in expanded form.
Titanic.expanded <- expand.table(Titanic)
write.csv(Titanic.expanded, "Titanic.csv", row.names = FALSE)
