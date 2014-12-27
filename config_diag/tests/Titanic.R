library("datasets")
library("epitools")
library("arules")

# Write Titanic.csv in expanded form.
#Titanic.expanded <- expand.table(Titanic)
#write.csv(Titanic.expanded, "Titanic.csv", quote = FALSE, row.names = FALSE)

# test_assoc_rules.py
rules <- apriori(Titanic.expanded,
                 parameter = list(minlen = 2, target = "rules",
                                  supp = 0.5, conf = 0.95))
inspect(rules)
