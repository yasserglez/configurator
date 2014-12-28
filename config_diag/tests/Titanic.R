library("datasets")
library("epitools")
library("arules")
library("prob")

# Write Titanic.csv in expanded form.
Titanic.expanded <- expand.table(Titanic)
write.csv(Titanic.expanded, "Titanic.csv", quote = FALSE, row.names = FALSE)

# test_assoc_rules.py
rules <- apriori(Titanic.expanded,
                 parameter = list(minlen = 2, target = "rules",
                                  supp = 0.5, conf = 0.95))
inspect(rules)

# test_freq_table.py
Titanic.data <- as.data.frame(Titanic)
space <- probspace(Titanic.data[ , 1:4], Titanic.data[ , 5])
Prob(space, Sex == "Male")
Prob(space, Sex == "Male" & Age == "Adult")
Prob(space, Survived == "Yes", Class == "1st")
Prob(space, Survived == "Yes", given = Age == "Adult")
Prob(space, Survived == "Yes", given = Age == "Adult" & Class == "1st")
