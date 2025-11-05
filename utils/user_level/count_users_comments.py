import pandas as pd


#comments = pd.read_csv("../../../processed/verysmall.tsv", sep="\t")
comments = pd.read_csv("../../../processed/small.tsv", sep="\t")

user_comment_counts = (
    comments.groupby("author")["num_comments"]
            .sum()
            .reset_index()
            .rename(columns={"num_comments": "total_comments"})
            .sort_values("total_comments", ascending=False)
)

print(user_comment_counts.head(10))

#user_comment_counts.to_csv("user_total_comments_verysmall.csv", index=False)
user_comment_counts.to_csv("user_total_comments_small.csv", index=False)


