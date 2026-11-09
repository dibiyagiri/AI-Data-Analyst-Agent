
import re
import pandas as pd


def answer_question(df, question):
    """Answer common questions about an uploaded sales dataset."""

    q = question.lower().strip()
    q = re.sub(r"[?!.,]", "", q)

    if not q:
        return "Please enter a question."

    if df.empty:
        return "The dataset is empty."

    if "revenue" in q and any(
        word in q for word in ["total", "overall", "sum"]
    ):
        return f"Total revenue: ₹{df['revenue'].sum():,.2f}"

    if "average" in q or "mean" in q:
        if "price" in q:
            return f"Average price: ₹{df['price'].mean():,.2f}"

        if "revenue" in q:
            return (
                f"Average revenue per record: "
                f"₹{df['revenue'].mean():,.2f}"
            )

        if "quantity" in q:
            return (
                f"Average quantity: "
                f"{df['quantity'].mean():,.2f}"
            )

    if any(word in q for word in ["highest", "top", "best"]):
        if "product" in q or "sales" in q:
            totals = (
                df.groupby("product")["revenue"]
                .sum()
                .sort_values(ascending=False)
            )
            if not totals.empty:
                return (
                    f"Top product by revenue: {totals.index[0]} "
                    f"(₹{totals.iloc[0]:,.2f})"
                )

    if "category" in q and "revenue" in q:
        totals = (
            df.groupby("category")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        return "Revenue by category:\n\n" + "\n".join(
            f"- {category}: ₹{amount:,.2f}"
            for category, amount in totals.items()
        )

    if "how many" in q and any(
        word in q for word in ["order", "row", "record"]
    ):
        return f"Total records: {len(df)}"

    if "missing" in q or "null" in q:
        missing = int(df.isna().sum().sum())
        return f"Missing values: {missing}"

    return (
        "I couldn't understand that question yet. "
        "Try asking about total revenue, top products, "
        "average price, category revenue, record count, "
        "or missing values."
    )
