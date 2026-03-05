import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType

def compute_features_fn(input_df, timestamp_column, start_date, end_date):
    """
    Computes features for the FAQs by generating synthetic FAQs and computing their vector embeddings.
    """
    from pyspark.sql import SparkSession
    import datetime
    
    spark = SparkSession.builder.getOrCreate()
    
    # Generate synthetic FAQ data
    faqs = [
        {"id": "faq_1", "question": "What is your return policy?", "answer": "You can return most items within 30 days of receipt."},
        {"id": "faq_2", "question": "How do I track my order?", "answer": "You can track your order in your account settings under 'Orders'."},
        {"id": "faq_3", "question": "Do you ship internationally?", "answer": "Yes, we ship to over 100 countries worldwide."},
        {"id": "faq_4", "question": "How can I contact customer support?", "answer": "You can reach us via email at support@example.com or call 1-800-123-4567."},
        {"id": "faq_5", "question": "What payment methods are accepted?", "answer": "We accept Visa, MasterCard, American Express, and PayPal."}
    ]
    # Pad to 50 FAQs
    for i in range(6, 51):
        faqs.append({"id": f"faq_{i}", "question": f"Sample FAQ question {i}", "answer": f"Sample FAQ answer {i}"})
        
    now = datetime.datetime.now()
    for faq in faqs:
        faq[timestamp_column] = now

    df = spark.createDataFrame(faqs)
    
    # Define a Pandas UDF to compute embeddings
    @pandas_udf(ArrayType(FloatType()))
    def compute_embeddings(texts: pd.Series) -> pd.Series:
        from sentence_transformers import SentenceTransformer
        # Load the pre-trained embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts.tolist(), convert_to_numpy=True)
        return pd.Series(embeddings.tolist())
        
    # We will embed the questions 
    features_df = df.withColumn("embedding", compute_embeddings(col("question")))
    
    return features_df
