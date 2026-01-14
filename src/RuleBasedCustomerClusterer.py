# src/cluster_library.py
# ============================================
# Mini Project: Customer Clustering from Association Rules
# Author: (Student)
# ============================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


class RuleBasedCustomerClusterer:
    """
    Convert association rules into feature vectors and perform customer clustering.
    """

    def __init__(self, transactions_df, rules_df):
        """
        Parameters
        ----------
        transactions_df : DataFrame
            Columns: [customer_id, invoice_id, item]
        rules_df : DataFrame
            Columns: [antecedents, consequents, support, confidence, lift]
        """
        self.transactions_df = transactions_df
        self.rules_df = rules_df

        self.customer_item_matrix = None
        self.rule_feature_matrix = None
        self.final_features = None
        self.cluster_labels = None
        self.pca_2d = None

    # -------------------------------------------------
    # 1. Build Customer × Item matrix
    # -------------------------------------------------
    def build_customer_item_matrix(self):
        """
        Build binary Customer × Item matrix
        """
        basket = (
            self.transactions_df
            .groupby(["customer_id", "item"])
            .size()
            .unstack(fill_value=0)
        )
        self.customer_item_matrix = (basket > 0).astype(int)
        return self.customer_item_matrix

    # -------------------------------------------------
    # 2. Build Rule-based Feature Matrix
    # -------------------------------------------------
    def build_rule_feature_matrix(self, top_k=20, weighted=False, weight_col="lift"):
        """
        Each column = one association rule
        Value = 1 if customer satisfies antecedent, else 0
        Optionally weighted by lift/confidence
        """
        if self.customer_item_matrix is None:
            self.build_customer_item_matrix()

        rules = self.rules_df.sort_values("lift", ascending=False).head(top_k)

        features = pd.DataFrame(index=self.customer_item_matrix.index)

        for idx, row in rules.iterrows():
            antecedents = list(row["antecedents"])
            rule_name = " & ".join(antecedents)

            satisfied = self.customer_item_matrix[antecedents].all(axis=1).astype(int)

            if weighted:
                satisfied = satisfied * row[weight_col]

            features[rule_name] = satisfied

        self.rule_feature_matrix = features
        return features

    # -------------------------------------------------
    # 3. Build Final Feature Vector (Rule + optional RFM)
    # -------------------------------------------------
    def build_final_features(self, rfm_df=None, scale=True):
        """
        Combine rule-features and RFM (optional)
        """
        X = self.rule_feature_matrix.copy()

        if rfm_df is not None:
            X = X.join(rfm_df, how="left")

        if scale:
            scaler = StandardScaler()
            X[:] = scaler.fit_transform(X)

        self.final_features = X
        return X

    # -------------------------------------------------
    # 4. Choose K by Silhouette
    # -------------------------------------------------
    def choose_k_by_silhouette(self, k_range=range(2, 11)):
        scores = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.final_features)
            score = silhouette_score(self.final_features, labels)
            scores[k] = score
        return scores

    # -------------------------------------------------
    # 5. Fit KMeans
    # -------------------------------------------------
    def fit_kmeans(self, k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.cluster_labels = km.fit_predict(self.final_features)
        return self.cluster_labels

    # -------------------------------------------------
    # 6. Project to 2D (PCA)
    # -------------------------------------------------
    def project_2d(self):
        pca = PCA(n_components=2, random_state=42)
        self.pca_2d = pca.fit_transform(self.final_features)
        return self.pca_2d

    # -------------------------------------------------
    # 7. Cluster Profiling
    # -------------------------------------------------
    def profile_clusters(self):
        profile = self.final_features.copy()
        profile["cluster"] = self.cluster_labels

        summary = profile.groupby("cluster").mean()
        summary["size"] = profile.groupby("cluster").size()

        return summary
