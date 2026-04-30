import json
import numpy as np
from sklearn.decomposition import PCA


class SurrogateFeaturePipeline:
    """
    Feature pipeline for surrogate predictors.

    Modes:
    - ofa: original integer encoding from OFA search space
    - cole: architecture -> code string -> embedding (+ optional PCA)
    """

    def __init__(
        self,
        search_space,
        feature_repr="ofa",
        cole_model="sentence-transformers/all-MiniLM-L6-v2",
        cole_pca_components=None,
    ):
        self.search_space = search_space
        self.feature_repr = feature_repr
        self.cole_model = cole_model
        self.cole_pca_components = cole_pca_components

        self._embedder = None
        self._pca = None
        self._fitted = False

    @staticmethod
    def arch_to_code(arch):
        """Canonical text serialization for architecture configs."""
        return (
            "def architecture():\n"
            f"    ks = {json.dumps(arch['ks'])}\n"
            f"    e = {json.dumps(arch['e'])}\n"
            f"    d = {json.dumps(arch['d'])}\n"
            f"    r = {int(arch['r'])}\n"
            "    return {'ks': ks, 'e': e, 'd': d, 'r': r}\n"
        )

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "COLE mode requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._embedder = SentenceTransformer(self.cole_model)
        return self._embedder

    def _embed_archs(self, archs):
        texts = [self.arch_to_code(arch) for arch in archs]
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_archs(self, archs):
        return np.asarray([self.search_space.encode(arch) for arch in archs], dtype=np.float32)

    def fit_archs(self, archs):
        if self.feature_repr == "ofa":
            features = self._encode_archs(archs)
        elif self.feature_repr == "cole":
            features = self._embed_archs(archs)
        else:
            raise ValueError(f"Unknown feature_repr: {self.feature_repr}")

        if self.cole_pca_components is not None and self.cole_pca_components > 0:
            n_components = min(self.cole_pca_components, features.shape[1], features.shape[0])
            self._pca = PCA(n_components=n_components)
            features = self._pca.fit_transform(features)

        self._fitted = True
        return features

    def transform_archs(self, archs):
        if self.feature_repr == "ofa":
            features = self._encode_archs(archs)
        elif self.feature_repr == "cole":
            features = self._embed_archs(archs)
        else:
            raise ValueError(f"Unknown feature_repr: {self.feature_repr}")

        if self._pca is not None:
            features = self._pca.transform(features)
        return features

    def transform_encoded(self, encoded_x):
        archs = [self.search_space.decode(x) for x in encoded_x]
        return self.transform_archs(archs)
