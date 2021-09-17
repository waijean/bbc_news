from sklearn.feature_extraction.text import TfidfVectorizer


class CustomVectorizer(TfidfVectorizer):
    # overwrite the build_analyzer method to create a custom analyzer for the vectorizer
    def build_analyzer(self):

        # create the custom analyzer that will be returned by this method
        def analyzer(text):
            """
            Handles n-grams generation.
            https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes

            Args:
                text: A string of text

            Returns: A list of tokens

            """
            # If documents are pre-tokenized by an external package, then store them in files (or strings) with
            # the tokens separated by whitespace and pass analyzer=str.split
            tokens = str.split(text)

            # lowercase the token
            tokens = [token.lower() for token in tokens]

            # use TfidfVectorizer's _word_ngrams built in method to extract ngrams
            return self._word_ngrams(tokens)

        return analyzer
