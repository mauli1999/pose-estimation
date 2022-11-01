from sklearn.datasets import fetch_20newsgroups
train_raw = fetch_20newsgroups(subset='train')
test_raw = fetch_20newsgroups(subset='test')

x_train = train_raw.data
y_train = train_raw.target

x_test = test_raw.data
y_test = test_raw.target

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def build_lsa(x_train, x_test, dim=50):
    tfidf_vec = TfidfVectorizer(use_idf=True, norm='l2')
    svd = TruncatedSVD(n_components=dim)

    transformed_x_train = tfidf_vec.fit_transform(x_train)
    transformed_x_test = tfidf_vec.transform(x_test)

    print('TF-IDF output shape:', transformed_x_train.shape)

    x_train_svd = svd.fit_transform(transformed_x_train)
    x_test_svd = svd.transform(transformed_x_test)

    print('LSA output shape:', x_train_svd.shape)

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))

    return tfidf_vec, svd, x_train_svd, x_test_svd


tfidf_vec, svd, x_train_svd, x_test_svd = build_lsa(x_train, x_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

lr_model = LogisticRegression(solver='newton-cg', n_jobs=-1)
lr_model.fit(x_train_svd, y_train)

cv = KFold(n_splits=5, shuffle=True)

scores = cross_val_score(lr_model, x_test_svd, y_test, cv=cv, scoring='accuracy')
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

