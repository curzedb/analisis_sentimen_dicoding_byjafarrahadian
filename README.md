# ANALISIS SENTIMEN ULASAN APLIKASI PADA GOOGLE PLAY STORE
**Proyek Analisis Sentimen pada Ulasan Google Play Store Menggunakan Algoritma Naive Bayes, Random Forest, Logistic Regression, Decission Tree serta Perbandingannya antara TF-IDF dengan W2V**

📌 **Dibuat untuk**: Proyek 1 Kelas Machine Learning Intermediate(Menengah)-Dicoding X IDCamp

-------------------------------------------------------------------


## 📝 Deskripsi Proyek
Proyek ini membandingkan antara representasi **TF-IDF (Term Frequency-Inverse Document Frequency)** dengan **Word2Vec (W2V)** serta dengan menggunakan beberapa model seperti **Naive Bayes**, **Random Forest Classifier**, **Logistic Regression**, & **Decission Tree** untuk melakukan analisis sentimen berdasarkan beberapa parameter seperti: 
- reviewId
- userName
- userImage
- content
- score
- thumbsUpCount
- reviewCreatedVersion
- at
- replyContent
- repliedAt
- appVersion

Dataset yang digunakan: Review Aplikasi MiChat di Google Play Store, Scraping menggunakan API google_play_scraper (180000+ Data).

```python
# Mengambil semua ulasan dari aplikasi MiChat dengan ID 'com.michat.im' di Google Play Store.
scrapreview = reviews_all(
    'com.michatapp.im',      # ID aplikasi
    lang='id',               # Bahasa ulasan (menggunakan 'id' bahasa indonesia)
    country='id',            # Negara ('id' negara indonesia)
    sort=Sort.NEWEST,        # Urutan ulasan ('NEWEST' artinya terbaru)
    count=40000              # Jumlah maksimum ulasan yang ingin diambil
)
```
```python
#Dataset yang sudah diunduh selanjutnya akan disimpan dalam bentuk .CSV
with open('ulasan_michat.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Review'])
    for review in scrapreview:
        writer.writerow([review['content']])
```
-----------------------------------------------------------------------

## 🛠️ Tools
- **Bahasa Pemrograman**: Python 3
- **Libraries**: csv, datetime, gensim.models, google_play_scraper, io, matplotlib.pyplot, numpy, nltk, pandas, re, requests, seaborn, Sastrawi, string, sklearn, wordcloud
- **Platform**: Google Colab

-----------------------------------------------------------------------
## 📊 Struktur Dataset  
```
dataset/  
  ├── reviewId 
  ├── userName
  ├── userImage
  ├── content
  ├── score
  ├── thumbsUpCount
  ├── reviewCreatedVersion
  ├── at
  ├── replyContent
  ├── repliedAt
  └── appVersion
   
```

**Perlu diperhatikan bahwa data yang digunakan hanya komentar karena analisis sentimen hanya menyeleksi data berdasarkan komentar dan perkata**

### Polaritas Data:
| Jenis Data  | Jumlah Data | 
|-------------|-------------|
| Neutral     | 74476       | 
| Positive    | 63411       | 
| Negative    | 51113       |

![image](https://github.com/user-attachments/assets/dd2a3def-9e4d-417d-a241-847eaa4df103)
<br>


### Wordcloud: 
![image](https://github.com/user-attachments/assets/df897bb2-db05-4739-acae-be7d990f3672)
<br>


**Neutral**
<br>
![image](https://github.com/user-attachments/assets/6d685f2d-501c-44aa-85bd-609c0aeb0888)


**Positif**
<br>
![image](https://github.com/user-attachments/assets/1345bb40-1669-40d9-9601-02bc4c2b934c)


**Negative**
<br>
![image](https://github.com/user-attachments/assets/020164d2-11f4-4eab-b85a-5dcd18b4b83a)

### Kata Terbanyak:
![image](https://github.com/user-attachments/assets/72f24a9d-3681-4db6-b71a-e32d77161b28)
<br>

### Pembagian Data:
| Jenis Data  | Persentase Data |
|-------------|-----------------|
| Data Latih  | 65%             |
| Data Uji    | 35%             |

---------------------------------------------------------------------
## 🧠 Arsitektur Representasi Data

1. TF-IDF (Term Frequency-Inverse Document Frequency):
```python
# Ekstraksi fitur menggunakan TF-IDF
tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8 )
X_tfidf = tfidf.fit_transform(X)
```
```python
# Konversi hasil ekstraksi fitur menjadi dataframe
features_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
features_df
```

2. Word2Vec (W2V)
```python
# Ekstraksi Fitur untuk Word2vec
model_w2v = Word2Vec(sentences=clean_df['text_stopword'], vector_size=100, window=5, min_count=5, sg=1)
```
```python
def document_vector(doc):
       doc_vec = np.zeros((model_w2v.vector_size,), dtype="float32")
       num_words = 0
       for word in doc:
           if word in model_w2v.wv:
               doc_vec = np.add(doc_vec, model_w2v.wv[word])
               num_words += 1
       if num_words > 0:
           doc_vec = np.divide(doc_vec, num_words)
       return doc_vec
X_w2v = clean_df['text_stopword'].apply(lambda x: document_vector(x)).to_list()
X_w2v = np.array(X_w2v)
```
```python
# Contoh penggunaan dengan w2v:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# For training data:
X_train_w2v = clean_df.loc[y_train.index, 'text_stopword'].apply(lambda x: document_vector(x)).to_list()
X_train_w2v = np.array(X_train_w2v)

# For testing data:
X_test_w2v = clean_df.loc[y_test.index, 'text_stopword'].apply(lambda x: document_vector(x)).to_list()
X_test_w2v = np.array(X_test_w2v)

# Scale the Word2Vec features:
scaler = StandardScaler()
X_train_w2v_scaled = scaler.fit_transform(X_train_w2v)
X_test_w2v_scaled = scaler.transform(X_test_w2v)
```

-------------------------------------------------------------------------

## 🧠 Arsitektur Model

1. Naive Bayes
```python
naive_bayes = BernoulliNB()
naive_bayes.fit(X_train.toarray(), y_train)

y_pred_train_nb = naive_bayes.predict(X_train.toarray())
y_pred_test_nb = naive_bayes.predict(X_test.toarray())

accuracy_train_nb = accuracy_score(y_pred_train_nb, y_train)
accuracy_test_nb = accuracy_score(y_pred_test_nb, y_test)

print('Naive Bayes - accuracy_train:', accuracy_train_nb)
print('Naive Bayes - accuracy_test:', accuracy_test_nb)
```
```python
naive_bayes_w2v = BernoulliNB()
naive_bayes_w2v.fit(X_train_w2v, y_train)

y_pred_train_nb_w2v = naive_bayes_w2v.predict(X_train_w2v)
y_pred_test_nb_w2v = naive_bayes_w2v.predict(X_test_w2v)

accuracy_train_nb_w2v = accuracy_score(y_pred_train_nb_w2v, y_train)
accuracy_test_nb_w2v = accuracy_score(y_pred_test_nb_w2v, y_test)

print('Naive Bayes with Word2Vec - accuracy_train:', accuracy_train_nb_w2v)
print('Naive Bayes with Word2Vec - accuracy_test:', accuracy_test_nb_w2v)
```

2. Random Forest Classifier
```python
random_forest = RandomForestClassifier()
random_forest.fit(X_train.toarray(), y_train)

y_pred_train_rf = random_forest.predict(X_train.toarray())
y_pred_test_rf = random_forest.predict(X_test.toarray())

accuracy_train_rf = accuracy_score(y_pred_train_rf, y_train)
accuracy_test_rf = accuracy_score(y_pred_test_rf, y_test)

print('Random Forest - accuracy_train:', accuracy_train_rf)
print('Random Forest - accuracy_test:', accuracy_test_rf)
```
```python
random_forest_w2v = RandomForestClassifier()
random_forest_w2v.fit(X_train_w2v, y_train)

y_pred_train_rf_w2v = random_forest_w2v.predict(X_train_w2v)
y_pred_test_rf_w2v = random_forest_w2v.predict(X_test_w2v)

accuracy_train_rf_w2v = accuracy_score(y_pred_train_rf_w2v, y_train)
accuracy_test_rf_w2v = accuracy_score(y_pred_test_rf_w2v, y_test)

print('Random Forest dengan w2v - accuracy_train:', accuracy_train_rf_w2v)
print('Random Forest dengan w2v - accuracy_test:', accuracy_test_rf_w2v)
```

3. Logistic Regression
```python
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train.toarray(), y_train)

y_pred_train_lr = logistic_regression.predict(X_train.toarray())
y_pred_test_lr = logistic_regression.predict(X_test.toarray())

accuracy_train_lr = accuracy_score(y_pred_train_lr, y_train)

accuracy_test_lr = accuracy_score(y_pred_test_lr, y_test)

print('Logistic Regression - accuracy_train:', accuracy_train_lr)
print('Logistic Regression - accuracy_test:', accuracy_test_lr)
```
```python
logistic_regression_w2v = LogisticRegression()
logistic_regression_w2v.fit(X_train_w2v, y_train)

y_pred_train_lr_w2v = logistic_regression_w2v.predict(X_train_w2v)
y_pred_test_lr_w2v = logistic_regression_w2v.predict(X_test_w2v)

accuracy_train_lr_w2v = accuracy_score(y_pred_train_lr_w2v, y_train)

accuracy_test_lr_w2v = accuracy_score(y_pred_test_lr_w2v, y_test)

print('Logistic Regression dengan w2v - accuracy_train:', accuracy_train_lr_w2v)
print('Logistic Regression dengan w2v - accuracy_test:', accuracy_test_lr_w2v)
```

4. Decission Tree
```python
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train.toarray(), y_train)

y_pred_train_dt = decision_tree.predict(X_train.toarray())
y_pred_test_dt = decision_tree.predict(X_test.toarray())

accuracy_train_dt = accuracy_score(y_pred_train_dt, y_train)
accuracy_test_dt = accuracy_score(y_pred_test_dt, y_test)

print('Decision Tree - accuracy_train:', accuracy_train_dt)
print('Decision Tree - accuracy_test:', accuracy_test_dt)
```
```python
decision_tree_w2v = DecisionTreeClassifier()
decision_tree_w2v.fit(X_train_w2v, y_train)

y_pred_train_dt_w2v = decision_tree_w2v.predict(X_train_w2v)
y_pred_test_dt_w2v = decision_tree_w2v.predict(X_test_w2v)

accuracy_train_dt_w2v = accuracy_score(y_pred_train_dt_w2v, y_train)
accuracy_test_dt_w2v = accuracy_score(y_pred_test_dt_w2v, y_test)

print('Decision Tree - accuracy_train:', accuracy_train_dt_w2v)
print('Decision Tree - accuracy_test:', accuracy_test_dt_w2v)
```

-----------------------------------------------------------------------
## 📈 Hasil Evaluasi 

| Model | Accuracy Test | Accuracy Train |
|-------|---------------|----------------|
| Random Forest w2v | 0.913500 | 0.997452 |
| Random Forest TF-IDF | 0.910461 | 0.942499 |
| Logistic Regression TF-IDF | 0.909418 | 0.910322 |
| Decision Tree TF-IDF | 0.902373 | 0.942499 |
| Naive Bayes TF-IDF | 0.876024 | 0.876247 |
| Decision Tree w2v | 0.874558 | 0.997452 |
| Logistic Regression w2v | 0.844248 | 0.843875 |
| Naive Bayes w2v | 0.674452 | 0.674579 |

*Semakin Mendekati angka 1 maka semakin akurat model

-----------------------------------------------------------------------
## ☑️Implementasi
```python
# Input kalimat baru dari pengguna
kalimat_baru = input("Masukkan kalimat baru: ")

# Melakukan preprocessing pada kalimat baru
kalimat_baru_cleaned = cleaningText(kalimat_baru)
kalimat_baru_casefolded = casefoldingText(kalimat_baru_cleaned)
kalimat_baru_slangfixed = fix_slangwords(kalimat_baru_casefolded)
kalimat_baru_tokenized = tokenizingText(kalimat_baru_slangfixed)
kalimat_baru_filtered = filteringText(kalimat_baru_tokenized)
kalimat_baru_final = toSentence(kalimat_baru_filtered)

# Menggunakan objek tfidf yang sudah di-fit dari pelatihan sebelumnya
X_kalimat_baru = tfidf.transform([kalimat_baru_final])

# Memperoleh prediksi sentimen kalimat baru
prediksi_sentimen = logistic_regression.predict(X_kalimat_baru)

# Menampilkan hasil prediksi
if prediksi_sentimen[0] == 'positive':
    print("Sentimen kalimat baru adalah POSITIF.")
elif prediksi_sentimen[0] == 'negative':
    print("Sentimen kalimat baru adalah NEGATIVE.")
else:
    print("Sentimen kalimat baru adalah NETRAL.")
```
Hasil:
<br>
![image](https://github.com/user-attachments/assets/05fa29cd-d04e-457a-8c5a-9fe75aecf1db)
