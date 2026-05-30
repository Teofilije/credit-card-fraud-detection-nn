# Credit Card Fraud Detection using MLP and Autoencoder models in TensorFlow/Keras

> Academic project for Neural Networks and Deep Learning course.  
> Topic: **Credit Card Fraud Detection using Neural Networks**

## 1. Opis problema

Cilj projekta je razvoj neuronskih mreža koje mogu da prepoznaju potencijalne prevare u transakcijama kreditnim karticama.

Problem je postavljen kao binarna klasifikacija:

- `0` — regularna transakcija,
- `1` — prevarantska transakcija.

Ovaj problem je posebno zanimljiv zato što je dataset ekstremno nebalansiran. Broj regularnih transakcija je mnogo veći od broja prevara, pa obična accuracy metrika nije dovoljna za procenu kvaliteta modela.

Na primer, model koji svaku transakciju označi kao regularnu može imati veoma visoku accuracy, ali takav model nema praktičnu vrednost jer ne otkriva nijednu prevaru.

Zbog toga se u projektu koriste metrike kao što su precision, recall, F1-score, PR-AUC, ROC-AUC i confusion matrix.

## 2. Podaci: izvor, struktura, analiza i preprocesiranje

Korišćen je poznati **Credit Card Fraud Detection** dataset, originalno objavljen na Kaggle-u od strane ULB Machine Learning Group.

Dataset sadrži transakcije evropskih korisnika kartica iz septembra 2013. godine.

Struktura podataka:

| Kolona | Opis |
|---|---|
| `Time` | Broj sekundi od prve transakcije u datasetu |
| `Amount` | Iznos transakcije |
| `V1`–`V28` | Anonimizovane PCA komponente |
| `Class` | Ciljna promenljiva: 0 regularna transakcija, 1 prevara |

Preprocesiranje obuhvata:

1. učitavanje dataseta,
2. proveru nedostajućih vrednosti,
3. analizu raspodele klasa,
4. stratifikovanu podelu na train, validation i test skup,
5. skaliranje podataka pomoću `RobustScaler`,
6. sprečavanje data leakage problema tako što se scaler fituje samo nad trening skupom.

Dataset se u notebook-u automatski preuzima, tako da nije neophodno ručno dodavati `creditcard.csv`.

## 3. Arhitektura modela

U projektu su implementirana dva pristupa.

### 3.1 Supervised MLP model

MLP model je klasična feed-forward neuronska mreža pogodna za tabularne numeričke podatke.

Model koristi:

- ulazni sloj sa atributima transakcije,
- `Dense` slojeve,
- `ReLU` aktivaciju,
- `BatchNormalization`,
- `Dropout`,
- sigmoid izlazni sloj.

Sigmoid izlaz vraća vrednost između 0 i 1, odnosno procenu rizika da je transakcija prevara.

Testirane su različite MLP konfiguracije:

1. osnovni MLP bez `class_weight`,
2. MLP sa `class_weight`,
3. jači MLP sa više neurona, većim dropout-om i drugačijim learning rate-om.

### 3.2 Autoencoder model

Autoencoder je dodatni anomaly detection pristup.

Za razliku od MLP klasifikatora, Autoencoder se trenira samo nad regularnim transakcijama. Njegov zadatak je da nauči kako izgledaju normalne transakcije.

Ako model neku transakciju ne može dobro da rekonstruiše, rekonstrukciona greška je visoka i transakcija se označava kao potencijalna anomalija, odnosno moguća prevara.

Arhitektura Autoencodera:

```text
input -> 32 -> 16 -> 8 -> 16 -> 32 -> output
```

## 4. Trening

Za MLP modele korišćeni su:

- `Adam` optimizator,
- `binary_crossentropy` loss,
- `EarlyStopping`,
- validation PR-AUC za praćenje treninga,
- `class_weight` za rešavanje disbalansa klasa.

Za Autoencoder:

- koristi se `mse` reconstruction loss,
- trenira se samo na regularnim transakcijama,
- threshold se bira na validation skupu.

## 5. Analiza osetljivosti i hiperparametarska optimizacija

Hiperparametarska analiza obuhvata poređenje više MLP konfiguracija:

- različit broj neurona,
- različit dropout,
- različit learning rate,
- modeli sa i bez `class_weight`.

Pored toga, radi se i analiza threshold-a. Model vraća skor rizika, ali threshold određuje od koje vrednosti se transakcija proglašava prevarom.

Analiza osetljivosti je urađena pomoću permutation importance metode. Jedna po jedna kolona se nasumično izmeša i meri se pad PR-AUC metrike. Ako metrika značajno opadne, ta kolona je važna za model.

## 6. Rezultati evaluacije

Modeli se porede pomoću sledećih metrika:

- accuracy,
- precision,
- recall,
- F1-score,
- ROC-AUC,
- PR-AUC,
- confusion matrix,
- business cost.

Najvažnije metrike za ovaj problem su precision, recall, F1-score i PR-AUC, jer dataset ima jako mali broj prevara u odnosu na regularne transakcije.

Confusion matrix omogućava da se jasno vidi broj:

- true negative,
- false positive,
- false negative,
- true positive predikcija.

## 7. Diskusija

Kod detekcije prevara postoji realan poslovni kompromis.

False Negative znači da je stvarna prevara propuštena. To je opasno jer direktno pravi finansijsku štetu.

False Positive znači da je regularna transakcija pogrešno označena kao prevara. To može iznervirati korisnika ili blokirati legitimnu kupovinu.

Zato threshold nije samo tehnička odluka, već i poslovna odluka.

MLP model je glavni model u projektu, dok je Autoencoder dodat kao alternativni anomaly detection pristup.

## 8. Zaključak

Projekat pokazuje da je detekcija prevara sa kreditnim karticama tipičan primer problema gde accuracy nije dovoljna metrika.

Zbog ekstremnog disbalansa klasa, potrebno je koristiti metrike koje bolje ocenjuju ponašanje modela nad retkom fraud klasom.

MLP model sa odgovarajućim threshold-om može dati dobar balans između precision i recall vrednosti. Autoencoder daje dodatnu perspektivu jer posmatra prevare kao anomalije u odnosu na normalne transakcije.

Najvažniji zaključak je da najbolji model nije nužno onaj sa najvećom accuracy vrednošću, već onaj koji pravi najbolji kompromis između propuštenih prevara i lažnih uzbuna.

## Kako pokrenuti projekat u Google Colab-u

1. Otvoriti Google Colab.
2. Uploadovati notebook iz foldera `notebooks/`.
3. Preporučeni notebook je:

```text
notebooks/Credit_Card_Fraud_Detection_FIXED_V2.ipynb
```

4. Pokrenuti ćelije redom.
5. Dataset se automatski preuzima u notebook-u.
6. Rezultati evaluacije se generišu na kraju notebook-a.

## Struktura repozitorijuma

```text
credit-card-fraud-keras-autoencoder-project/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── defense_notes.md
├── walkthrough_comments.md
│
├── data/
│   └── README.md
│
└── notebooks/
    └── Credit_Card_Fraud_Detection_FIXED_V2.ipynb
```

## Reference

- Kaggle / ULB Credit Card Fraud Detection dataset
- TensorFlow/Keras documentation
- Scikit-learn documentation
