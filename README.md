# Primena neuronskih mreža za detekciju prevara u finansijskim transakcijama

## 1. Uvod i opis problema

<p align="justify">
U savremenom digitalnom poslovanju, veliki broj finansijskih transakcija obavlja se putem kreditnih i debitnih kartica. Sa porastom broja elektronskih plaćanja raste i rizik od zloupotreba, odnosno prevara u transakcijama. Detekcija prevara predstavlja jedan od važnih problema u oblasti finansijske sigurnosti, jer je cilj sistema da na vreme prepozna sumnjivu transakciju i spreči potencijalnu štetu.
</p>

<p align="justify">
Problem detekcije prevara može se posmatrati kao problem binarne klasifikacije. Svaka transakcija se klasifikuje kao regularna transakcija ili kao prevara. U datasetu koji se koristi u ovom projektu ciljna promenljiva je <code>Class</code>, gde vrednost <code>0</code> označava regularnu transakciju, dok vrednost <code>1</code> označava prevarantsku transakciju.
</p>

<p align="justify">
Glavni izazov ovog problema je ekstreman disbalans klasa. U realnim sistemima, regularne transakcije čine ogromnu većinu ukupnog broja transakcija, dok se prevare javljaju veoma retko. Zbog toga obična metrika tačnosti, odnosno <b>accuracy</b>, nije dovoljna za procenu kvaliteta modela. Model može postići veoma visoku tačnost ako skoro sve transakcije označi kao regularne, ali takav model nema praktičnu vrednost jer ne uspeva da detektuje prevare.
</p>

<p align="justify">
Zbog toga se u ovom projektu fokus stavlja na metrike koje bolje opisuju ponašanje modela nad manjinskom klasom, kao što su <b>precision</b>, <b>recall</b>, <b>F1-score</b>, <b>ROC-AUC</b> i <b>PR-AUC</b>. Posebna pažnja posvećena je i izboru threshold-a, jer prag odlučivanja direktno utiče na odnos između propuštenih prevara i lažnih uzbuna.
</p>

<p align="justify">
U okviru projekta implementirano je više pristupa zasnovanih na neuronskim mrežama. Glavni pristup čine MLP modeli za nadgledanu binarnu klasifikaciju, dok je dodatno implementiran Autoencoder kao anomaly detection pristup.
</p>

---

## 2. Podaci: izvor, struktura, analiza i preprocesiranje

<p align="justify">
U projektu je korišćen poznati <b>Credit Card Fraud Detection</b> dataset. Dataset sadrži transakcije izvršene kreditnim karticama od strane evropskih korisnika u septembru 2013. godine.
</p>

<p align="justify">
Dataset sadrži ukupno <b>284.807 transakcija</b>, od čega je samo <b>492 transakcije</b> označeno kao prevara. To znači da prevare čine približno <b>0.17%</b> ukupnog skupa podataka, što jasno pokazuje ekstreman disbalans klasa.
</p>

<p align="justify">
Većina atributa u datasetu je anonimna zbog zaštite privatnosti korisnika. Kolone <code>V1</code> do <code>V28</code> predstavljaju rezultate PCA transformacije. PCA, odnosno Principal Component Analysis, koristi se za smanjenje dimenzionalnosti i transformaciju originalnih atributa u nove numeričke komponente. Zbog toga nije moguće direktno tumačiti poslovno značenje ovih kolona.
</p>

<p align="justify">
Jedine kolone koje nisu transformisane PCA metodom su <code>Time</code> i <code>Amount</code>. Kolona <code>Time</code> predstavlja broj sekundi proteklih od prve transakcije u datasetu, dok kolona <code>Amount</code> predstavlja iznos transakcije.
</p>

<div align="center">

| Naziv atributa | Tip podatka | Opis |
| :--- | :--- | :--- |
| **V1–V28** | Numerički | Anonimizovane PCA komponente |
| **Time** | Numerički | Sekunde protekle od prve transakcije |
| **Amount** | Numerički | Iznos transakcije |
| **Class** | Binarni | 0 = regularna transakcija, 1 = prevara |

</div>

### Učitavanje dataseta

<p align="justify">
Notebook koristi hybrid pristup za učitavanje podataka. Prvo se proverava da li postoji lokalni fajl:
</p>

```text
data/creditcard.csv
```

<p align="justify">
Ako fajl postoji, dataset se učitava lokalno. Ako fajl ne postoji, notebook automatski preuzima ZIP fajl sa interneta, raspakuje ga i učitava CSV. Na ovaj način projekat ostaje praktičan za pokretanje u Google Colab okruženju, a veliki CSV fajl ne mora da se čuva direktno u GitHub repozitorijumu.
</p>

### Preprocesiranje

<p align="justify">
Preprocesiranje podataka obuhvata sledeće korake:
</p>

1. učitavanje dataseta,
2. proveru dimenzija i strukture podataka,
3. proveru nedostajućih vrednosti,
4. analizu raspodele klasa,
5. podelu na trening, validacioni i test skup,
6. skaliranje numeričkih atributa,
7. računanje class weight vrednosti za balansiranje klasa.

<p align="justify">
Posebno je važno da se skaliranje podataka ne radi nad celim datasetom pre podele, jer bi to dovelo do curenja informacija, odnosno <i>data leakage</i> problema. Zbog toga se scaler fituje samo nad trening skupom, dok se validacioni i test skup samo transformišu istim scaler-om.
</p>

---

## 3. Arhitektura modela

### 3.1 MLP modeli

<p align="justify">
Glavni modeli u projektu zasnovani su na MLP arhitekturi. MLP, odnosno višeslojni perceptron, pogodan je za rad sa tabularnim numeričkim podacima, jer svaku transakciju posmatra kao vektor ulaznih atributa.
</p>

<p align="justify">
Osnovna ideja MLP mreže je da kroz više potpuno povezanih slojeva nauči nelinearne odnose između ulaznih atributa i ciljne promenljive. U ovom slučaju, cilj je da model nauči obrasce koji razlikuju regularne transakcije od prevarantskih.
</p>

Osnovna MLP arhitektura koristi:

- ulazni sloj sa numeričkim atributima transakcije,
- skrivene Dense slojeve,
- ReLU aktivacione funkcije,
- Batch Normalization,
- Dropout regularizaciju kod jačeg modela,
- izlazni sloj sa jednim neuronom,
- sigmoid aktivaciju za dobijanje verovatnoće prevare.

<p align="justify">
Sigmoid funkcija na izlazu vraća vrednost između 0 i 1. Ta vrednost se može posmatrati kao rizik da je transakcija prevara. Na osnovu izabranog threshold-a, transakcija se zatim klasifikuje kao regularna ili prevarantska.
</p>

### Implementirane MLP konfiguracije

| Model | Opis |
| :--- | :--- |
| **Basic MLP** | Osnovni MLP model bez class weight podešavanja |
| **Weighted MLP** | MLP model sa class weight balansiranjem |
| **Tuned Weighted MLP** | Jači weighted model sa više neurona i Dropout regularizacijom |

### 3.2 Autoencoder model

<p align="justify">
Pored MLP klasifikatora, u projektu je implementiran i Autoencoder. Autoencoder je neuronska mreža koja pokušava da rekonstruiše ulazne podatke. Sastoji se iz encoder dela, koji kompresuje ulazne podatke u manju reprezentaciju, i decoder dela, koji pokušava da rekonstruiše originalni ulaz.
</p>

<p align="justify">
Za problem detekcije prevara, Autoencoder se trenira samo nad regularnim transakcijama. Ideja je da model nauči kako izgledaju normalne transakcije. Kada se pojavi transakcija koja značajno odstupa od naučenog obrasca, rekonstrukciona greška će biti veća i ta transakcija se može označiti kao potencijalna anomalija.
</p>

Primer arhitekture Autoencoder modela:

```text
Input → 32 → 16 → 8 → 16 → 32 → Output
```

---

## 4. Trening

<p align="justify">
Trening MLP modela realizovan je pomoću TensorFlow/Keras biblioteke. Korišćen je Adam optimizator, dok je funkcija greške za binarnu klasifikaciju <code>binary_crossentropy</code>.
</p>

<p align="justify">
Kod MLP modela praćene su metrike relevantne za nebalansirane podatke, a posebno PR-AUC, jer Precision-Recall kriva bolje opisuje ponašanje modela kada je pozitivna klasa retka.
</p>

### Basic MLP model

<p align="justify">
Basic MLP model predstavlja osnovnu neuronsku mrežu bez dodatnog balansiranja klasa. Ovaj model služi za proveru kako se običan MLP ponaša nad ekstremno nebalansiranim podacima.
</p>

### Weighted MLP model

<p align="justify">
Weighted MLP model koristi <code>class_weight</code> parametar, kojim se većem značaju dodeljuje manjinska klasa, odnosno klasa prevare. Pošto su prevare veoma retke, bez balansiranja postoji rizik da model nauči da favorizuje regularne transakcije.
</p>

### Tuned Weighted MLP model

<p align="justify">
Tuned Weighted MLP model predstavlja jaču konfiguraciju weighted modela. U ovoj konfiguraciji koristi se veći broj neurona, Dropout regularizacija i podešeni hiperparametri. Cilj je da se proveri da li složenija arhitektura može da ostvari bolji balans između precision i recall vrednosti.
</p>

### Autoencoder

<p align="justify">
Autoencoder se trenira samo nad regularnim transakcijama. Kao funkcija greške koristi se rekonstrukciona greška, najčešće MSE. Nakon treninga, za svaku transakciju se računa reconstruction error. Ako je greška veća od odabranog threshold-a, transakcija se označava kao potencijalna prevara.
</p>

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

<p align="justify">
Hiperparametarska analiza u ovom projektu obuhvata poređenje više konfiguracija MLP modela. Posmatra se kako na performanse utiču:
</p>

- broj neurona u skrivenim slojevima,
- Dropout regularizacija,
- class weight balansiranje,
- learning rate,
- izbor threshold-a.

<p align="justify">
Poseban deo analize čini <b>threshold analiza</b>. Model ne vraća direktno klasu, već verovatnoću ili skor rizika. Threshold određuje granicu iznad koje se transakcija označava kao prevara. Promenom threshold-a može se povećati recall, ali često po cenu smanjenja precision vrednosti.
</p>

<p align="justify">
U kontekstu finansijskih prevara, threshold nije samo tehnički parametar, već i poslovna odluka. Niži threshold znači da će model uhvatiti više prevara, ali će verovatno proizvesti više lažnih uzbuna. Viši threshold smanjuje broj lažnih uzbuna, ali povećava rizik da stvarne prevare budu propuštene.
</p>

### Analiza osetljivosti

<p align="justify">
Analiza osetljivosti se koristi kako bi se procenilo koja obeležja najviše utiču na odluke modela. U projektu se koristi permutation importance pristup. Jedna po jedna kolona se nasumično izmeša, a zatim se meri koliko se pogoršava performansa modela.
</p>

<p align="justify">
Ako izmeštanje neke kolone značajno smanji PR-AUC ili drugu izabranu metriku, to znači da je ta kolona važna za model. Pošto su kolone <code>V1</code> do <code>V28</code> anonimizovane PCA transformacijom, njihova interpretacija je ograničena. Sa druge strane, kolona <code>Amount</code> je direktno interpretabilna i može imati jasno poslovno značenje.
</p>

---

## 6. Rezultati evaluacije

<p align="justify">
Modeli se evaluiraju nad test skupom koji nije korišćen tokom treninga. Time se proverava sposobnost modela da generalizuje na nove, neviđene podatke.
</p>

<p align="justify">
Zbog ekstremnog disbalansa klasa, accuracy nije dovoljna za procenu kvaliteta modela. Zato su korišćene sledeće metrike:
</p>

- **Accuracy** — ukupan procenat tačnih klasifikacija,
- **Precision** — koliko transakcija označenih kao prevara zaista jesu prevare,
- **Recall** — koliko stvarnih prevara je model uspeo da pronađe,
- **F1-score** — balans između precision i recall vrednosti,
- **ROC-AUC** — površina ispod ROC krive,
- **PR-AUC** — površina ispod Precision-Recall krive,
- **Confusion matrix** — prikaz TP, TN, FP i FN vrednosti,
- **Business cost** — procena poslovne cene grešaka modela.

<div align="center">

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline — sve regularno | 0.00 | 0.00 | 0.00 | 0.50 | 0.001738 |
| Basic MLP | 0.842105 | 0.808081 | 0.824742 | 0.908638 | 0.834066 |
| Weighted MLP | 0.692308 | 0.818182 | 0.75 | 0.948902 | 0.716121 |
| Tuned Weighted MLP | 0.833333 | 0.757576 | 0.793651 | 0.961360 | 0.740445 |
| Autoencoder | 0.523438 | 0.676768 | 0.590308 | 0.956284 | 0.601954 |

</div>

<p align="justify">
Konačna tabela rezultata se automatski generiše u notebook-u nakon izvršavanja svih modela. Najbolji model se ne bira isključivo na osnovu accuracy vrednosti, već na osnovu odnosa precision, recall, F1-score i PR-AUC metrike.
</p>

---

## 7. Diskusija

<p align="justify">
Dobijeni rezultati jasno pokazuju koliko je problem detekcije prevara specifičan zbog ekstremnog disbalansa klasa. Baseline model, koji sve transakcije proglašava regularnim, ostvaruje Precision, Recall i F1-score jednake 0.00 za klasu prevara. ROC-AUC vrednost od 0.50 pokazuje da ovakav model nema nikakvu sposobnost razlikovanja regularnih i prevarantskih transakcija. Ovaj rezultat potvrđuje da jednostavno proglašavanje svih transakcija regularnim nije upotrebljivo, bez obzira na to što bi accuracy kod ovakvog dataseta mogla da deluje visoko.
</p>

<p align="justify">
Basic MLP model pokazao se kao najbolji model prema najvažnijim metrikama za ovaj problem. Ostvario je Precision od 0.842105, Recall od 0.808081 i F1-score od 0.842742. Takođe ima najvišu PR-AUC vrednost od 0.834066, što je posebno važno kod nebalansiranih podataka, jer Precision-Recall kriva bolje opisuje ponašanje modela nad retkom pozitivnom klasom. Ovi rezultati pokazuju da Basic MLP postiže najbolji balans između detektovanja prevara i smanjenja broja lažnih uzbuna.
</p>

<p align="justify">
Weighted MLP model ostvaruje nešto veći Recall od Basic MLP modela, odnosno 0.818182, što znači da uspeva da pronađe nešto veći deo stvarnih prevara. Međutim, Precision pada na 0.692308, a F1-score na 0.75. To pokazuje da class weight balansiranje pomaže modelu da obrati više pažnje na manjinsku klasu, ali istovremeno povećava broj transakcija koje model pogrešno označava kao prevaru. Zbog toga Weighted MLP nije najbolji ukupni izbor, iako ima nešto bolji Recall.
</p>

<p align="justify">
Tuned Weighted MLP model ostvaruje najvišu ROC-AUC vrednost od 0.961360, što znači da model generalno dobro rangira transakcije prema riziku. Ipak, njegov Recall iznosi 0.757576, a F1-score 0.793651, što je slabije od Basic MLP modela. Ovo pokazuje da složenija arhitektura i dodatna regularizacija ne moraju nužno dovesti do boljeg konačnog modela. Kod nebalansiranih skupova podataka nije dovoljno posmatrati samo ROC-AUC, već je važno analizirati i Precision, Recall, F1-score i PR-AUC.
</p>

<p align="justify">
Autoencoder model daje slabije rezultate u odnosu na MLP modele za nadgledanu klasifikaciju. Ostvaruje Precision od 0.523438, Recall od 0.676768 i F1-score od 0.590308. Iako ima relativno visoku ROC-AUC vrednost od 0.956284, njegova PR-AUC vrednost od 0.601954 i niži F1-score pokazuju da je manje precizan u konkretnom označavanju prevarantskih transakcija. Ipak, Autoencoder je koristan kao dodatni anomaly detection pristup, jer problem posmatra iz drugačijeg ugla i ne zavisi direktno od velikog broja označenih primera prevare.
</p>

<p align="justify">
Na osnovu svih rezultata, Basic MLP se izdvaja kao najstabilniji i najupotrebljiviji model. Weighted modeli pokazuju da balansiranje klasa može povećati Recall, ali po cenu smanjenja Precision vrednosti. Autoencoder pokazuje da anomaly detection pristup ima potencijal, ali u ovom eksperimentu nije nadmašio nadgledane MLP modele. Najvažniji zaključak diskusije jeste da se najbolji model ne bira prema jednoj metrici, već prema ukupnom odnosu Precision, Recall, F1-score i PR-AUC vrednosti.
</p>

---

## 8. Zaključak

<p align="justify">
U ovom projektu implementirani su i upoređeni različiti pristupi za detekciju prevara u transakcijama kreditnim karticama pomoću neuronskih mreža. Problem je analiziran kao binarna klasifikacija pomoću MLP modela, ali i kao anomaly detection zadatak pomoću Autoencoder modela.
</p>

<p align="justify">
Eksperimenti su pokazali da accuracy nije dovoljna metrika kod ekstremno nebalansiranih skupova podataka. Baseline model, koji sve transakcije označava kao regularne, ima nultu vrednost za Precision, Recall i F1-score nad klasom prevara, čime se potvrđuje da model bez stvarne sposobnosti detekcije prevara nema praktičnu vrednost.
</p>

<p align="justify">
Najbolje ukupne rezultate ostvario je Basic MLP model. Sa Precision vrednošću od 0.842105, Recall vrednošću od 0.808081, F1-score vrednošću od 0.842742 i PR-AUC vrednošću od 0.834066, ovaj model ostvaruje najbolji balans između otkrivanja prevara i izbegavanja lažnih uzbuna. Zbog toga se Basic MLP može smatrati najpogodnijim modelom u okviru sprovedenih eksperimenata.
</p>

<p align="justify">
Weighted MLP model je pokazao da class weight može povećati Recall, ali uz smanjenje Precision vrednosti. Tuned Weighted MLP je ostvario najviši ROC-AUC, ali nije imao najbolji F1-score ni PR-AUC, što pokazuje da složeniji model nije nužno i bolji za praktičnu klasifikaciju. Autoencoder je dao koristan alternativni pogled na problem kroz detekciju anomalija, ali je po ključnim metrikama bio slabiji od MLP modela.
</p>

<p align="justify">
Konačan zaključak je da je za detekciju finansijskih prevara najvažnije pronaći dobar kompromis između hvatanja stvarnih prevara i smanjenja broja lažnih uzbuna. U ovom projektu taj kompromis najbolje ostvaruje Basic MLP model, dok ostali modeli pokazuju kako različite tehnike, poput class weight balansiranja, regularizacije i anomaly detection pristupa, utiču na ponašanje modela.
</p>

---

## Autor
- Jovan Teofilović 2022/0192
