# Data

Ovaj folder je ostavljen zbog opcione lokalne verzije dataseta.

Notebook prvo proverava da li postoji:

```text
data/creditcard.csv
```

Ako fajl postoji, koristiće lokalni CSV.

Ako fajl ne postoji, notebook automatski preuzima dataset sa interneta, raspakuje ZIP i učitava CSV.

Zbog veličine dataseta, `creditcard.csv` nije dodat direktno u GitHub repozitorijum.
