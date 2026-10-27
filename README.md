# MarketPulse

Radar de sentiment de marche : des titres de presse financiere aux signaux
haussiers / baissiers par action, avec FinBERT.

Application de demonstration realisee dans le cadre du cours
*Advanced AI Algorithms* (AAA Group Assignment).

## Ce que fait l'application

- Saisie de plusieurs tickers (AAPL, MSFT, MC.PA, ^GSPC...).
- Recuperation de titres de presse recents, en cascade : yfinance / Yahoo Finance,
  puis flux RSS Yahoo Finance, puis Google News.
- Analyse de chaque titre par `ProsusAI/finbert` (Hugging Face).
- Score agrege par ticker entre -1 et +1, verdict, momentum, graphique et export CSV.
- Mode hors-ligne : import d'un CSV `date,ticker,headline`.
- Fiabilite du modele (`fiabilite_ia.py`) :
  - **Evaluation de FinBERT sur des donnees etiquetees par des humains** :
    exactitude (avec intervalle de confiance a 95 %), F1 macro, precision et
    rappel par classe, matrice de confusion, comparaison a une reference naive
    et a la regle de seuils de l'application, erreurs les plus confiantes.
    Trois jeux : titres financiers jamais vus par FinBERT (Twitter Financial
    News), Financial PhraseBank (corpus d'entrainement de FinBERT, donc score
    optimiste), ou un CSV annote par l'utilisateur.
  - **Explication mot par mot** de chaque prediction, par occlusion : on retire
    chaque mot un a un et on mesure la variation du score. Vert = le mot tire
    vers haussier, rouge = vers baissier.

## Donnees d'evaluation hors-ligne

Au premier lancement d'une evaluation, les phrases annotees sont telechargees
depuis Hugging Face puis copiees dans `donnees_eval/`. Ajoutez ce dossier au
depot : l'evaluation fonctionnera ensuite sans connexion (utile en soutenance).

## Lancer l'application

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Le navigateur s'ouvre sur <http://localhost:8501>. La premiere execution
telecharge le modele FinBERT (environ 440 Mo), mis en cache ensuite.

## Format du CSV

```
date,ticker,headline
2024-05-02,AAPL,Apple beats quarterly revenue expectations
2024-05-02,KO,Coca-Cola cuts full-year guidance
```

Le fichier `exemple_titres.csv` fourni permet de tester l'application sans
connexion.

## Limites

- Les sources d'actualites sont des services tiers : disponibilite, volume et
  couverture ne sont pas garantis. Yahoo Finance limite le debit des requetes.
- Le sentiment d'un article n'indique pas la reaction du cours. L'outil ne prend
  en compte ni les resultats financiers detailles, ni les anticipations de marche.
- Prototype pedagogique : ne constitue pas un conseil en investissement.

## Citations

- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained
  Language Models.* arXiv:1908.10063.
- Modele : <https://huggingface.co/ProsusAI/finbert> (Apache-2.0).
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., Takala, P. (2014). *Good debt
  or bad debt: Detecting semantic orientations in economic texts.* JASIST.
  Financial PhraseBank, licence CC BY-NC-SA 3.0.
- Twitter Financial News Sentiment :
  <https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment> (MIT).
- Bibliotheques : [Transformers](https://github.com/huggingface/transformers),
  [yfinance](https://github.com/ranaroussi/yfinance),
  [Streamlit](https://streamlit.io) (Apache-2.0).

## Structure du depot

```
app.py               interface Streamlit
geo_engine.py        contexte geopolitique detecte dans la presse mondiale
fiabilite_ia.py      evaluation et explication des predictions de FinBERT
design.py            identite visuelle : feuille de style et composants HTML
.streamlit/          theme Streamlit (couleurs de l'interface)
donnees_eval/        copies locales des jeux annotes (creees au 1er lancement)
requirements.txt     dependances
exemple_titres.csv   jeu de donnees de test hors-ligne
README.md            ce document
```
