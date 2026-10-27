#!/usr/bin/env python3
"""
fiabilite_ia — Ouvrir la boite noire FinBERT
=============================================
Deux outils pour juger le modele de sentiment utilise par MarketPulse.

1. Explication mot par mot (methode par occlusion)
   On retire chaque mot du titre, un a la fois, et on mesure de combien le score
   FinBERT (P(positif) - P(negatif)) change. Un mot dont le retrait fait chuter le
   score tirait le titre vers le haut ; l'inverse pour un mot baissier. La methode
   ne depend pas de l'architecture du modele et se lit sans connaissance technique.
   Limite : les mots sont retires un par un, les interactions entre mots
   ("not good", "less bad than feared") ne sont captees qu'en partie.

2. Evaluation sur des donnees etiquetees par des humains
   - Twitter Financial News Sentiment (zeroshot, licence MIT) : titres et tweets
     financiers annotes baissier / haussier / neutre. FinBERT ne les a jamais vus :
     c'est le test le plus proche de l'usage reel de l'application.
   - Financial PhraseBank (Malo et al., 2014, CC BY-NC-SA 3.0) : 2 264 phrases sur
     lesquelles tous les annotateurs sont d'accord. FinBERT a ete entraine sur ce
     corpus (Araci, 2019) : le score obtenu est optimiste, a lire comme un plafond.
   - Un fichier annote par l'utilisateur (CSV texte,label).
   Metriques : exactitude avec intervalle de confiance a 95 % (Wilson), F1 macro,
   precision / rappel par classe, matrice de confusion, comparaison a une
   reference naive (toujours predire la classe majoritaire) et a la regle de
   seuils utilisee par l'application.

Le module n'utilise que numpy, pandas et la bibliotheque standard. Le modele est
passe en parametre, avec la meme interface que le pipeline transformers de
app.py : clf(liste_de_textes, truncation=True, batch_size=16) renvoie, pour
chaque texte, une liste de {"label": ..., "score": ...}. On peut donc tester ce
module avec un faux modele, sans telecharger FinBERT.
"""

from __future__ import annotations

import datetime as dt
import html
import io
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
API_HF = "https://datasets-server.huggingface.co"
DOSSIER_CACHE = Path(__file__).resolve().parent / "donnees_eval"
GRAINE = 42

CLASSES = ("negative", "neutral", "positive")
NOMS_FR = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}

# Toutes les facons courantes d'ecrire une etiquette, en anglais et en francais.
_ALIAS = {
    "negative": "negative", "neg": "negative", "negatif": "negative",
    "bearish": "negative", "baissier": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "neutre": "neutral", "0": "neutral",
    "positive": "positive", "pos": "positive", "positif": "positive",
    "bullish": "positive", "haussier": "positive", "1": "positive", "+1": "positive",
}

JEUX = {
    "titres": {
        "nom": "Titres financiers (Twitter Financial News) : jamais vus par FinBERT",
        "court": "Twitter Financial News",
        "dataset": "zeroshot/twitter-financial-news-sentiment",
        "configs": ["default"],
        "splits": ["validation", "valid", "test", "train"],
        # Codes du jeu : 0 = Bearish, 1 = Bullish, 2 = Neutral
        "codes": {0: "negative", 1: "positive", 2: "neutral"},
        "csv_secours": (
            "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment"
            "/resolve/main/sent_valid.csv"
        ),
        "lecture": (
            "Le test le plus honnête : des titres et tweets financiers que FinBERT n'a "
            "jamais vus, proches de ce que l'application analyse vraiment. Les "
            "annotateurs ont jugé l'effet probable sur le marché (haussier, baissier, "
            "neutre), pas seulement le ton de la phrase."
        ),
        "avertissement": None,
        "citation": "zeroshot/twitter-financial-news-sentiment (Hugging Face, licence MIT)",
    },
    "phrasebank": {
        "nom": "Financial PhraseBank : corpus d'entraînement de FinBERT (score optimiste)",
        "court": "Financial PhraseBank",
        "dataset": "takala/financial_phrasebank",
        "configs": ["sentences_allagree"],
        "splits": ["train"],
        "codes": {0: "negative", 1: "neutral", 2: "positive"},
        "csv_secours": None,
        "lecture": (
            "Phrases de presse financière sur lesquelles tous les annotateurs "
            "(professionnels de la finance) sont d'accord."
        ),
        "avertissement": (
            "FinBERT a été entraîné sur ce corpus. Le résultat mesure surtout ce que "
            "le modèle a retenu : c'est un plafond, pas sa performance réelle. "
            "Comparez-le au jeu « Titres financiers » : l'écart entre les deux montre "
            "ce que le modèle perd face à des textes nouveaux."
        ),
        "citation": (
            "Malo, P. et al. (2014). Good debt or bad debt: Detecting semantic "
            "orientations in economic texts. JASIST. Licence CC BY-NC-SA 3.0"
        ),
    },
}


# ---------------------------------------------------------------------------
# 1. Etiquettes et chargement des donnees
# ---------------------------------------------------------------------------
def normaliser_label(valeur) -> str | None:
    """Ramene une etiquette ecrite librement a negative / neutral / positive."""
    if valeur is None or (isinstance(valeur, float) and math.isnan(valeur)):
        return None
    texte = str(valeur).strip().lower()
    texte = "".join(
        c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn"
    )
    if re.fullmatch(r"[+-]?\d+\.0+", texte):
        texte = texte.split(".")[0]
    return _ALIAS.get(texte)


def _http_get(url: str, timeout: int = 20) -> bytes:
    requete = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(requete, timeout=timeout) as reponse:
        return reponse.read()


def _http_json(url: str, timeout: int = 20, essais: int = 3) -> dict:
    for essai in range(essais):
        try:
            return json.loads(_http_get(url, timeout).decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 429 and essai < essais - 1:  # debit limite : on patiente
                time.sleep(2 * (essai + 1))
                continue
            raise
    raise RuntimeError("Service de données indisponible")


def _choisir(options: list[str], preferes: list[str]) -> str:
    for p in preferes:
        if p in options:
            return p
    for p in preferes:
        for o in options:
            if p in o:
                return o
    return options[0]


def _colonne(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    bas = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in bas:
            return bas[c]
    return None


def vers_jeu(df: pd.DataFrame, features: list | None = None, codes: dict | None = None) -> pd.DataFrame:
    """Transforme un tableau brut en deux colonnes : texte, verite."""
    col_texte = _colonne(df, ("texte", "text", "sentence", "headline", "titre", "phrase"))
    col_label = _colonne(df, ("label", "labels", "sentiment", "etiquette", "étiquette", "verite"))
    if col_texte is None or col_label is None:
        raise ValueError(
            "Colonnes introuvables : il faut une colonne de texte (texte, text, "
            "sentence ou headline) et une colonne d'étiquette (label ou sentiment). "
            f"Colonnes lues : {', '.join(map(str, df.columns))}"
        )

    noms = None
    for f in features or []:
        type_f = f.get("type", {}) if isinstance(f, dict) else {}
        if f.get("name") == col_label and type_f.get("_type") == "ClassLabel":
            noms = type_f.get("names")

    def convertir(v):
        if noms is not None:
            try:
                return normaliser_label(noms[int(v)])
            except (ValueError, IndexError, TypeError):
                return normaliser_label(v)
        if codes:
            try:
                if float(v) == int(float(v)) and int(float(v)) in codes:
                    return codes[int(float(v))]
            except (ValueError, TypeError):
                pass
        return normaliser_label(v)

    sortie = pd.DataFrame(
        {
            "texte": df[col_texte].astype(str).str.strip(),
            "verite": df[col_label].map(convertir),
        }
    )
    sortie = sortie[(sortie["texte"] != "") & (sortie["texte"] != "nan")]
    sortie = sortie.dropna(subset=["verite"]).drop_duplicates("texte")
    return sortie.reset_index(drop=True)


def _via_api_hf(meta: dict, max_lignes: int = 5000) -> pd.DataFrame:
    q = lambda s: urllib.parse.quote(s, safe="")  # noqa: E731
    splits = _http_json(f"{API_HF}/splits?dataset={q(meta['dataset'])}").get("splits", [])
    if not splits:
        raise RuntimeError("aucun sous-ensemble publié")
    config = _choisir(list(dict.fromkeys(s["config"] for s in splits)), meta["configs"])
    split = _choisir([s["split"] for s in splits if s["config"] == config], meta["splits"])

    lignes, features, offset, total = [], None, 0, None
    while offset < max_lignes:
        url = (
            f"{API_HF}/rows?dataset={q(meta['dataset'])}&config={q(config)}"
            f"&split={q(split)}&offset={offset}&length=100"
        )
        page = _http_json(url)
        features = features or page.get("features", [])
        rows = page.get("rows", [])
        if not rows:
            break
        lignes.extend(r.get("row", {}) for r in rows)
        offset += len(rows)
        total = page.get("num_rows_total", total)
        if total is not None and offset >= total:
            break
        time.sleep(0.05)
    if not lignes:
        raise RuntimeError("aucune ligne reçue")
    return vers_jeu(pd.DataFrame(lignes), features, meta["codes"])


def charger_jeu(cle: str, dossier: Path = DOSSIER_CACHE) -> tuple[pd.DataFrame, str]:
    """Charge un jeu etiquete : copie locale si elle existe, sinon Hugging Face.

    La copie locale (donnees_eval/<cle>.csv) est ecrite au premier telechargement :
    l'evaluation fonctionne ensuite sans connexion, utile pour une soutenance.
    """
    meta = JEUX[cle]
    fichier = Path(dossier) / f"{cle}.csv"
    if fichier.exists():
        jeu = vers_jeu(pd.read_csv(fichier))
        if len(jeu):
            return jeu, f"{meta['court']} (copie locale, {len(jeu)} phrases)"

    erreurs = []
    jeu = None
    try:
        jeu = _via_api_hf(meta)
    except Exception as err:  # noqa: BLE001 - on tente la source suivante
        erreurs.append(f"API Hugging Face : {err}")
    if (jeu is None or jeu.empty) and meta.get("csv_secours"):
        try:
            jeu = vers_jeu(pd.read_csv(io.BytesIO(_http_get(meta["csv_secours"]))), None, meta["codes"])
        except Exception as err:  # noqa: BLE001
            erreurs.append(f"fichier direct : {err}")
    if jeu is None or jeu.empty:
        raise RuntimeError(
            f"Impossible de télécharger {meta['court']} ({' ; '.join(erreurs)}). "
            "Vérifiez la connexion, ou utilisez « Mon propre fichier annoté »."
        )
    try:
        Path(dossier).mkdir(exist_ok=True)
        jeu.to_csv(fichier, index=False)
    except OSError:
        pass  # dossier en lecture seule (hebergement en ligne) : pas de copie locale
    return jeu, f"{meta['court']} (Hugging Face, {len(jeu)} phrases)"


def lire_csv_utilisateur(fichier) -> pd.DataFrame:
    """Lit un CSV annote par l'utilisateur (separateur , ou ; detecte seul)."""
    brut = fichier.getvalue() if hasattr(fichier, "getvalue") else fichier
    if isinstance(brut, bytes):
        brut = brut.decode("utf-8-sig", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(brut), sep=None, engine="python")
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"Fichier illisible : {err}") from err
    jeu = vers_jeu(df)
    if jeu.empty:
        raise ValueError(
            "Aucune ligne exploitable. Les étiquettes acceptées sont : positif / "
            "négatif / neutre (ou positive, negative, neutral, haussier, baissier)."
        )
    return jeu


def echantillon(jeu: pd.DataFrame, n: int) -> pd.DataFrame:
    """Tirage aleatoire reproductible : il conserve la repartition reelle des classes."""
    if n >= len(jeu):
        return jeu.reset_index(drop=True)
    return jeu.sample(n=n, random_state=GRAINE).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Predictions
# ---------------------------------------------------------------------------
def _probas(sortie) -> dict:
    if isinstance(sortie, dict):  # pipeline appele sans top_k=None
        sortie = [sortie]
    probas = {c: 0.0 for c in CLASSES}
    for d in sortie:
        nom = normaliser_label(d["label"]) or str(d["label"]).lower()
        if nom in probas:
            probas[nom] = float(d["score"])
    return probas


def predire(textes: list[str], clf, taille_lot: int = 32, progression=None) -> pd.DataFrame:
    """Probabilites FinBERT, score (P+ - P-) et classe predite pour chaque texte."""
    lignes = []
    for debut in range(0, len(textes), taille_lot):
        lot = list(textes[debut : debut + taille_lot])
        for sortie in clf(lot, truncation=True, batch_size=16):
            p = _probas(sortie)
            lignes.append(
                {
                    "p_negative": p["negative"],
                    "p_neutral": p["neutral"],
                    "p_positive": p["positive"],
                    "score": p["positive"] - p["negative"],
                    "prediction": max(CLASSES, key=lambda c: p[c]),
                    "confiance": max(p.values()),
                }
            )
        if progression:
            progression(min(1.0, (debut + len(lot)) / max(1, len(textes))))
    return pd.DataFrame(lignes)


def regle_appli(score: float, seuil: float) -> str:
    """Classe deduite par l'application : score compare au seuil +/- SEUIL_LEGER."""
    if score >= seuil:
        return "positive"
    if score <= -seuil:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# 3. Metriques
# ---------------------------------------------------------------------------
def intervalle_wilson(succes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance a 95 % d'une proportion (methode de Wilson)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = succes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def evaluer(verite, prediction) -> dict:
    """Exactitude, F1 macro, precision / rappel par classe et matrice de confusion."""
    v, p = list(verite), list(prediction)
    n = len(v)
    idx = {c: i for i, c in enumerate(CLASSES)}
    m = np.zeros((3, 3), dtype=int)
    for a, b in zip(v, p):
        m[idx[a], idx[b]] += 1

    lignes, f1s = [], []
    for i, c in enumerate(CLASSES):
        tp, support, predits = m[i, i], m[i, :].sum(), m[:, i].sum()
        precision = tp / predits if predits else float("nan")
        rappel = tp / support if support else float("nan")
        if support and predits and (precision + rappel) > 0:
            f1 = 2 * precision * rappel / (precision + rappel)
        else:
            f1 = 0.0 if support else float("nan")
        if support:
            f1s.append(f1)
        lignes.append(
            {"Classe": NOMS_FR[c], "Précision": precision, "Rappel": rappel, "F1": f1, "Phrases": int(support)}
        )

    bons = int(np.trace(m))
    supports = m.sum(axis=1)
    i_maj = int(np.argmax(supports))

    hors_diag = m.copy()
    np.fill_diagonal(hors_diag, 0)
    confusion = None
    if hors_diag.sum():
        i, j = np.unravel_index(int(np.argmax(hors_diag)), hors_diag.shape)
        confusion = {
            "vrai": CLASSES[i],
            "predit": CLASSES[j],
            "n": int(hors_diag[i, j]),
            "part": hors_diag[i, j] / supports[i] if supports[i] else 0.0,
        }

    return {
        "n": n,
        "exactitude": bons / n if n else float("nan"),
        "ic95": intervalle_wilson(bons, n),
        "f1_macro": float(np.mean(f1s)) if f1s else float("nan"),
        "reference": supports[i_maj] / n if n else float("nan"),
        "classe_majoritaire": CLASSES[i_maj],
        "matrice": m,
        "par_classe": pd.DataFrame(lignes),
        "confusion_principale": confusion,
    }


def lecture_resultats(ev: dict, ev_regle: dict | None, nom_jeu: str) -> str:
    """Resume en francais courant, pour la soutenance."""
    bas, haut = ev["ic95"]
    texte = (
        f"Sur ces {ev['n']} phrases, FinBERT donne la bonne réponse dans "
        f"**{ev['exactitude']:.0%}** des cas (intervalle de confiance à 95 % : "
        f"{bas:.0%} à {haut:.0%}). Une référence naïve qui répondrait toujours "
        f"« {NOMS_FR[ev['classe_majoritaire']].lower()} » atteindrait {ev['reference']:.0%}."
    )
    c = ev["confusion_principale"]
    if c:
        texte += (
            f" Son erreur la plus fréquente : classer en « {NOMS_FR[c['predit']].lower()} » "
            f"des phrases jugées « {NOMS_FR[c['vrai']].lower()} » par les humains "
            f"({c['n']} cas, soit {c['part']:.0%} de ces phrases)."
        )
    if ev_regle is not None:
        ecart = ev_regle["exactitude"] - ev["exactitude"]
        if abs(ecart) < 0.01:
            texte += " La règle de seuils de l'application ne change presque rien à ce résultat."
        elif ecart < 0:
            texte += (
                f" La règle de seuils de l'application fait perdre {-ecart:.0%} "
                "d'exactitude par rapport au choix brut du modèle : ses seuils "
                "tranchent autrement sur les cas limites."
            )
        else:
            texte += (
                f" La règle de seuils de l'application gagne {ecart:.0%} d'exactitude "
                "par rapport au choix brut du modèle."
            )
    if nom_jeu == "phrasebank":
        texte += " Rappel : FinBERT a déjà vu ces phrases à l'entraînement."
    return texte


# ---------------------------------------------------------------------------
# 4. Explication mot par mot (occlusion)
# ---------------------------------------------------------------------------
def expliquer(texte: str, clf, max_mots: int = 60) -> dict | None:
    """Contribution de chaque mot au score FinBERT (P(positif) - P(negatif)).

    contribution = score du texte complet - score du texte sans ce mot.
    Positive : le mot tire vers haussier. Negative : il tire vers baissier.
    """
    mots = re.findall(r"\S+", str(texte).strip())[:max_mots]
    if not mots:
        return None
    complet = " ".join(mots)
    variantes = [" ".join(mots[:i] + mots[i + 1 :]) or "." for i in range(len(mots))]
    sorties = clf([complet] + variantes, truncation=True, batch_size=16)
    probas = [_probas(s) for s in sorties]
    scores = [p["positive"] - p["negative"] for p in probas]
    base = probas[0]
    return {
        "mots": mots,
        "contributions": [scores[0] - s for s in scores[1:]],
        "score": scores[0],
        "probas": base,
        "prediction": max(CLASSES, key=lambda c: base[c]),
    }


def html_explication(expl: dict) -> str:
    """Titre surligne : vert = pousse vers haussier, rouge = vers baissier."""
    contribs = expl["contributions"]
    echelle = max(0.15, max((abs(c) for c in contribs), default=0.0))
    morceaux = []
    for mot, c in zip(expl["mots"], contribs):
        intensite = min(1.0, abs(c) / echelle)
        if abs(c) < 0.02:
            fond = "transparent"
        elif c > 0:
            fond = f"rgba(22, 163, 74, {0.12 + 0.6 * intensite:.2f})"
        else:
            fond = f"rgba(220, 38, 38, {0.12 + 0.6 * intensite:.2f})"
        morceaux.append(
            f'<span title="{c:+.2f}" style="background:{fond};padding:2px 4px;'
            f'border-radius:4px;margin:0 1px;">{html.escape(mot)}</span>'
        )
    return (
        '<div style="font-size:1.08rem;line-height:2.1;margin:4px 0 8px 0;">'
        + " ".join(morceaux)
        + "</div>"
    )


def mots_dominants(expl: dict, n: int = 3) -> tuple[list, list]:
    paires = list(zip(expl["mots"], expl["contributions"]))
    hausse = [p for p in sorted(paires, key=lambda x: -x[1]) if p[1] >= 0.02][:n]
    baisse = [p for p in sorted(paires, key=lambda x: x[1]) if p[1] <= -0.02][:n]
    return hausse, baisse


# ---------------------------------------------------------------------------
# 5. Affichage Streamlit
# ---------------------------------------------------------------------------
def _en_fragment(st, fonction):
    """Les widgets d'un fragment ne relancent que leur bloc, pas toute l'application."""
    fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    return fragment(fonction) if fragment else fonction


def afficher_explication(st, texte: str, charger_modele) -> None:
    cache = st.session_state.setdefault("_explications_finbert", {})
    if texte not in cache:
        with st.spinner("FinBERT relit le titre mot par mot..."):
            cache[texte] = expliquer(texte, charger_modele())
    expl = cache[texte]
    if expl is None:
        st.caption("Texte vide : rien à expliquer.")
        return

    st.markdown(html_explication(expl), unsafe_allow_html=True)
    p = expl["probas"]
    st.caption(
        f"Verdict de FinBERT : **{NOMS_FR[expl['prediction']]}** · score {expl['score']:+.2f} · "
        f"probabilités : positif {p['positive']:.0%}, neutre {p['neutral']:.0%}, "
        f"négatif {p['negative']:.0%}. Survolez un mot pour voir sa contribution."
    )
    hausse, baisse = mots_dominants(expl)
    col_h, col_b = st.columns(2)
    with col_h:
        st.markdown("**🟩 Tire vers haussier**")
        if hausse:
            for mot, c in hausse:
                st.markdown(f"`{mot}` {c:+.2f}")
        else:
            st.caption("Aucun mot nettement haussier.")
    with col_b:
        st.markdown("**🟥 Tire vers baissier**")
        if baisse:
            for mot, c in baisse:
                st.markdown(f"`{mot}` {c:+.2f}")
        else:
            st.caption("Aucun mot nettement baissier.")


def afficher_explication_titres(details: pd.DataFrame | None, charger_modele) -> None:
    """Bloc a placer sous le tableau des titres analyses."""
    import streamlit as st

    def bloc():
        st.markdown("#### 🔎 Pourquoi FinBERT a-t-il donné ce score ?")
        st.caption(
            "Méthode par occlusion : on retire chaque mot, un à la fois, et on mesure "
            "de combien le score change. Vert : le mot pousse vers haussier. Rouge : "
            "vers baissier. Plus la couleur est franche, plus le mot pèse. Les mots "
            "sont testés un par un : une négation (« not », « less ») n'est captée qu'en partie."
        )
        if details is None or details.empty:
            st.info("Aucun titre à expliquer avec ce filtre.")
            return
        tri = (
            details.assign(_poids=details["score"].abs())
            .sort_values("_poids", ascending=False)
            .reset_index(drop=True)
        )

        def libelle(i: int) -> str:
            r = tri.iloc[i]
            titre = str(r["headline"])
            titre = titre if len(titre) <= 95 else titre[:94] + "…"
            return f"{r.get('ticker', '')} · {r['score']:+.2f} · {titre}"

        choix = st.selectbox(
            "Titre à expliquer (les plus tranchés en premier)",
            list(range(len(tri))),
            format_func=libelle,
            key="explication_titre_choisi",
        )
        if choix is not None and choix < len(tri):
            afficher_explication(st, str(tri.iloc[choix]["headline"]), charger_modele)

    _en_fragment(st, bloc)()


def _matrice_html(ev: dict) -> str:
    m = ev["matrice"]
    entete = "".join(
        f'<th style="padding:6px 10px;text-align:center;">{NOMS_FR[c]}</th>' for c in CLASSES
    )
    lignes = []
    for i, c in enumerate(CLASSES):
        total = m[i].sum()
        cellules = []
        for j in range(3):
            part = m[i, j] / total if total else 0.0
            if i == j:
                fond = f"rgba(22, 163, 74, {0.08 + 0.55 * part:.2f})"
            elif m[i, j]:
                fond = f"rgba(220, 38, 38, {0.08 + 0.7 * part:.2f})"
            else:
                fond = "transparent"
            cellules.append(
                f'<td style="padding:8px 10px;text-align:center;background:{fond};'
                f'border:1px solid rgba(128,128,128,0.25);">'
                f'<b>{m[i, j]}</b><br><span style="opacity:0.7;font-size:0.85em;">'
                f"{part:.0%}</span></td>"
            )
        lignes.append(
            f'<tr><th style="padding:6px 10px;text-align:left;">{NOMS_FR[c]}</th>'
            + "".join(cellules)
            + "</tr>"
        )
    return (
        '<table style="border-collapse:collapse;width:100%;">'
        '<tr><th style="padding:6px 10px;text-align:left;font-weight:normal;opacity:0.7;">'
        "Humains ↓ · FinBERT →</th>" + entete + "</tr>" + "".join(lignes) + "</table>"
    )


def _afficher_resultats(st, res: dict, charger_modele) -> None:
    ev, ev_regle = res["ev"], res["ev_regle"]
    st.markdown(f"**Résultats** · {res['source']} · {ev['n']} phrases · {res['horodatage']}")

    c1, c2, c3, c4 = st.columns(4)
    bas, haut = ev["ic95"]
    c1.metric(
        "Exactitude de FinBERT",
        f"{ev['exactitude']:.0%}",
        delta=f"{ev['exactitude'] - ev['reference']:+.0%} vs référence naïve",
        help=f"Part de bonnes réponses. Intervalle de confiance à 95 % : {bas:.0%} à {haut:.0%}.",
    )
    c2.metric(
        "F1 macro",
        f"{ev['f1_macro']:.2f}",
        help=(
            "Moyenne des F1 des trois classes, chacune comptant autant. Plus exigeant "
            "que l'exactitude quand une classe domine (souvent « neutre »)."
        ),
    )
    c3.metric(
        "Référence naïve",
        f"{ev['reference']:.0%}",
        help=(
            f"Exactitude obtenue en répondant toujours « {NOMS_FR[ev['classe_majoritaire']].lower()} ». "
            "Un modèle utile doit faire nettement mieux."
        ),
    )
    c4.metric(
        "Règle de l'application",
        f"{ev_regle['exactitude']:.0%}",
        delta=f"{ev_regle['exactitude'] - ev['exactitude']:+.0%} vs FinBERT brut",
        help=(
            f"L'application ne prend pas la classe la plus probable : elle classe un titre "
            f"haussier si P(positif) - P(négatif) ≥ {res['seuil']:.2f}, baissier si ≤ "
            f"-{res['seuil']:.2f}, neutre sinon. Cette colonne mesure l'effet de ce choix."
        ),
    )
    st.info(lecture_resultats(ev, ev_regle, res["cle"]))

    gauche, droite = st.columns([1, 1])
    with gauche:
        st.markdown("**Matrice de confusion**")
        st.markdown(_matrice_html(ev), unsafe_allow_html=True)
        st.caption(
            "Chaque ligne : les phrases d'une classe selon les humains, et ce qu'en a "
            "dit FinBERT. La diagonale verte, ce sont les bonnes réponses."
        )
    with droite:
        st.markdown("**Détail par classe**")
        tableau = ev["par_classe"].copy()
        for col in ("Précision", "Rappel"):
            tableau[col] = tableau[col].map(lambda x: "–" if pd.isna(x) else f"{x:.0%}")
        tableau["F1"] = tableau["F1"].map(lambda x: "–" if pd.isna(x) else f"{x:.2f}")
        st.dataframe(tableau, hide_index=True, use_container_width=True)
        st.caption(
            "Précision : quand FinBERT annonce cette classe, a-t-il raison ? "
            "Rappel : parmi les vraies phrases de cette classe, combien en trouve-t-il ?"
        )

    erreurs = res["predictions"][~res["predictions"]["correct"]].sort_values(
        "confiance", ascending=False
    )
    st.markdown("**Les erreurs les plus sûres d'elles**")
    if erreurs.empty:
        st.success("Aucune erreur sur cet échantillon.")
    else:
        st.caption(
            "Les phrases où FinBERT se trompe en étant le plus confiant : les plus "
            "instructives pour comprendre ses angles morts."
        )
        vue = erreurs.head(20).assign(
            Humains=lambda d: d["verite"].map(NOMS_FR),
            FinBERT=lambda d: d["prediction"].map(NOMS_FR),
            Confiance=lambda d: d["confiance"].map(lambda x: f"{x:.0%}"),
        )[["texte", "Humains", "FinBERT", "Confiance"]].rename(columns={"texte": "Phrase"})
        st.dataframe(vue, hide_index=True, use_container_width=True)

        textes = erreurs["texte"].head(20).tolist()
        choix = st.selectbox(
            "Expliquer une erreur mot par mot",
            list(range(len(textes))),
            format_func=lambda i: textes[i] if len(textes[i]) <= 110 else textes[i][:109] + "…",
            key=f"eval_erreur_{res['cle']}",
        )
        if choix is not None and choix < len(textes):
            afficher_explication(st, textes[choix], charger_modele)

    export = res["predictions"].rename(
        columns={"texte": "phrase", "verite": "etiquette_humaine", "prediction": "prediction_finbert"}
    )
    st.download_button(
        "Télécharger toutes les prédictions (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluation_finbert_{res['cle']}.csv",
        mime="text/csv",
        key=f"eval_export_{res['cle']}",
    )


def afficher_evaluation(charger_modele, seuil_appli: float, details_appli: pd.DataFrame | None = None) -> None:
    """Section complete d'evaluation de FinBERT (a placer dans un st.expander)."""
    import streamlit as st

    def bloc():
        st.markdown(
            "FinBERT est-il fiable ? On lui soumet des phrases **déjà classées par des "
            "humains**, puis on compte ses bonnes et ses mauvaises réponses."
        )
        options = ["titres", "phrasebank", "perso"]
        noms = {**{k: v["nom"] for k, v in JEUX.items()}, "perso": "Mon propre fichier annoté (CSV)"}
        cle = st.radio(
            "Données de test", options, format_func=noms.get, key="eval_jeu_choisi"
        )

        fichier = None
        if cle == "perso":
            st.caption(
                "Deux colonnes : `texte` et `label` (positif, négatif ou neutre ; les "
                "équivalents anglais et haussier / baissier sont acceptés)."
            )
            fichier = st.file_uploader("CSV annoté", type=["csv"], key="eval_csv_perso")
            if details_appli is not None and not details_appli.empty:
                modele = pd.DataFrame(
                    {"texte": details_appli["headline"].drop_duplicates(), "label": ""}
                )
                st.download_button(
                    f"Télécharger les {len(modele)} titres de votre dernière analyse, à annoter",
                    data=modele.to_csv(index=False).encode("utf-8"),
                    file_name="titres_a_annoter.csv",
                    mime="text/csv",
                    key="eval_modele_annotation",
                    help=(
                        "Remplissez la colonne label à la main, puis rechargez le "
                        "fichier : vous mesurez FinBERT sur vos propres titres."
                    ),
                )
        else:
            st.caption(JEUX[cle]["lecture"])
            if JEUX[cle]["avertissement"]:
                st.warning(JEUX[cle]["avertissement"])

        n = st.slider(
            "Nombre de phrases évaluées",
            100,
            2000,
            300,
            100,
            key="eval_taille",
            help=(
                "Tirage aléatoire fixe : relancer donne le même échantillon. 300 phrases "
                "prennent environ 10 à 30 secondes sur un ordinateur portable."
            ),
        )

        if st.button("Lancer l'évaluation", type="primary", key="eval_lancer"):
            try:
                if cle == "perso":
                    if fichier is None:
                        st.warning("Chargez d'abord un fichier CSV annoté.")
                        return
                    jeu, source = lire_csv_utilisateur(fichier), f"fichier {fichier.name}"
                else:
                    with st.spinner("Chargement des phrases annotées..."):
                        jeu, source = charger_jeu(cle)
                test = echantillon(jeu, n)
                with st.spinner("Chargement de FinBERT..."):
                    clf = charger_modele()
                barre = st.progress(0.0, text="FinBERT lit les phrases...")
                preds = predire(
                    test["texte"].tolist(),
                    clf,
                    progression=lambda x: barre.progress(x, text=f"FinBERT lit les phrases... {x:.0%}"),
                )
                barre.empty()
            except Exception as err:  # noqa: BLE001
                st.error(f"Évaluation impossible : {err}")
                return

            preds.insert(0, "texte", test["texte"].values)
            preds.insert(1, "verite", test["verite"].values)
            preds["regle_appli"] = preds["score"].map(lambda s: regle_appli(s, seuil_appli))
            preds["correct"] = preds["prediction"] == preds["verite"]
            st.session_state.setdefault("eval_resultats", {})[cle] = {
                "cle": cle,
                "source": source,
                "seuil": seuil_appli,
                "horodatage": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
                "predictions": preds,
                "ev": evaluer(preds["verite"], preds["prediction"]),
                "ev_regle": evaluer(preds["verite"], preds["regle_appli"]),
            }

        res = st.session_state.get("eval_resultats", {}).get(cle)
        if res:
            _afficher_resultats(st, res, charger_modele)

    _en_fragment(st, bloc)()
