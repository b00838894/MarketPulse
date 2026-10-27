#!/usr/bin/env python3
"""
design — Identite visuelle de MarketPulse
==========================================
Feuille de style et composants HTML de l'interface. Les composants sont des
fonctions pures qui renvoient du HTML : app.py les affiche avec
st.markdown(..., unsafe_allow_html=True). Aucun calcul financier ici.

Systeme visuel
- Fond gris-lavande, cartes blanches, ombre teintee de violet.
- Violet profond : marque et actions (boutons, selection).
- Trois couleurs de signal reservees au sens financier : vert = hausse,
  rouge = baisse, ambre = attente. Elles ne servent jamais a decorer.
- Element signature : la jauge « pouls », qui part du centre (0) et penche a
  droite (haussier) ou a gauche (baissier) selon le score composite.
- Police : Manrope, chiffres a chasse fixe pour comparer les scores d'un coup d'oeil.
"""

from __future__ import annotations

import html
import re

import numpy as np

ENCRE = "#1B1A33"
GRIS = "#6B6F86"
FOND = "#F4F5FA"
VIOLET = "#5534E8"
LAVANDE = "#ECE8FF"
LIGNE = "#E4E6EF"
HAUSSE = "#0F9D6E"
BAISSE = "#E5484D"
ATTENTE = "#C98A1B"
PRUDENCE = "#E07B2E"

TONS = {
    "hausse": {"couleur": HAUSSE, "fond": "#E3F6EF", "texte": "#0B7A55"},
    "baisse": {"couleur": BAISSE, "fond": "#FDE8E8", "texte": "#B42318"},
    "attente": {"couleur": ATTENTE, "fond": "#FBF1DC", "texte": "#8A5B0B"},
    "prudence": {"couleur": PRUDENCE, "fond": "#FEEBDD", "texte": "#B54708"},
    "neutre": {"couleur": GRIS, "fond": "#EEF0F6", "texte": "#4A4E66"},
}

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --mp-encre: #1B1A33; --mp-gris: #6B6F86; --mp-fond: #F4F5FA; --mp-carte: #FFFFFF;
  --mp-violet: #5534E8; --mp-violet-fonce: #3D22C4; --mp-lavande: #ECE8FF;
  --mp-ligne: #E4E6EF; --mp-hausse: #0F9D6E; --mp-baisse: #E5484D;
  --mp-ombre: 0 1px 2px rgba(27, 26, 51, .04), 0 14px 34px -14px rgba(61, 34, 196, .22);
}

/* Police : on vise les elements de texte, jamais les icones de Streamlit */
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp summary, .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"] {
  font-family: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--mp-fond); color: var(--mp-encre); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { letter-spacing: -0.02em; font-weight: 800; color: var(--mp-encre); }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 1.6rem; max-width: 1240px; }
[data-testid="stHeader"] { background: transparent; }

/* Conteneurs a bordure, expanders, metriques, tableaux */
/* Blocs encadres nommes dans app.py (st.container(border=True, key="mp_...")) */
.st-key-mp_panneau, .st-key-mp_detail {
  background: var(--mp-carte); border-radius: 22px; box-shadow: var(--mp-ombre);
}
[data-testid="stExpander"] details {
  background: var(--mp-carte); border: 1px solid var(--mp-ligne); border-radius: 18px;
}
[data-testid="stExpander"] summary { font-weight: 700; padding-top: .8rem; padding-bottom: .8rem; }
[data-testid="stExpander"] summary:hover { color: var(--mp-violet); }
[data-testid="stMetric"] {
  background: var(--mp-carte); border: 1px solid var(--mp-ligne); border-radius: 16px; padding: 14px 16px;
}
[data-testid="stMetricValue"] { font-weight: 800; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] p { color: var(--mp-gris); font-weight: 600; }
[data-testid="stDataFrame"] { border: 1px solid var(--mp-ligne); border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stCaptionContainer"], .stApp small { color: var(--mp-gris); }

/* Boutons : le violet est reserve a l'action principale */
.stButton button, .stDownloadButton button {
  border-radius: 999px; font-weight: 700; border: 1px solid var(--mp-ligne);
  background: var(--mp-carte); color: var(--mp-encre); transition: border-color .15s, color .15s;
}
.stButton button:hover, .stDownloadButton button:hover { border-color: var(--mp-violet); color: var(--mp-violet); }
.stButton button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: var(--mp-violet); border-color: var(--mp-violet); color: #fff; border-radius: 14px;
  min-height: 2.9rem; box-shadow: 0 10px 22px -10px rgba(85, 52, 232, .7);
}
.stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: var(--mp-violet-fonce); border-color: var(--mp-violet-fonce); color: #fff;
}
.stApp button:focus-visible, .stApp input:focus-visible, .stApp summary:focus-visible {
  outline: 3px solid rgba(85, 52, 232, .35); outline-offset: 2px;
}
.stTextInput input, [data-baseweb="select"] > div { border-radius: 12px; }

/* Onglets en commutateur */
[data-baseweb="tab-list"] { gap: 4px; background: var(--mp-lavande); padding: 4px; border-radius: 14px; }
[data-baseweb="tab"] { border-radius: 10px; padding: 6px 14px; font-weight: 700; }
[data-baseweb="tab"][aria-selected="true"] { background: var(--mp-carte); color: var(--mp-violet); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

/* ---------- Composants MarketPulse ---------- */
.mp-marque { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.05rem; color: var(--mp-encre); }
.mp-marque svg path { stroke-dasharray: 160; stroke-dashoffset: 160; animation: mp-trace 1.4s ease-out .2s forwards; }
@keyframes mp-trace { to { stroke-dashoffset: 0; } }
.mp-hero { margin: 1.4rem 0 1.1rem; max-width: 62rem; }
.mp-hero h1 { font-size: clamp(1.9rem, 4.2vw, 3rem); line-height: 1.08; margin: 0 0 .7rem; padding: 0; }
.mp-hero p { color: var(--mp-gris); font-size: 1.06rem; line-height: 1.55; max-width: 60ch; margin: 0; }
.mp-statuts { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1rem; }
.mp-statut {
  display: inline-flex; align-items: center; gap: 7px; padding: 5px 12px; border-radius: 999px;
  background: var(--mp-carte); border: 1px solid var(--mp-ligne); font-size: .84rem; font-weight: 600; color: #3A3D55;
}
.mp-point { width: 7px; height: 7px; border-radius: 50%; background: var(--mp-hausse); box-shadow: 0 0 0 3px rgba(15, 157, 110, .18); }
.mp-panneau-titre { font-weight: 800; font-size: 1.02rem; margin: 2px 0 2px; }
.mp-panneau-aide { color: var(--mp-gris); font-size: .9rem; margin-bottom: 6px; }

.mp-etapes { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: .4rem 0 1rem; }
.mp-etape { background: var(--mp-carte); border: 1px solid var(--mp-ligne); border-radius: 18px; padding: 18px; }
.mp-etape b { display: block; font-size: 1rem; margin: 10px 0 4px; }
.mp-etape span { color: var(--mp-gris); font-size: .92rem; line-height: 1.45; }
.mp-num {
  width: 30px; height: 30px; border-radius: 10px; background: var(--mp-lavande); color: var(--mp-violet);
  display: grid; place-items: center; font-weight: 800;
}

.mp-section { margin: 2.2rem 0 .8rem; }
.mp-section h3 { font-size: 1.45rem; margin: 0 0 .25rem; padding: 0; }
.mp-section p { color: var(--mp-gris); margin: 0; max-width: 70ch; }

.mp-tuile { background: var(--mp-lavande); border-radius: 16px; padding: 14px 16px; height: 100%; }
.mp-tuile-libelle { font-size: .86rem; font-weight: 600; color: #4A4370; }
.mp-tuile-valeur { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; margin-top: 2px; }
.mp-tuile-note { font-size: .82rem; color: var(--mp-gris); margin-top: 2px; }

.mp-focus {
  background: var(--mp-violet); color: #fff; border-radius: 24px; padding: 22px 26px; margin: 1rem 0 .4rem;
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px;
  box-shadow: 0 18px 40px -18px rgba(61, 34, 196, .75);
}
.mp-focus-libelle { font-size: .9rem; opacity: .82; font-weight: 600; }
.mp-focus-titre { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 2px 0 4px; }
.mp-focus-texte { font-size: .95rem; opacity: .9; max-width: 56ch; line-height: 1.45; }
.mp-focus-score { text-align: right; }
.mp-focus-score strong { display: block; font-size: 2.4rem; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1; }
.mp-focus-score span { font-size: .85rem; opacity: .82; }

.mp-carte {
  background: var(--mp-carte); border: 1px solid var(--mp-ligne); border-radius: 22px;
  padding: 18px 18px 14px; box-shadow: var(--mp-ombre); margin-bottom: 14px;
}
.mp-carte-tete { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.mp-ticker { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.01em; }
.mp-pastille { display: inline-block; padding: 4px 11px; border-radius: 999px; font-size: .82rem; font-weight: 700; white-space: nowrap; }
.mp-nuance { color: var(--mp-gris); font-size: .84rem; margin-top: 2px; min-height: 1.2em; }
.mp-jauge { display: block; width: 100%; max-width: 250px; margin: 6px auto 0; }
.mp-courbe-ligne { display: flex; justify-content: space-between; font-size: .8rem; color: var(--mp-gris); margin-top: 8px; }
.mp-courbe { display: block; width: 100%; height: 52px; }
.mp-confiance { margin-top: 10px; }
.mp-confiance-tete { display: flex; justify-content: space-between; font-size: .82rem; font-weight: 600; color: #3A3D55; }
.mp-piste { height: 8px; background: var(--mp-lavande); border-radius: 999px; margin-top: 5px; overflow: hidden; }
.mp-piste > div { height: 100%; background: var(--mp-violet); border-radius: 999px; }
.mp-faits { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--mp-ligne); }
.mp-faits dt { font-size: .74rem; color: var(--mp-gris); font-weight: 600; }
.mp-faits dd { margin: 0; font-weight: 800; font-variant-numeric: tabular-nums; font-size: .95rem; }

.mp-suite { background: var(--mp-carte); border: 1px solid var(--mp-ligne); border-radius: 22px; padding: 20px 22px; margin-top: 2rem; }
.mp-suite h4 { margin: 0 0 .6rem; font-size: 1.1rem; }
.mp-suite ul { margin: 0; padding-left: 1.1rem; color: #3A3D55; line-height: 1.7; }
.mp-pied { color: var(--mp-gris); font-size: .82rem; line-height: 1.5; max-width: 80ch; margin: 1.6rem 0 .6rem; }

@media (max-width: 720px) {
  .mp-etapes { grid-template-columns: 1fr; }
  .mp-faits { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .mp-focus-score { text-align: left; }
}
@media (prefers-reduced-motion: reduce) {
  .mp-marque svg path { animation: none; stroke-dashoffset: 0; }
}
"""


def _e(texte) -> str:
    return html.escape(str(texte))


def appliquer_style() -> None:
    import streamlit as st

    st.markdown(f"<style>{STYLE}</style>", unsafe_allow_html=True)


def cadre(cle: str):
    """Bloc encadre nomme (classe CSS st-key-<cle>). Les versions de Streamlit qui
    ne connaissent pas encore key= pour les conteneurs gardent le cadre standard."""
    import streamlit as st

    try:
        return st.container(border=True, key=cle)
    except TypeError:
        return st.container(border=True)


# ---------------------------------------------------------------------------
# Lecture des recommandations
# ---------------------------------------------------------------------------
def ton_recommandation(recommandation: str) -> str:
    debut = str(recommandation).lower()
    if debut.startswith(("achat", "accumuler")):
        return "hausse"
    if debut.startswith("vente"):
        return "baisse"
    if debut.startswith("prudence"):
        return "prudence"
    if debut.startswith("attente"):
        return "attente"
    return "neutre"


def ton_score(valeur: float | None, seuil: float = 0.15) -> str:
    if valeur is None or not np.isfinite(valeur):
        return "neutre"
    if valeur >= seuil:
        return "hausse"
    if valeur <= -seuil:
        return "baisse"
    return "attente"


def decouper_recommandation(recommandation: str) -> tuple[str, str]:
    """'Achat — conviction moderee' -> ('Achat', 'conviction moderee')."""
    morceaux = str(recommandation).split(" — ", 1)
    return morceaux[0].strip(), (morceaux[1].strip() if len(morceaux) > 1 else "")


def pastille(recommandation: str) -> str:
    ton = TONS[ton_recommandation(recommandation)]
    verdict, _ = decouper_recommandation(recommandation)
    return (
        f'<span class="mp-pastille" style="background:{ton["fond"]};color:{ton["texte"]};">'
        f"{_e(verdict)}</span>"
    )


# ---------------------------------------------------------------------------
# Graphiques SVG
# ---------------------------------------------------------------------------
def _point(angle_deg: float, rayon: float = 80, cx: float = 100, cy: float = 100) -> tuple[float, float]:
    a = np.radians(angle_deg)
    return cx + rayon * np.cos(a), cy - rayon * np.sin(a)


def jauge(valeur: float | None, libelle: str = "score composite", confiance: float = 1.0) -> str:
    """Jauge « pouls » : demi-cercle de -1 (gauche) a +1 (droite), arc depuis le centre.

    L'arc s'estompe quand la confiance est faible : un signal que l'application
    juge elle-meme peu fiable ne doit pas s'afficher en couleur pleine.
    """
    if valeur is None or not np.isfinite(valeur):
        v, texte = 0.0, "n/d"
    else:
        v = float(np.clip(valeur, -1, 1))
        texte = f"{v:+.2f}"
    couleur = TONS[ton_score(v)]["couleur"]
    opacite = 0.35 + 0.65 * float(np.clip(confiance if confiance is not None else 1.0, 0, 1))
    x0, y0 = _point(180)
    x1, y1 = _point(0)
    xv, yv = _point(90 - v * 90)
    arc = ""
    if abs(v) >= 0.005:
        balayage = 1 if v > 0 else 0
        arc = (
            f'<path d="M100 20 A80 80 0 0 {balayage} {xv:.1f} {yv:.1f}" fill="none" '
            f'stroke="{couleur}" stroke-opacity="{opacite:.2f}" stroke-width="14" stroke-linecap="round"/>'
        )
    return (
        f'<svg class="mp-jauge" viewBox="0 0 200 128" role="img" '
        f'aria-label="{_e(libelle)} {texte}, de -1 baissier à +1 haussier">'
        f'<path d="M{x0:.1f} {y0:.1f} A80 80 0 0 1 {x1:.1f} {y1:.1f}" fill="none" '
        f'stroke="{LAVANDE}" stroke-width="14" stroke-linecap="round"/>'
        f'<line x1="100" y1="8" x2="100" y2="32" stroke="{LIGNE}" stroke-width="2"/>'
        f"{arc}"
        f'<circle cx="{xv:.1f}" cy="{yv:.1f}" r="8" fill="#fff" stroke="{couleur}" '
        f'stroke-opacity="{opacite:.2f}" stroke-width="4"/>'
        f'<text x="100" y="92" text-anchor="middle" font-size="30" font-weight="800" '
        f'fill="{ENCRE}" style="font-variant-numeric:tabular-nums">{texte}</text>'
        f'<text x="100" y="112" text-anchor="middle" font-size="11" fill="{GRIS}">{_e(libelle)}</text>'
        f'<text x="20" y="124" text-anchor="middle" font-size="10" fill="{GRIS}">baisse</text>'
        f'<text x="180" y="124" text-anchor="middle" font-size="10" fill="{GRIS}">hausse</text>'
        "</svg>"
    )


def courbe(cours, identifiant: str = "c") -> str:
    """Mini-courbe des cours avec aire degradee, verte si le cours a monte."""
    try:
        valeurs = np.asarray([float(x) for x in cours], dtype=float)
    except (TypeError, ValueError):
        return ""
    valeurs = valeurs[np.isfinite(valeurs)]
    if len(valeurs) < 2:
        return ""
    bas, haut = valeurs.min(), valeurs.max()
    etendue = (haut - bas) or 1.0
    l, h = 240, 52
    xs = np.linspace(0, l, len(valeurs))
    ys = h - 4 - (valeurs - bas) / etendue * (h - 8)
    trace = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    couleur = HAUSSE if valeurs[-1] >= valeurs[0] else BAISSE
    gid = "g" + re.sub(r"[^A-Za-z0-9]", "", str(identifiant))
    return (
        f'<svg class="mp-courbe" viewBox="0 0 {l} {h}" preserveAspectRatio="none" aria-hidden="true">'
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{couleur}" stop-opacity=".22"/>'
        f'<stop offset="1" stop-color="{couleur}" stop-opacity="0"/></linearGradient></defs>'
        f'<polygon points="0,{h} {trace} {l},{h}" fill="url(#{gid})"/>'
        f'<polyline points="{trace}" fill="none" stroke="{couleur}" stroke-width="2" '
        'vector-effect="non-scaling-stroke" stroke-linejoin="round"/>'
        "</svg>"
    )


# ---------------------------------------------------------------------------
# Blocs de page
# ---------------------------------------------------------------------------
MARQUE_SVG = (
    '<svg width="34" height="22" viewBox="0 0 68 44" aria-hidden="true">'
    f'<path d="M2 24 H18 L24 8 L32 38 L40 14 L46 24 H66" fill="none" stroke="{VIOLET}" '
    'stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


def entete(statuts: list[str] | None = None) -> str:
    puces = "".join(
        f'<span class="mp-statut">{"<i class=mp-point></i>" if i == 0 else ""}{_e(s)}</span>'
        for i, s in enumerate(statuts or [])
    )
    return (
        f'<div class="mp-marque">{MARQUE_SVG}MarketPulse</div>'
        '<div class="mp-hero"><h1>Le pouls des marchés, lu dans la presse.</h1>'
        "<p>Choisissez des actions : MarketPulse lit les derniers titres de presse avec "
        "FinBERT, les croise avec les cours et le climat géopolitique, puis vous dit où "
        "se placer, et pourquoi.</p>"
        f'<div class="mp-statuts">{puces}</div></div>'
    )


def panneau_titre(titre: str, aide: str = "") -> str:
    return (
        f'<div class="mp-panneau-titre">{_e(titre)}</div>'
        + (f'<div class="mp-panneau-aide">{_e(aide)}</div>' if aide else "")
    )


def etapes_accueil() -> str:
    etapes = [
        ("Choisissez un panier", "Un clic sur un secteur, ou vos propres tickers séparés par une virgule."),
        ("Lancez l'analyse", "FinBERT lit jusqu'à 25 titres de presse récents par action."),
        ("Lisez la recommandation", "Chaque verdict est expliqué : facteurs, scénarios et seuils de bascule."),
    ]
    blocs = "".join(
        f'<div class="mp-etape"><div class="mp-num">{i}</div><b>{_e(t)}</b><span>{_e(d)}</span></div>'
        for i, (t, d) in enumerate(etapes, start=1)
    )
    return f'<div class="mp-etapes">{blocs}</div>'


def titre_section(titre: str, texte: str = "") -> str:
    return (
        f'<div class="mp-section"><h3>{_e(titre)}</h3>'
        + (f"<p>{_e(texte)}</p>" if texte else "")
        + "</div>"
    )


def tuile(libelle: str, valeur: str, note: str = "", ton: str = "neutre") -> str:
    couleur = TONS.get(ton, TONS["neutre"])["couleur"] if ton != "neutre" else ENCRE
    return (
        f'<div class="mp-tuile"><div class="mp-tuile-libelle">{_e(libelle)}</div>'
        f'<div class="mp-tuile-valeur" style="color:{couleur};">{_e(valeur)}</div>'
        + (f'<div class="mp-tuile-note">{_e(note)}</div>' if note else "")
        + "</div>"
    )


def climat_vix(percentile: float | None) -> tuple[str, str, str]:
    """(mot, note, ton) a partir du percentile du VIX sur un an."""
    if percentile is None or not np.isfinite(percentile):
        return "n/d", "VIX indisponible", "neutre"
    note = f"VIX plus haut que {percentile:.0%} de l'année écoulée"
    if percentile < 0.4:
        return "Calme", note, "hausse"
    if percentile < 0.75:
        return "Nerveux", note, "attente"
    return "Tendu", note, "baisse"


def bandeau_focus(ticker: str, recommandation: str, composite: float, confiance: float) -> str:
    verdict, nuance = decouper_recommandation(recommandation)
    titre = ticker
    if composite >= 0.15:
        libelle = "L'action la mieux placée du panier"
        texte = (
            f"{verdict}{' (' + nuance + ')' if nuance else ''}. Ouvrez son analyse détaillée "
            "pour voir les facteurs qui portent ce signal et ce qui pourrait le faire basculer."
        )
    else:
        libelle = "Pas de signal net sur ce panier"
        titre = "Aucune action ne se détache"
        texte = (
            f"La mieux notée, {ticker}, reste en « {verdict.lower()} ». Quand le marché "
            "ne tranche pas, attendre est aussi une décision."
        )
    return (
        '<div class="mp-focus"><div>'
        f'<div class="mp-focus-libelle">{_e(libelle)}</div>'
        f'<div class="mp-focus-titre">{_e(titre)}</div>'
        f'<div class="mp-focus-texte">{_e(texte)}</div></div>'
        f'<div class="mp-focus-score"><strong>{composite:+.2f}</strong>'
        f"<span>score composite, confiance {confiance:.0%}</span></div></div>"
    )


def _fait(libelle: str, valeur: str) -> str:
    return f"<div><dt>{_e(libelle)}</dt><dd>{_e(valeur)}</dd></div>"


def carte_action(
    ticker: str,
    recommandation: str,
    composite: float,
    confiance: float,
    sentiment: float,
    nb_titres: int,
    momentum: float | None = None,
    rsi: float | None = None,
    cours=None,
) -> str:
    _, nuance = decouper_recommandation(recommandation)
    bloc_courbe = ""
    if cours is not None and len(cours) >= 2:
        valeurs = [float(x) for x in cours]
        variation = valeurs[-1] / valeurs[0] - 1 if valeurs[0] else 0.0
        bloc_courbe = (
            f'<div class="mp-courbe-ligne"><span>Cours sur 3 mois</span>'
            f'<span style="color:{HAUSSE if variation >= 0 else BAISSE};font-weight:700;">'
            f"{variation:+.1%}</span></div>{courbe(valeurs, ticker)}"
        )
    confiance = float(np.clip(confiance, 0, 1))
    faits = (
        _fait("Articles", str(nb_titres))
        + _fait("Presse", f"{sentiment:+.2f}")
        + _fait("Momentum", f"{momentum:+.2f}" if momentum is not None else "n/d")
        + _fait("RSI", f"{rsi:.0f}" if rsi else "n/d")
    )
    return (
        '<div class="mp-carte">'
        f'<div class="mp-carte-tete"><span class="mp-ticker">{_e(ticker)}</span>{pastille(recommandation)}</div>'
        f'<div class="mp-nuance">{_e(nuance)}</div>'
        f"{jauge(composite, confiance=confiance)}"
        f"{bloc_courbe}"
        '<div class="mp-confiance"><div class="mp-confiance-tete">'
        f"<span>Confiance du signal</span><span>{confiance:.0%}</span></div>"
        f'<div class="mp-piste"><div style="width:{confiance * 100:.0f}%;"></div></div></div>'
        f'<dl class="mp-faits">{faits}</dl>'
        "</div>"
    )


def suite_parcours() -> str:
    return (
        '<div class="mp-suite"><h4>Et maintenant ?</h4><ul>'
        "<li>Comparez avec un autre secteur : un panier différent se lance en un clic, en haut de page.</li>"
        "<li>Testez la robustesse : dans les réglages, changez le poids d'un facteur et regardez si le verdict tient.</li>"
        "<li>Vérifiez le modèle : le panneau « Fiabilité de FinBERT » mesure ses erreurs sur des phrases annotées.</li>"
        "</ul></div>"
    )


def pied_de_page(texte: str) -> str:
    return f'<p class="mp-pied">{_e(texte)}</p>'
