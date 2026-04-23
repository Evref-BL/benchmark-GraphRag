# benchmark-graphRag

Benchmark pour évaluer la capacité d’un GraphRAG à localiser les classes Java impactées à partir de descriptions d’issues GitHub.

Le workflow est en 2 étapes:
1. Miner un repository GitHub pour construire un dataset `Issue -> MRs merged -> impacted files`.
2. Exécuter GraphRAG sur chaque issue retenue et calculer `precision`, `recall`, `F1`.

## Objectif

À partir d’une issue (titre + description), retrouver les classes Java effectivement modifiées dans les merge requests associées à cette issue.

La vérité terrain est dérivée des fichiers modifiés dans les PR mergées liées à l’issue.

## Structure du projet

```text
benchmark-graphRag/
├── scripts/
│   ├── mine_github_issues.py
│   ├── evaluator_core.py
│   ├── evaluate_graphrag_benchmark.py
│   ├── evaluate_llm_api_benchmark.py
│   ├── evaluate_colgrep_benchmark.py
│   ├── evaluate_random_benchmark.py
│   ├── evaluate_manual_input_benchmark.py
│   └── export_benchmark_queries.py
├── mined_projects/              # sorties JSON (ignoré par git)
├── evaluation_results/          # rapports evaluator (ignoré par git)
├── queries/                     # queries LLM exportées (ignoré par git)
├── .env.github                  # token GitHub local (ignoré par git)
└── README.md
```

## Prérequis

- Python 3.10+
- GraphRAG installé:
  - soit via un `.venv` local contenant `graphrag`,
  - soit via `uv`.
- Un projet GraphRAG déjà indexé, avec un dossier `output` exploitable.
- Optionnel mais fortement recommandé: un token GitHub (sinon rate limits API très bas).

## Architecture Evaluators

Les évaluateurs reposent sur une base commune:
- `BaseBenchmarkEvaluator` dans `scripts/evaluator_core.py` gère le pipeline séquentiel complet (chargement benchmark, filtrage, prompts, métriques, rapport).
- `evaluate_graphrag_benchmark.py` est une spécialisation GraphRAG (appel `.venv/bin/graphrag query` ou `uv run ...` selon le mode d’exécution).
- `evaluate_llm_api_benchmark.py` est une spécialisation API LLM (Ollama/OpenAI/Mistral).
- `evaluate_colgrep_benchmark.py` est une spécialisation `colgrep` (recherche sémantique embedding + ranking).
- `evaluate_random_benchmark.py` est une spécialisation baseline aléatoire (tirage de fichiers depuis le projet source).

Nouveautés d’évaluation robustes (communes):
- `in_repo_only`: métriques recalculées en ne gardant, côté vérité terrain, que les fichiers Java réellement présents dans un projet local de référence.
- `IC95%`: intervalles de confiance à 95% via bootstrap (micro + macro).
- reporting enrichi au niveau global et par issue.

## Configuration du token GitHub

Créer/éditer `.env.github` à la racine:

```bash
GITHUB_TOKEN=ghp_xxx
```

Le script de mining résout le token dans cet ordre:
1. `--token`
2. variable d’environnement `GITHUB_TOKEN`
3. fichier `.env.github` (répertoire courant puis racine du projet)

## 1) Mining GitHub

Script: `scripts/mine_github_issues.py`

### Ce que le script retient

Une issue est conservée si:
- elle est `closed`,
- au moins une PR mentionnée dans sa conversation est `merged`,
- la fermeture de l’issue est postérieure (ou égale) au merge d’au moins une de ces PR.

Les PR liées sont détectées dans:
- le body de l’issue,
- les commentaires de l’issue,
- la timeline de l’issue.

Si plusieurs PR mergées sont liées à une même issue, elles sont toutes conservées.

### Commandes

Mining simple (sortie par défaut dans `mined_projects/`):

```bash
python3 scripts/mine_github_issues.py "https://github.com/stleary/JSON-java"
```

Plusieurs repos:

```bash
python3 scripts/mine_github_issues.py \
  "https://github.com/mybatis/jpetstore-6" \
  "https://github.com/NanoHttpd/nanohttpd"
```

Limiter le nombre d’issues scannées:

```bash
python3 scripts/mine_github_issues.py "https://github.com/stleary/JSON-java" --issue-limit 300
```

`--issue-limit` (alias `--max-issues`) vaut `200` par défaut.

### Format de sortie (fichier de mining)

Un fichier JSON par projet, par exemple:
`mined_projects/stleary__json-java__YYYYMMDDTHHMMSSZ.json`

Structure simplifiée:

```json
{
  "project_name": "stleary/json-java",
  "github_url": "https://github.com/stleary/JSON-java",
  "issues": [
    {
      "number": 296,
      "title": "...",
      "description_message": "...",
      "linked_merged_pull_requests": [
        {
          "number": 646,
          "url": "...",
          "merged_at": "...",
          "impacted_files": [
            "src/main/java/org/json/JSONObject.java"
          ]
        }
      ]
    }
  ]
}
```

## 2) Évaluation GraphRAG

Script: `scripts/evaluate_graphrag_benchmark.py`

Le script:
1. lit un fichier de mining,
2. construit une requête par issue avec `title + description + prompt`,
   - ajoute systématiquement un pré-prompt obligatoire aligné sur le `response_type` par défaut du script,
3. exécute GraphRAG:
   - en mode `venv`: `<graphrag-dir>/.venv/bin/graphrag query --root . --method <method> --data ./output --response-type "<valeur par défaut interne du script>" "<query>"`
   - en mode `uv`: `uv run python -m graphrag query --root . --method <method> --data ./output --response-type "<valeur par défaut interne du script>" "<query>"`
4. extrait les classes prédites,
5. compare aux fichiers Java attendus,
6. calcule les métriques par issue et globales.

Le `response_type` n’est pas paramétrable en ligne de commande: il est fixé dans le script pour garantir un format de sortie stable.

### Commande type

```bash
python3 scripts/evaluate_graphrag_benchmark.py \
  mined_projects/stleary__json-java__20260420T090843Z.json \
  --graphrag-dir /path/to/graphrag \
  --execution-mode auto \
  --method local
```

### Méthode GraphRAG configurable

`--method` accepte uniquement:
- `local`
- `drift`
- `global`
- `basic`

Exemples:

```bash
python3 scripts/evaluate_graphrag_benchmark.py <mined.json> --graphrag-dir /path/to/graphrag --method drift
python3 scripts/evaluate_graphrag_benchmark.py <mined.json> --graphrag-dir /path/to/graphrag --method global
python3 scripts/evaluate_graphrag_benchmark.py <mined.json> --graphrag-dir /path/to/graphrag --method basic
```

### Options utiles

- `--issue-limit N`: limite le nombre d’issues évaluées.
- `--output-dir /path/folder`: dossier de sortie des rapports (par défaut: `evaluation_results/`).
- `--output-file /path/report.json`: chemin de sortie du rapport.
- `--timeout-seconds 300`: timeout par requête GraphRAG.
- `--extra-prompt "..."`: suffixe de prompt custom.
- `--keep-raw-response`: inclut la réponse brute GraphRAG dans le rapport.
- `--include-empty-java`: inclut aussi les issues sans cible Java.
- `--dry-run`: n’exécute pas GraphRAG (test pipeline uniquement).
- `--project-root /path/to/source`: active les métriques `in_repo_only` avec ce repo local.
- `--bootstrap-samples 1000`: nombre de tirages bootstrap pour IC95%.
- `--bootstrap-seed 42`: seed RNG du bootstrap.
- `--execution-mode auto|venv|uv`:
  - `auto` (défaut): essaie d’abord `<graphrag-dir>/.venv/bin/graphrag`, sinon fallback `uv`.
  - `venv`: force l’usage de `<graphrag-dir>/.venv/bin/graphrag`.
  - `uv`: force l’usage de `uv run python -m graphrag`.

### Métriques calculées

Par issue:
- `TP`: classes Java attendues retrouvées.
- `FP`: classes prédites non matchées.
- `FN`: classes attendues non retrouvées.
- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `f1 = 2 * precision * recall / (precision + recall)`

Global:
- `micro` (agrégation globale TP/FP/FN)
- `macro` (moyenne des scores par issue)
- `confidence_interval_95` (IC95% bootstrap pour `micro` et `macro`)
- `in_repo_only` (même métriques, restreintes aux cibles présentes dans le repo local de référence)

### Détail IC95% (bootstrap)

L’`IC95%` est un intervalle de confiance calculé par bootstrap non paramétrique:
- on prend la liste des issues évaluées (hors erreurs),
- on fait `B` rééchantillonnages avec remise (par défaut `B=1000`), chaque échantillon ayant la même taille que le nombre d’issues,
- à chaque tirage, on recalcule les métriques `micro` et `macro`,
- on obtient ainsi une distribution de scores,
- l’IC95% est donné par les percentiles 2.5% (borne basse) et 97.5% (borne haute).

Interprétation rapide:
- intervalle serré: estimation stable,
- intervalle large: forte variabilité entre issues.

Dans ce projet:
- `--bootstrap-samples` règle le nombre de rééchantillonnages,
- `--bootstrap-seed` fixe la reproductibilité des IC.
- un IC95% est calculé pour les métriques globales et, quand activé, pour `in_repo_only`.

### Matching attendu vs prédit

Le matching accepte plusieurs formats côté prédiction:
- chemin `.java`,
- nom pleinement qualifié (FQCN),
- nom de classe simple.

Si un nom simple est ambigu (plusieurs classes possibles), il n’est pas matché automatiquement.

### Pré-prompt obligatoire

Chaque requête envoyée à GraphRAG inclut automatiquement un pré-prompt obligatoire qui impose:
- un format strict conforme au `response_type` par défaut du script,
- aucune explication hors format attendu.

## Rapport d’évaluation

Sortie par défaut: dans `evaluation_results/`, avec suffixe `__graphrag_eval__<timestamp>.json`.

Le rapport contient:
- les paramètres d’exécution,
- un résumé global,
- les métriques par issue,
- les classes attendues / prédites.

## Workflow recommandé

1. Préparer le token GitHub (`.env.github`).
2. Miner un projet:
   - `python3 scripts/mine_github_issues.py "<repo-url>"`
3. Lancer l’évaluation GraphRAG:
   - `python3 scripts/evaluate_graphrag_benchmark.py <mined-file.json> --graphrag-dir <path> --execution-mode auto --method local`
4. Comparer les résultats entre `local`, `drift`, `global`.

## 3) Export des queries LLM

Script: `scripts/export_benchmark_queries.py`

Objectif:
- générer un JSON contenant uniquement les queries à envoyer au LLM,
- en réutilisant la même construction de prompt que l’evaluator,
- inclure le contexte d’issue et les chemins de classes attendues.

Sortie:
- par défaut dans `queries/`,
- un fichier nommé selon le projet benchmark, par exemple:
  - `queries/mybatis__jpetstore-6__queries.json`

Commande type:

```bash
python3 scripts/export_benchmark_queries.py \
  mined_projects/mybatis__jpetstore-6__20260420T093019Z.json
```

Options utiles:
- `--output-dir /path/folder`
- `--output-file /path/file.json`
- `--issue-limit N`
- `--include-empty-java`
- `--extra-prompt "..."`

## 4) Évaluation Via API LLM (Ollama/OpenAI/Mistral)

Script: `scripts/evaluate_llm_api_benchmark.py`

Objectif:
- évaluer le benchmark sans passer par `graphrag query`,
- interroger directement une API LLM (Ollama local, OpenAI, Mistral),
- calculer les mêmes métriques (`precision`, `recall`, `f1`).

Exemples:

Ollama (local):

```bash
python3 scripts/evaluate_llm_api_benchmark.py \
  mined_projects/mybatis__jpetstore-6__20260420T093019Z.json \
  --provider ollama \
  --model gemma2 \
  --keep-raw-response
```

OpenAI:

```bash
export OPENAI_API_KEY="..."
python3 scripts/evaluate_llm_api_benchmark.py \
  mined_projects/mybatis__jpetstore-6__20260420T093019Z.json \
  --provider openai \
  --model gpt-4.1-mini
```

Mistral:

```bash
export MISTRAL_API_KEY="..."
python3 scripts/evaluate_llm_api_benchmark.py \
  mined_projects/mybatis__jpetstore-6__20260420T093019Z.json \
  --provider mistral \
  --model mistral-small-latest
```

Options utiles:
- `--base-url` pour un endpoint custom,
- `--api-key` pour surcharger la clé env,
- `--temperature`, `--max-tokens`,
- `--issue-limit`,
- `--project-root` pour activer `in_repo_only`,
- `--bootstrap-samples`, `--bootstrap-seed` pour IC95%,
- `--keep-raw-response`,
- `--dry-run`.

## 5) Évaluation Semi-Automatique (saisie manuelle)

Script: `scripts/evaluate_manual_input_benchmark.py`

Objectif:
- lire un benchmark,
- construire la query pour chaque issue,
- afficher cette query à l’utilisateur,
- attendre la réponse manuelle (copier/coller depuis un LLM/GraphRAG),
- calculer automatiquement les métriques.

Commande type:

```bash
python3 scripts/evaluate_manual_input_benchmark.py \
  mined_projects/mybatis__jpetstore-6__20260420T093019Z.json \
  --keep-raw-response
```

Mode de saisie:
- le script affiche la query,
- la query est automatiquement copiée dans le presse-papier (si supporté par l’environnement),
- tu colles la réponse multi-ligne,
- tu termines en faisant simplement `Entrée` sur une ligne vide,
- les lignes vides internes dans un gros collage multi-ligne sont désormais tolérées (le script évite de passer involontairement à l’issue suivante),
- optionnellement, tu peux aussi terminer avec `EOF` (ou un token custom via `--end-token`),
- si tu tapes `STOP` sur une ligne, l’évaluation s’arrête immédiatement et un fichier de résultats partiels est généré avec les données déjà collectées.

Options utiles:
- `--issue-limit N`
- `--include-empty-java`
- `--keep-raw-response`
- `--end-token DONE`
- `--no-copy-query-to-clipboard`
- `--project-root` pour activer `in_repo_only`
- `--bootstrap-samples`, `--bootstrap-seed` pour IC95%

Format de sortie (par entrée):
- `issue_number`
- `query`
- `expected_classes_paths`

## 6) Évaluation Baseline Aléatoire

Script: `scripts/evaluate_random_benchmark.py`

Objectif:
- construire une baseline aléatoire pour comparaison,
- lister les fichiers d’un projet source selon une extension (par défaut `.java`),
- pour chaque issue, tirer un nombre `N`, puis tirer `N` fichiers aléatoires dans ce pool,
- stratégie recommandée par défaut: `size-matched` (N = nombre de cibles attendues présentes dans le repo local),
- produire le même format de rapport JSON que les autres évaluateurs.

Commande type:

```bash
python3 scripts/evaluate_random_benchmark.py \
  mined_projects/stleary__json-java__20260420T090843Z.json \
  --project-root /path/to/source/project \
  --file-extension .java \
  --keep-raw-response
```

Options utiles:
- `--project-root /path/to/project` (obligatoire)
- `--file-extension .java` (défaut: `.java`, configurable)
- `--sampling-strategy size-matched|uniform` (défaut: `size-matched`)
- `--random-n-min 1` (défaut: 1)
- `--random-n-max 50` (défaut: taille totale du pool)
- `--seed 42` pour rejouer exactement le même tirage
- `--bootstrap-samples`, `--bootstrap-seed` pour IC95%
- `--issue-limit N`
- `--include-empty-java`
- `--keep-raw-response`

## 7) Évaluation Colgrep (Embedding Search)

Script: `scripts/evaluate_colgrep_benchmark.py`

Objectif:
- évaluer un moteur de recherche de code basé embeddings (`colgrep`),
- exécuter une recherche sémantique avec la query issue (titre + description + prompt),
- récupérer les fichiers retournés et calculer `precision/recall/f1` comme les autres évaluateurs.

Commande type:

```bash
python3 scripts/evaluate_colgrep_benchmark.py \
  mined_projects/stleary__json-java__20260420T090843Z.json \
  --project-root /Users/nicolashlad/Development/Projects/JSON-java \
  --results 15 \
  --include-pattern "*.java" \
  --keep-raw-response
```

Options utiles:
- `--project-root /path/to/project` (obligatoire)
- `--colgrep-bin colgrep` (binaire colgrep custom si besoin)
- `--results 15` (top-k)
- `--include-pattern "*.java"` (filtre fichiers)
- `--bootstrap-samples`, `--bootstrap-seed` pour IC95%
- `--issue-limit N`
- `--include-empty-java`
- `--keep-raw-response`

## Troubleshooting

- Erreur `API rate limit exceeded`:
  - vérifier `GITHUB_TOKEN` (`.env.github` ou env var).
- Erreur `GraphRAG directory not found`:
  - passer `--graphrag-dir` vers le bon dossier.
- Beaucoup de `FP`:
  - réduire le prompt à une sortie plus contrainte,
  - ajuster le `DEFAULT_RESPONSE_TYPE` directement dans le script,
  - vérifier l’ambiguïté des noms de classes simples.
- Pas d’issues pertinentes:
  - le repository n’a peut-être pas d’issues fermées reliées à des PR mergées selon les règles du miner.

## Notes

- `mined_projects/` est ignoré par git.
- `evaluation_results/` est ignoré par git.
- `queries/` est ignoré par git.
- `.env.github` est ignoré par git.
- Le projet privilégie des outputs JSON pour faciliter l’analyse offline et les comparaisons de runs.

## Références Méthodologiques

Documentation et articles utilisés pour guider la stratégie baseline/évaluation:
- [scikit-learn: `precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [scikit-learn: `DummyClassifier`](https://sklearn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
- [SciPy: `scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [NIST: Retrieval Evaluation with Incomplete Information (Buckley & Voorhees, 2004)](https://www.nist.gov/publications/retrieval-evaluation-incomplete-information)
- [NIST TRECVID: Inferred Average Precision (infAP)](https://www-nlpir.nist.gov/projects/tv2006/infAP.html)
