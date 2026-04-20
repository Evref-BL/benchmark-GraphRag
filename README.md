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
│   ├── evaluate_graphrag_benchmark.py
│   └── export_benchmark_queries.py
├── mined_projects/              # sorties JSON (ignoré par git)
├── evaluation_results/          # rapports evaluator (ignoré par git)
├── queries/                     # queries LLM exportées (ignoré par git)
├── .env.github                  # token GitHub local (ignoré par git)
└── README.md
```

## Prérequis

- Python 3.10+
- `uv` (pour lancer GraphRAG)
- Un projet GraphRAG déjà indexé, avec un dossier `output` exploitable.
- Optionnel mais fortement recommandé: un token GitHub (sinon rate limits API très bas).

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
   - `uv run python -m graphrag query "<query>" --root . --method <method> --data ./output --response-type "<valeur par défaut interne du script>"`
4. extrait les classes prédites,
5. compare aux fichiers Java attendus,
6. calcule les métriques par issue et globales.

Le `response_type` n’est pas paramétrable en ligne de commande: il est fixé dans le script pour garantir un format de sortie stable.

### Commande type

```bash
python3 scripts/evaluate_graphrag_benchmark.py \
  mined_projects/stleary__json-java__20260420T090843Z.json \
  --graphrag-dir /path/to/graphrag \
  --method local
```

### Méthode GraphRAG configurable

`--method` accepte uniquement:
- `local`
- `drift`
- `global`

Exemples:

```bash
python3 scripts/evaluate_graphrag_benchmark.py <mined.json> --graphrag-dir /path/to/graphrag --method drift
python3 scripts/evaluate_graphrag_benchmark.py <mined.json> --graphrag-dir /path/to/graphrag --method global
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
   - `python3 scripts/evaluate_graphrag_benchmark.py <mined-file.json> --graphrag-dir <path> --method local`
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

Format de sortie (par entrée):
- `issue_number`
- `query`
- `expected_classes_paths`

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
